import abc
import io
import typing
import pathlib
import gzip
import warnings

import Bio.Seq
import gb_io
import pandas
import rich.progress

from .mmseqs import MMSeqs
from .mmseqs import Database

_GZIP_MAGIC = b'\x1f\x8b'


# TODO: implement extract_sequences and extract_proteins methods for defense-finder output
# TODO: check gff files for

class BaseDataset(abc.ABC):
    """Base class for dataset extraction.
    This class defines the basic structure and methods for extracting nucleotide and
    protein sequences from various file formats. It serves as a base class for specific 
    dataset classes like GenBankDataset and GFFDataset.
    """
    @abc.abstractmethod
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        ) -> pandas.DataFrame:
        pass
    
    @abc.abstractmethod
    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
        ) -> typing.Dict[str, int]:
        pass
    
    def write_fasta(self, file: typing.TextIO, name: str, sequence: str) -> None:
        file.write(">{}\n".format(name))
        file.write(sequence)
        file.write("\n")
        return None

    def translate_orf(self, sequence: typing.Union[str, bytes], translation_table: int = 11) -> str:
        return str(Bio.Seq.Seq(sequence).translate(translation_table))

    
    def create_sequence_database(
        self, 
        mmseqs: MMSeqs, 
        progress: rich.progress.Progress, 
        inputs: typing.List[pathlib.Path],
        output_db_path: pathlib.Path
    ) -> Database:
        """Default implementation creates a temporary file then a database"""
        # Create temporary FASTA file
        tmp_fasta = output_db_path.with_suffix(".fna")
        self.extract_sequences(progress, inputs, tmp_fasta)
        # Create database from temporary file
        return Database.create(mmseqs, tmp_fasta)
    
    def create_protein_database(
        self, 
        mmseqs: MMSeqs,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        representatives: typing.Container[str],
        output_db_path: pathlib.Path
    ) -> typing.Tuple[Database, typing.Dict[str, int]]:
        """Default implementation creates a temporary file then a database"""
        # Create temporary FASTA file
        tmp_fasta = output_db_path.with_suffix(".faa")
        protein_sizes = self.extract_proteins(progress, inputs, tmp_fasta, representatives)
        # Create database from temporary file
        return Database.create(mmseqs, tmp_fasta), protein_sizes


class GenBankDataset(BaseDataset):
    """GenBank dataset class.
    This class is used to extract nucleotide and protein sequences from GenBank files.
    It inherits from the BaseDataset class and implements the extract_sequences
    and extract_proteins methods.
    """
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Extracts nucleotide sequences from GenBank files.
        Args:
            progress (rich.progress.Progress): Progress bar for tracking progress.
            inputs (typing.List[pathlib.Path]): List of input GenBank files.
            output (pathlib.Path): Output file path for the extracted sequences. 
        Returns:
            pandas.DataFrame: DataFrame containing the extracted sequences.
        """
        data = []
        done = set()
        n_duplicate = 0
        with open(output, "w") as dst:
            task1 = progress.add_task(f"[bold blue]{'Working':>9}[/]")
            for input_path in progress.track(inputs, task_id=task1):
                task2 = progress.add_task(f"[bold blue]{'Reading':>9}[/]")
                with io.BufferedReader(progress.open(input_path, "rb", task_id=task2)) as reader:  # type: ignore
                    if reader.peek().startswith(_GZIP_MAGIC):
                        reader = gzip.GzipFile(mode="rb", fileobj=reader)  # type: ignore
                    for record in gb_io.iter(reader):
                        if record.name in done:
                            n_duplicate += 1
                        else:
                            self.write_fasta(dst, record.name, record.sequence.decode("ascii"))
                            data.append((record.name, len(record.sequence), input_path))
                            done.add(record.name)
                progress.remove_task(task2)
            progress.remove_task(task1)
        if n_duplicate > 0:
            progress.console.print(
                f"[bold yellow]{'Skipped':>12}[/] {n_duplicate} clusters with duplicate identifiers"
            )
        return pandas.DataFrame(
            data=data,
            columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")

    
    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str],
    ) -> typing.Dict[str, int]:
        """Extracts protein sequences from GenBank files.
        Args:
            progress (rich.progress.Progress): Progress bar for tracking progress.
            inputs (typing.List[pathlib.Path]): List of input GenBank files.
            output (pathlib.Path): Output file path for the extracted protein sequences.
            representatives (typing.Container[str]): Set of representative cluster IDs.
        Returns:
            typing.Dict[str, int]: Dictionary containing protein IDs and their sizes. 
        """
        protein_sizes = {}
        with output.open("w") as dst:
            for input_path in inputs:
                task = progress.add_task(f"[bold blue]{'Reading':>9}[/]")
                with io.BufferedReader(progress.open(input_path, "rb", task_id=task)) as reader:  # type: ignore
                    if reader.peek()[:2] == b'\x1f\x8b':
                        reader = gzip.GzipFile(mode="rb", fileobj=reader)  # type: ignore
                    for record in gb_io.iter(reader):
                        if record.name in representatives:
                            for i, feat in enumerate(
                                filter(lambda f: f.kind == "CDS", record.features)
                            ):
                                qualifier = next((qualifier for qualifier in feat.qualifiers if qualifier.key == "translation"), None)
                                if qualifier is None:
                                    rich.print(f"[bold yellow]{'Warning':>12}[/] no 'translation' qualifier found in CDS feature of {record.name!r}")
                                    translation = self.translate_orf(record.sequence[feat.location.start:feat.location.end])
                                else:
                                    translation = qualifier.value.rstrip("*")
                                protein_id = "{}_{}".format(record.name, i)
                                if protein_id not in protein_sizes:
                                    self.write_fasta(dst, protein_id, translation)
                                    protein_sizes[protein_id] = len(translation)
                progress.remove_task(task)
        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(representatives)} nucleotide representative"
        )
        return protein_sizes


class DefenseFinderDataset(BaseDataset):
    """DefenseFinder dataset class.
    This class is used to extract nucleotide and protein sequences from DefenseFinder output files.
    """
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Extracts nucleotide sequences from defense system FASTA files.
        
        Args:
            progress: Progress bar for tracking.
            inputs: List of input files (FASTA files or TSV summary).
            output: Output path for combined sequences.
            
        Returns:
            DataFrame with cluster_id, cluster_length, filename
        """
        # Handle TSV summary file
        if len(inputs) == 1 and inputs[0].suffix.lower() == ".tsv":
            progress.console.print(f"[bold blue]{'Found':>12}[/] TSV summary file")
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                # First check for fasta_file column (genome sequences)
                if "fasta_file" in df.columns:
                    fa_files = [pathlib.Path(f) for f in df["fasta_file"] if f]
                    progress.console.print(f"[bold blue]{'Using':>12}[/] fasta_file column with {len(fa_files)} files")
                    return self._process_fa_files(progress, fa_files, output)
                # Fall back to genomic_file for backward compatibility
                elif "genomic_file" in df.columns:
                    fa_files = [pathlib.Path(f) for f in df["genomic_file"] if f]
                    progress.console.print(f"[bold blue]{'Using':>12}[/] genomic_file column with {len(fa_files)} files")
                    return self._process_fa_files(progress, fa_files, output)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] TSV missing required columns"
                    )
            except Exception as e:
                progress.console.print(f"[bold red]{'Error':>12}[/] reading TSV: {e}")


        # Handle directory with FASTA files
        elif len(inputs) == 1 and inputs[0].is_dir():
            fa_files = list(inputs[0].glob("**/*.fna"))
            progress.console.print(f"[bold blue]{'Found':>12}[/] {len(fa_files)} FASTA files")
            return self._process_fa_files(progress, fa_files, output)
            
        # Handle direct list of FASTA files
        elif all(f.suffix.lower() in (".fa", ".fna", ".fasta") for f in inputs):
            progress.console.print(f"[bold blue]{'Found':>12}[/] {len(inputs)} FASTA files")
            return self._process_fa_files(progress, inputs, output)
            
        # Return empty DataFrame if no valid input
        progress.console.print(f"[bold red]{'Error':>12}[/] No valid input files")
        return pandas.DataFrame(columns=["cluster_id", "cluster_length", "filename"]).set_index("cluster_id")
    
    def _process_fa_files(
        self,
        progress: rich.progress.Progress,
        fa_files: typing.List[pathlib.Path],
        output: pathlib.Path
    ) -> pandas.DataFrame:
        """Process FASTA files containing defense system sequences.
        
        Args:
            progress: Progress bar for tracking.
            fa_files: List of FASTA files.
            output: Output path for combined sequences.
            
        Returns:
            DataFrame with cluster_id, cluster_length, filename
        """
        data = []
        done = set()
        n_duplicate = 0
        
        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] FASTA files", total=len(fa_files))
            
            for fa_file in fa_files:
                if not fa_file.exists():
                    progress.console.print(f"[bold yellow]{'Missing':>12}[/] {fa_file}")
                    progress.update(task, advance=1)
                    continue
                
                # Get system ID from filename
                sys_id = fa_file.stem
                progress.update(task, description=f"[bold blue]{'Reading':>9}[/] {sys_id}")
                
                try:
                    # Check for gzip
                    is_gzipped = False
                    with open(fa_file, "rb") as test_f:
                        if test_f.read(2) == _GZIP_MAGIC:
                            is_gzipped = True
                    
                    # Process file
                    open_func = gzip.open if is_gzipped else open
                    with open_func(fa_file, "rt") as src:
                        seq_id = None
                        seq_parts = []
                        total_length = 0
                        
                        for line in src:
                            line = line.strip()
                            if not line:
                                continue
                                
                            if line.startswith(">"):
                                # Process previous sequence
                                if seq_id and seq_parts:
                                    sequence = "".join(seq_parts)
                                    self.write_fasta(dst, seq_id, sequence)
                                    total_length += len(sequence)
                                
                                # Start new sequence
                                seq_id = sys_id
                                # header = line[1:].strip()
                                # seq_id = f"{sys_id}_{header.split()[0]}"
                                seq_parts = []
                            else:
                                seq_parts.append(line)
                        
                        # Process final sequence
                        if seq_id and seq_parts:
                            sequence = "".join(seq_parts)
                            self.write_fasta(dst, seq_id, sequence)
                            total_length += len(sequence)
                    
                    # Record system data
                    if sys_id not in done:
                        data.append((sys_id, total_length, str(fa_file)))
                        done.add(sys_id)
                    else:
                        n_duplicate += 1
                        
                except Exception as e:
                    progress.console.print(f"[bold red]{'Error':>12}[/] processing {fa_file}: {e}")
                    
                finally:
                    progress.update(task, advance=1)
            
            progress.remove_task(task)
        
        # Report results
        if n_duplicate > 0:
            progress.console.print(f"[bold yellow]{'Skipped':>12}[/] {n_duplicate} duplicate systems")
            
        if data:
            progress.console.print(f"[bold green]{'Extracted':>12}[/] {len(data)} systems")
        else:
            progress.console.print(f"[bold red]{'Warning':>12}[/] No sequences extracted")
            
        return pandas.DataFrame(
            data=data,
            columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")
    
    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
    ) -> typing.Dict[str, int]:
        """Extracts protein sequences from defense system protein files.
        
        Args:
            progress: Progress bar for tracking.
            inputs: List of input files (protein FASTA files or TSV summary).
            output: Output path for combined sequences.
            representatives: Set of representative cluster IDs.
            
        Returns:
            Dictionary mapping protein IDs to their lengths.
        """
        # Handle TSV summary file
        if len(inputs) == 1 and inputs[0].suffix.lower() == ".tsv":
            progress.console.print(f"[bold blue]{'Found':>12}[/] TSV summary file")
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                if "faa_file" in df.columns:
                    # Filter to representatives if provided
                    if representatives:
                        str_reps = set(str(r) for r in representatives)
                        progress.console.print(
                            f"[bold blue]{'Filtering':>12}[/] to {len(str_reps)} representatives"
                        )
                        sys_ids = [id for id in df["system_id"].astype(str) if id in "\t".join(str_reps)]
                        df = df[df["system_id"].astype(str).isin(sys_ids)]
                    
                    faa_files = [pathlib.Path(f) for f in df["faa_file"] if f]
                    return self._process_faa_files(progress, faa_files, output, representatives)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] TSV missing 'faa_file' column"
                    )
            except Exception as e:
                progress.console.print(f"[bold red]{'Error':>12}[/] reading TSV: {e}")
                
        # Handle directory with protein files
        elif len(inputs) == 1 and inputs[0].is_dir():
            faa_files = list(inputs[0].glob("**/*.faa"))
            progress.console.print(f"[bold blue]{'Found':>12}[/] {len(faa_files)} protein files")
            return self._process_faa_files(progress, faa_files, output, representatives)
            
        # Handle direct list of protein files
        elif all(f.suffix.lower() in (".faa", ".fa", ".fasta") for f in inputs):
            progress.console.print(f"[bold blue]{'Found':>12}[/] {len(inputs)} protein files")
            return self._process_faa_files(progress, inputs, output, representatives)
            
        # Return empty dictionary if no valid input
        progress.console.print(f"[bold red]{'Error':>12}[/] No valid protein input files")
        return {}
    
    def _process_faa_files(
        self,
        progress: rich.progress.Progress,
        faa_files: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Optional[typing.Container[str]] = None
    ) -> typing.Dict[str, int]:
        """Process protein FASTA files.
        
        Args:
            progress: Progress bar for tracking.
            faa_files: List of protein FASTA files.
            output: Output path for combined sequences.
            representatives: Set of representative cluster IDs (unused, filtering done earlier).
            
        Returns:
            Dictionary mapping protein IDs to their lengths.
        """
        protein_sizes = {}
        
        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] protein files", total=len(faa_files))
            
            for faa_file in faa_files:
                if not faa_file.exists():
                    progress.console.print(f"[bold yellow]{'Missing':>12}[/] {faa_file}")
                    progress.update(task, advance=1)
                    continue
                
                # Get system ID from filename
                sys_id = faa_file.stem
                progress.update(task, description=f"[bold blue]{'Reading':>9}[/] {sys_id}")
                
                try:
                    # Check for gzip
                    is_gzipped = False
                    with open(faa_file, "rb") as test_f:
                        if test_f.read(2) == _GZIP_MAGIC:
                            is_gzipped = True
                    
                    # Process file
                    open_func = gzip.open if is_gzipped else open
                    with open_func(faa_file, "rt") as src:
                        seq_id = None
                        seq_parts = []
                        
                        for line in src:
                            line = line.strip()
                            if not line:
                                continue
                                
                            if line.startswith(">"):
                                # Process previous sequence
                                if seq_id and seq_parts:
                                    sequence = "".join(seq_parts)
                                    self.write_fasta(dst, seq_id, sequence)
                                    protein_sizes[seq_id] = len(sequence)
                                
                                # Start new sequence
                                # # Format as system_id_protein_id for uniqueness
                                # # Extract protein ID from the FASTA header
                                # header = line[1:].strip()
                                # protein_id = header.split()[0]  # Get first word of header as protein identifier
                                # seq_id = f"{sys_id}_{protein_id}"
                                # seq_parts = []
                                seq_id = sys_id
                                seq_parts = []
                            else:
                                seq_parts.append(line)
                        
                        # Process final sequence
                        if seq_id and seq_parts:
                            sequence = "".join(seq_parts)
                            self.write_fasta(dst, seq_id, sequence)
                            protein_sizes[seq_id] = len(sequence)
                        
                except Exception as e:
                    progress.console.print(f"[bold red]{'Error':>12}[/] processing {faa_file}: {e}")
                    
                finally:
                    progress.update(task, advance=1)
            
            progress.remove_task(task)
        
        # Report results
        if protein_sizes:
            progress.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(faa_files)} files"
            )
        else:
            progress.console.print(f"[bold red]{'Warning':>12}[/] No protein sequences extracted")
            
        return protein_sizes



class GFFDataset(BaseDataset):
    """GFF dataset class.
    This class is used to extract nucleotide and protein sequences from GFF files.
    It inherits from the BaseDataset class and implements the extract_sequences
    and extract_proteins methods.
    """
    def __init__(self, gff: typing.Optional[typing.Union[str, pathlib.Path]] = None):
        """Initializes the GFFDataset class.

        Args:
            gff (typing.Optional[typing.Union[str, pathlib.Path]], optional): Path to the GFF file. Defaults to None.
        """
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        ) -> pandas.DataFrame: 
        raise NotImplementedError("extract_sequences method is not implemented for GFFDataset")

    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
        ) -> typing.Dict[str, int]:
        raise NotImplementedError("extract_proteins method is not implemented for GFFDataset")
