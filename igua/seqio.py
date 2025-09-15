import abc
import io
import typing
import pathlib
import tempfile
import gzip
import warnings
import gc

import Bio.Seq
import gb_io
import pandas
import rich.progress

from .mmseqs import MMSeqs
from .mmseqs import Database
from .defense_extractor import DefenseExtractor


_GZIP_MAGIC = b'\x1f\x8b'


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
    def __init__(self) -> None:
        """Initialize the DefenseFinderDataset class."""
        self.defense_metadata: typing.Optional[typing.Union[pathlib.Path, str]] = None  # TSV with paths to defense finder files
        self.defense_systems_tsv: typing.Optional[typing.Union[pathlib.Path, str]] = None  # DefenseFinder output systems TSV
        self.defense_genes_tsv: typing.Optional[typing.Union[pathlib.Path, str]] = None  # DefenseFinder output genes TSV
        self.gff_file: typing.Optional[typing.Union[pathlib.Path, str]] = None  # GFF file (genome annotation)
        self.genome_file: typing.Optional[typing.Union[pathlib.Path, str]] = None  # FASTA file (genome sequence)
        self.protein_file: typing.Optional[typing.Union[pathlib.Path, str]] = None  # protein FASTA file
        self.gene_file: typing.Optional[typing.Union[pathlib.Path, str]] = None  # gene nucleotide FASTA file
        self.write_output: bool = False  # write output files to self.output_dir
        self.output_dir: typing.Optional[typing.Union[pathlib.Path, str]] = None  # output directory for writing files
        self.verbose: bool = False  # verbose output
        self.activity_filter: str = "defense"


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
        temp_dir = None
        if self.write_output and not self.output_dir:
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = pathlib.Path(temp_dir.name)
        else:
            output_dir = self.output_dir
        
        try:
            # Create extractor
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=output_dir,
                write_output=self.write_output, 
                verbose=self.verbose
            )
            
            progress.console.print(f"[bold blue]{'Using':>12}[/] defense metadata file: {self.defense_metadata}")
            try:
                df = pandas.read_csv(self.defense_metadata, sep="\t")
                return self._process_defense_files_from_tsv(progress, df, output, extractor)
            except Exception as e:
                progress.console.print(f"[bold red]{'Error':>12}[/] reading defense metadata: {e}")
                return pandas.DataFrame(columns=["cluster_id", "cluster_length", "filename"]).set_index("cluster_id")
            
            
        finally:
            # clean up temporary directory if created
            if temp_dir:
                temp_dir.cleanup()

    def _process_defense_files_from_tsv(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        extractor: typing.Optional[DefenseExtractor] = None
    ) -> pandas.DataFrame:
        """Process defense systems from a TSV file with paths to defense finder files."""
        # create extractor if not provided
        if extractor is None:
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=self.output_dir,
                write_output=self.write_output, 
                verbose=self.verbose
            )
        
        # process in small chunks to minimize memory usage
        chunk_size = 5
        all_data = []

        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] defense systems", total=len(df))
            
            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                df_chunk = df.iloc[chunk_start:chunk_end]
                
                chunk_data = []
                for _, row in df_chunk.iterrows():
                    strain_id = row.get("strain_id", None)
                    progress.update(task, description=f"[bold blue]{'Processing':>9}[/] strain: [bold cyan]{strain_id}")
                    
                    systems_tsv = pathlib.Path(row["systems_tsv"])
                    genes_tsv = pathlib.Path(row["genes_tsv"])
                    gff_file = pathlib.Path(row["gff_file"])
                    fasta_file = pathlib.Path(row["fasta_file"])
                    
                    missing_files = []
                    for f, name in [
                        (systems_tsv, "systems_tsv"), (genes_tsv, "genes_tsv"),(gff_file, "gff_file"), (fasta_file, "fasta_file")
                    ]:
                        if not f.exists():
                            missing_files.append(f"{name}: {f}")
                    
                    if missing_files:
                        progress.console.print(f"[bold yellow]{'Missing':>12}[/] files for {strain_id}: {', '.join(missing_files)}")
                        progress.update(task, advance=1)
                        continue
                    
                    # extract systems for this strain only
                    strain_output_dir = None
                    if self.write_output and self.output_dir:
                        strain_output_dir = pathlib.Path(self.output_dir) / (strain_id if strain_id else "unknown")
                        strain_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    try:
                        systems = extractor.extract_systems(
                            systems_tsv_file=systems_tsv,
                            genes_tsv_file=genes_tsv,
                            gff_file=gff_file,
                            fasta_file=fasta_file,
                            output_dir=strain_output_dir,
                            strain_id=strain_id,
                            activity_filter=self.activity_filter 
                        )
                        
                        # process systems immediately
                        # don't store sequences
                        for unique_sys_id, system in systems.items():
                            sequence = system["sequence"]
                            length = system["length"]
                            file_path = system.get("file_path", str(fasta_file))
                            
                            self.write_fasta(dst, unique_sys_id, sequence)
                            
                            # only store metadata, not sequences
                            chunk_data.append((unique_sys_id, length, file_path))
                        
                        del systems
                        
                    except Exception as e:
                        progress.console.print(f"[bold red]{'Error':>12}[/] processing {strain_id}: {e}")
                        
                    progress.update(task, advance=1)
                
                all_data.extend(chunk_data)
                del chunk_data, df_chunk
                gc.collect()
            
            progress.remove_task(task)
        
        # create and return DataFrame
        progress.console.print(f"[bold green]{'Extracted':>12}[/] {len(all_data)} systems in total from {df.shape[0]} strains/genomes")
        
        # create pd.DataFrame and immediately clean up data
        result_df = pandas.DataFrame(
            data=all_data,
            columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")
        
        del all_data
        gc.collect()
        
        return result_df

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
        temp_dir = None
        if self.write_output and not self.output_dir:
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = pathlib.Path(temp_dir.name)
        else:
            output_dir = self.output_dir
        
        try:
            # create extractor
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=output_dir,
                write_output=self.write_output, 
                verbose=self.verbose
            )

            progress.console.print(f"[bold blue]{'Using':>12}[/] defense metadata file: {self.defense_metadata}")
            try:
                df = pandas.read_csv(self.defense_metadata, sep="\t")
                return self._extract_proteins_from_tsv(progress, df, output, representatives, extractor)
            except Exception as e:
                progress.console.print(f"[bold red]{'Error':>12}[/] reading defense metadata: {e}")
                return {}
            
        finally:
            # Clean up temporary directory if created
            if temp_dir:
                temp_dir.cleanup()

    def _extract_proteins_from_tsv(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        representatives: typing.Container[str],
        extractor: typing.Optional[DefenseExtractor] = None
    ) -> typing.Dict[str, int]:
        """Extract proteins from defense systems specified in a TSV file."""
        # Create extractor if not provided
        if extractor is None:
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=self.output_dir,
                write_output=self.write_output, 
                verbose=self.verbose
            )
        
        
        # iterate over each genome/strain in DataFrame
        protein_sizes = {}
        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] protein sequences", total=len(df))
            
            for _, row in df.iterrows():
                strain_id = row.get("strain_id", None)
                progress.update(task, description=f"[bold blue]{'Processing':>9}[/] strain: [bold cyan]{strain_id}")
                
                systems_tsv = pathlib.Path(row["systems_tsv"])
                genes_tsv = pathlib.Path(row["genes_tsv"])
                faa_file = pathlib.Path(row["faa_file"])
                fna_file = pathlib.Path(row.get("fna_file", "")) # fna file is optional
                
                missing_files = []
                for f, name in [(systems_tsv, "systems_tsv"), (genes_tsv, "genes_tsv"), (faa_file, "faa_file")]:
                    if not f.exists():
                        missing_files.append(f"{name}: {f}")
                
                if missing_files:
                    progress.console.print(f"[bold yellow]{'Missing':>12}[/] files for {strain_id}: {', '.join(missing_files)}")
                    progress.update(task, advance=1)
                    raise FileNotFoundError(f"Missing required files: {', '.join(missing_files)}")
                
                # extract gene sequences
                strain_output_dir = None
                if self.write_output and self.output_dir:
                    strain_output_dir = self.output_dir / (strain_id if strain_id else "unknown") / "proteins"
                    strain_output_dir.mkdir(parents=True, exist_ok=True)
                
                # all systems' proteins extracted 
                gene_data = extractor.extract_gene_sequences(
                    systems_tsv_file=systems_tsv,
                    genes_tsv_file=genes_tsv,
                    faa_file=faa_file,
                    fna_file=fna_file if fna_file.exists() else None,
                    output_dir=strain_output_dir,
                    strain_id=strain_id, 
                    activity_filter=self.activity_filter
                )
                
                # write proteins to output file and record sizes
                # filter by representatives at this stage 
                for sys_id, system in gene_data.items():
                    
                    # skip if not in representatives 
                    if representatives and sys_id not in representatives:
                        continue

                    # write proteins from representative clusters
                    for prot_id, protein in system.get("proteins", {}).items():
                        # unique protein ID already created by DefenseExtractor
                        unique_protein_id = protein.get("unique_protein_id", f"{sys_id}@@{prot_id}")
                        sequence = protein["sequence"]
                        
                        # write to output
                        self.write_fasta(dst, unique_protein_id, sequence)
                        
                        # record size
                        protein_sizes[unique_protein_id] = len(sequence)

                        # debug 
                        # progress.console.print(f"[bold #ff875f]{'Writing':>12}[/] {seq_id}")
                
                progress.update(task, advance=1)
            
            progress.remove_task(task)
        
        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(representatives)} representative systems"
            )
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
            gff (typing.Optional[typing.Union[str, pathlib.Path]], optional): pathlib.Path to the GFF file. Defaults to None.
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