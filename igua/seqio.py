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
        """Extracts nucleotide sequences from DefenseFinder output files.

        Args:
            progress (rich.progress.Progress): Progress bar for tracking progress.
            inputs (typing.List[pathlib.Path]): List of input files. Expects a TSV with paths to FNA files.
            output (pathlib.Path): Output file path for the extracted sequences.

        Returns:
            pandas.DataFrame: DataFrame containing the extracted sequences.
        """
        # Check if input is a TSV file containing paths to FNA files
        if len(inputs) == 1 and str(inputs[0]).endswith(".tsv"):
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                # Check if the TSV contains the expected columns
                if "fna_file" in df.columns and "tsv_file" in df.columns:
                    progress.console.print(
                        f"[bold blue]{'Found':>12}[/] {len(df)} FNA files to process"
                    )
                    return self._process_existing_fna_files(progress, df, output)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] Input TSV is missing required columns. Expected 'fna_file' and 'tsv_file'"
                    )
            except Exception as e:
                progress.console.print(
                    f"[bold red]{'Error':>12}[/] reading input TSV: {e}"
                )

        # If not a valid DefenseFinder input or processing failed, return empty DataFrame
        progress.console.print(
            f"[bold red]{'Error':>12}[/] DefenseFinderDataset requires a TSV file with paths to FNA files"
        )
        return pandas.DataFrame(
            columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")


    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
    ) -> typing.Dict[str, int]:
        """Extracts protein sequences from DefenseFinder output files.

        Args:
            progress (rich.progress.Progress): Progress bar for tracking progress.
            inputs (typing.List[pathlib.Path]): List of input files. Expects a TSV with paths to FAA files.
            output (pathlib.Path): Output file path for the extracted protein sequences.
            representatives (typing.Container[str]): Set of representative cluster IDs.

        Returns:
            typing.Dict[str, int]: Dictionary containing protein IDs and their sizes.
        """
        # Check if input is a TSV file containing paths to FAA files
        if len(inputs) == 1 and str(inputs[0]).endswith(".tsv"):
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                # Check if the TSV contains the expected columns
                if "faa_file" in df.columns and "tsv_file" in df.columns:
                    progress.console.print(
                        f"[bold blue]{'Found':>12}[/] {len(df)} FAA files to process"
                    )
                    return self._process_existing_faa_files(progress, df, output, representatives)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] Input TSV is missing required columns. Expected 'faa_file' and 'tsv_file'"
                    )
            except Exception as e:
                progress.console.print(
                    f"[bold red]{'Error':>12}[/] reading input TSV: {e}"
                )

        # If not a valid DefenseFinder input or processing failed, return empty dictionary
        progress.console.print(
            f"[bold red]{'Error':>12}[/] DefenseFinderDataset requires a TSV file with paths to FAA files"
        )
        return {}

    def _process_existing_fna_files(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Process existing FNA files from defense finder output.

        Args:
            progress: Progress bar for tracking.
            df: DataFrame containing paths to FNA files.
            output: Output path for combined sequences.

        Returns:
            DataFrame with the same format as GenBankDataset.extract_sequences.
        """
        data = []
        done = set()
        n_duplicate = 0

        # Create a task for overall progress
        task1 = progress.add_task(
            f"[bold blue]{'Processing':>9}[/] FNA files", total=len(df)
        )

        with open(output, "w") as dst:
            # Process each FNA file
            for _, row in df.iterrows():
                fna_file = pathlib.Path(row["fna_file"])
                tsv_file = pathlib.Path(row["tsv_file"])

                # Read defense finder TSV to get hit IDs
                progress.update(
                    task1, description=f"[bold blue]{'Reading':>9}[/] {tsv_file.name}"
                )
                try:
                    tsv_data = pandas.read_csv(tsv_file, sep="\t")
                    hit_ids = (
                        set(tsv_data["hit_id"].unique())
                        if "hit_id" in tsv_data.columns
                        else set()
                    )
                except Exception as e:
                    progress.console.print(
                        f"[bold red]{'Error':>12}[/] reading {tsv_file}: {e}"
                    )
                    continue

                if not hit_ids:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] No hit IDs found in {tsv_file}"
                    )
                    continue

                # Create a subtask for reading the FNA file
                task2 = progress.add_task(
                    f"[bold blue]{'Reading':>9}[/] {fna_file.name}", total=1
                )

                try:
                    # Check if file is gzipped
                    is_gzipped = False
                    with open(fna_file, "rb") as test_f:
                        if test_f.read(2) == _GZIP_MAGIC:
                            is_gzipped = True

                    # Open file based on compression
                    open_func = gzip.open if is_gzipped else open
                    with open_func(fna_file, "rt") as src:
                        # Parse FASTA format
                        current_id = None
                        current_seq = []

                        for line in src:
                            line = line.strip()
                            if not line:
                                continue

                            if line.startswith(">"):
                                # Process previous sequence if we were collecting one
                                if current_id is not None and current_seq:
                                    # Check if this is a hit we're interested in
                                    seq_id = current_id.split()[0].lstrip(">")
                                    if any(hit_id in seq_id for hit_id in hit_ids):
                                        if seq_id in done:
                                            n_duplicate += 1
                                        else:
                                            sequence = "".join(current_seq)
                                            self.write_fasta(dst, seq_id, sequence)
                                            data.append(
                                                (seq_id, len(sequence), str(fna_file))
                                            )
                                            done.add(seq_id)

                                # Start new sequence
                                current_id = line
                                current_seq = []
                            else:
                                # Continue collecting sequence
                                current_seq.append(line)

                        # Don't forget the last sequence
                        if current_id is not None and current_seq:
                            seq_id = current_id.split()[0].lstrip(">")
                            if any(hit_id in seq_id for hit_id in hit_ids):
                                if seq_id in done:
                                    n_duplicate += 1
                                else:
                                    sequence = "".join(current_seq)
                                    self.write_fasta(dst, seq_id, sequence)
                                    data.append((seq_id, len(sequence), str(fna_file)))
                                    done.add(seq_id)

                except Exception as e:
                    progress.console.print(
                        f"[bold red]{'Error':>12}[/] processing {fna_file}: {e}"
                    )

                finally:
                    progress.remove_task(task2)
                    progress.update(task1, advance=1)

        # Clean up
        progress.remove_task(task1)

        if n_duplicate > 0:
            progress.console.print(
                f"[bold yellow]{'Skipped':>12}[/] {n_duplicate} clusters with duplicate identifiers"
            )

        if not data:
            progress.console.print(
                f"[bold red]{'Warning':>12}[/] No sequences were extracted from the input files"
            )
        else:
            progress.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(data)} sequences from {len(df)} input files"
            )

        # Return DataFrame in the same format as GenBankDataset.extract_sequences
        return pandas.DataFrame(
            data=data, columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")

    # def _process_existing_fna_files(
    #     self,
    #     progress: rich.progress.Progress,
    #     df: pandas.DataFrame,
    #     output: pathlib.Path
    # ) -> pandas.DataFrame:
    #     """
    #     Process existing FNA files from the defense finder output and combine them.
        
    #     Args:
    #         progress: Progress bar for tracking.
    #         df: DataFrame containing paths to FNA files.
    #         output: Output path for combined nucleotide sequences.
            
    #     Returns:
    #         DataFrame with cluster_id, cluster_length, and filename information.
    #     """
    #     result_data = []
        
    #     # Create a task for overall progress
    #     task = progress.add_task(
    #         f"[bold blue]{'Processing':>9}[/] FNA files", total=len(df)
    #     )
        
    #     # Open the output file to concatenate all sequences
    #     with open(output, "w") as out_file:
    #         for idx, row in df.iterrows():
    #             fna_file = pathlib.Path(row["fna_file"])
                
    #             progress.update(
    #                 task, description=f"[bold blue]{'Processing':>9}[/] {fna_file.name}"
    #             )
                
    #             try:
    #                 # Check if file is gzipped
    #                 is_gzipped = False
    #                 with open(fna_file, "rb") as test_f:
    #                     if test_f.read(2) == b"\x1f\x8b":  # GZIP magic number
    #                         is_gzipped = True
                    
    #                 # Open file based on compression
    #                 open_func = gzip.open if is_gzipped else open
                    
    #                 # Read and process this FNA file
    #                 with open_func(fna_file, "rt") as src:
    #                     sequences = []
    #                     current_id = None
    #                     current_seq = []
                        
    #                     for line in src:
    #                         line = line.strip()
    #                         if not line:
    #                             continue
                                
    #                         if line.startswith(">"):
    #                             # Process previous sequence if there was one
    #                             if current_id is not None and current_seq:
    #                                 sequence = "".join(current_seq)
    #                                 sequences.append((current_id, sequence))
                                    
    #                                 # Write to output file
    #                                 out_file.write(f"{current_id}\n")
    #                                 out_file.write(f"{sequence}\n")
                                
    #                             # Start new sequence
    #                             current_id = line
    #                             current_seq = []
    #                         else:
    #                             current_seq.append(line)
                        
    #                     # Don't forget the last sequence
    #                     if current_id is not None and current_seq:
    #                         sequence = "".join(current_seq)
    #                         sequences.append((current_id, sequence))
                            
    #                         # Write to output file
    #                         out_file.write(f"{current_id}\n")
    #                         out_file.write(f"{sequence}\n")
                    
    #                 # Generate cluster info for this file
    #                 cluster_id = fna_file.stem
    #                 result_data.append({
    #                     "cluster_id": cluster_id,
    #                     "cluster_length": sum(len(seq) for _, seq in sequences),
    #                     "filename": str(fna_file)
    #                 })
                    
    #             except Exception as e:
    #                 progress.console.print(
    #                     f"[bold red]{'Error':>12}[/] processing {fna_file}: {e}"
    #                 )
                
    #             progress.update(task, advance=1)
        
    #     progress.remove_task(task)
        
    #     # Create DataFrame with required columns
    #     result_df = pandas.DataFrame(result_data)
    #     if not result_df.empty:
    #         result_df.set_index("cluster_id", inplace=True)
        
    #     progress.console.print(
    #         f"[bold green]{'Processed':>12}[/] {len(result_df)} clusters from FNA files"
    #     )
        
    #     return result_df



    def _process_existing_faa_files(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        representatives: typing.Container[str]
    ) -> typing.Dict[str, int]:
        """Process existing FAA files from defense finder output.

        Args:
            progress: Progress bar for tracking.
            df: DataFrame containing paths to FAA files.
            output: Output path for combined protein sequences.
            representatives: Set of representative cluster IDs.

        Returns:
            Dictionary mapping protein IDs to their sequence lengths.
        """
        protein_sizes = {}
        
        # Create a task for overall progress
        task1 = progress.add_task(
            f"[bold blue]{'Processing':>9}[/] FAA files", total=len(df)
        )

        with open(output, "w") as dst:
            # Process each FAA file
            for _, row in df.iterrows():
                faa_file = pathlib.Path(row["faa_file"])
                tsv_file = pathlib.Path(row.get("tsv_file", ""))
                
                # Read defense finder TSV to get hit IDs if available
                relevant_hit_ids = set()
                if tsv_file and tsv_file.exists():
                    progress.update(
                        task1, description=f"[bold blue]{'Reading':>9}[/] {tsv_file.name}"
                    )
                    try:
                        tsv_data = pandas.read_csv(tsv_file, sep="\t")
                        if "hit_id" in tsv_data.columns:
                            relevant_hit_ids = set(tsv_data["hit_id"].unique())
                    except Exception as e:
                        progress.console.print(
                            f"[bold yellow]{'Warning':>12}[/] Error reading {tsv_file}: {e}"
                        )
                
                # Create a subtask for reading the FAA file
                task2 = progress.add_task(
                    f"[bold blue]{'Reading':>9}[/] {faa_file.name}", total=1
                )

                try:
                    # Check if file is gzipped
                    is_gzipped = False
                    with open(faa_file, "rb") as test_f:
                        if test_f.read(2) == _GZIP_MAGIC:
                            is_gzipped = True

                    # Open file based on compression
                    open_func = gzip.open if is_gzipped else open
                    with open_func(faa_file, "rt") as src:
                        # Parse FASTA format
                        current_id = None
                        current_seq = []

                        for line in src:
                            line = line.strip()
                            if not line:
                                continue

                            if line.startswith(">"):
                                # Process previous sequence if we were collecting one
                                if current_id is not None and current_seq:
                                    # Extract the ID without the ">"
                                    seq_id = current_id.split()[0].lstrip(">")
                                    
                                    # Check if this is from a representative cluster or relevant hit
                                    cluster_id = seq_id.split("_")[0] if "_" in seq_id else seq_id
                                    is_representative = cluster_id in representatives
                                    is_relevant_hit = relevant_hit_ids and seq_id in relevant_hit_ids
                                    
                                    if is_representative or is_relevant_hit:
                                        sequence = "".join(current_seq)
                                        if seq_id not in protein_sizes:  # Avoid duplicates
                                            self.write_fasta(dst, seq_id, sequence)
                                            protein_sizes[seq_id] = len(sequence)

                                # Start new sequence
                                current_id = line
                                current_seq = []
                            else:
                                # Continue collecting sequence
                                current_seq.append(line)

                        # Don't forget the last sequence
                        if current_id is not None and current_seq:
                            seq_id = current_id.split()[0].lstrip(">")
                            cluster_id = seq_id.split("_")[0] if "_" in seq_id else seq_id
                            
                            is_representative = cluster_id in representatives
                            is_relevant_hit = relevant_hit_ids and seq_id in relevant_hit_ids
                            
                            if is_representative or is_relevant_hit:
                                sequence = "".join(current_seq)
                                if seq_id not in protein_sizes:  # Avoid duplicates
                                    self.write_fasta(dst, seq_id, sequence)
                                    protein_sizes[seq_id] = len(sequence)

                except Exception as e:
                    progress.console.print(
                        f"[bold red]{'Error':>12}[/] processing {faa_file}: {e}"
                    )

                finally:
                    progress.remove_task(task2)
                    progress.update(task1, advance=1)

        # Clean up
        progress.remove_task(task1)

        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from representatives"
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
