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

    def _process_existing_fna_files(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path
    ) -> pandas.DataFrame:
        """Process existing FNA files from defense finder output.

        Args:
            progress: Progress bar for tracking.
            df: DataFrame containing paths to FNA files.
            output: Output path for combined sequences.

        Returns:
            DataFrame with cluster_id, cluster_length, filename
        """
        column_name = "nucleotide_file"
        data = []
        done = set()
        n_duplicate = 0
        
        # Filter out rows with missing files
        df = df[df[column_name].notna()].copy()
        if df.empty:
            progress.console.print(
                f"[bold yellow]{'Warning':>12}[/] No valid FNA files found in input"
            )
            return pandas.DataFrame(
                columns=["cluster_id", "cluster_length", "filename"]
            ).set_index("cluster_id")

        # Create a task for overall progress
        task1 = progress.add_task(
            f"[bold blue]{'Processing':>9}[/] FNA files", total=len(df)
        )

        with open(output, "w") as dst:
            # Process each file
            for _, row in df.iterrows():
                file_path = pathlib.Path(row[column_name])
                if not file_path.exists():
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] FNA file not found: {file_path}"
                    )
                    progress.update(task1, advance=1)
                    continue
                    
                # Use system_id or sys_id as cluster identifier
                sys_id = row.get("system_id", row.get("sys_id", "unknown"))
                progress.update(
                    task1, description=f"[bold blue]{'Reading':>9}[/] {sys_id}"
                )

                try:
                    # Check if file is gzipped
                    is_gzipped = False
                    with open(file_path, "rb") as test_f:
                        first_bytes = test_f.read(2)
                        if first_bytes == _GZIP_MAGIC:
                            is_gzipped = True

                    # Open file based on compression
                    open_func = gzip.open if is_gzipped else open
                    with open_func(file_path, "rt") as src:
                        # Parse FASTA format
                        current_id = None
                        current_seq = []
                        total_length = 0

                        for line in src:
                            line = line.strip()
                            if not line:
                                continue

                            if line.startswith(">"):
                                # Process previous sequence if we were collecting one
                                if current_id is not None and current_seq:
                                    sequence = "".join(current_seq)
                                    seq_length = len(sequence)
                                    total_length += seq_length
                                    
                                    # For nucleotide files, use the system ID
                                    if sys_id not in done:
                                        self.write_fasta(dst, current_id, sequence)
                                    else:
                                        n_duplicate += 1

                                # Start new sequence
                                current_id = line.lstrip(">").split()[0]
                                current_seq = []
                            else:
                                # Continue collecting sequence
                                current_seq.append(line)

                        # Don't forget the last sequence
                        if current_id is not None and current_seq:
                            sequence = "".join(current_seq)
                            seq_length = len(sequence)
                            total_length += seq_length
                            
                            # For nucleotide files, use the system ID
                            if sys_id not in done:
                                self.write_fasta(dst, current_id, sequence)
                    
                    # Record this system in the results
                    if sys_id not in done:
                        data.append((sys_id, total_length, str(file_path)))
                        done.add(sys_id)

                except Exception as e:
                    progress.console.print(
                        f"[bold red]{'Error':>12}[/] processing {file_path}: {e}"
                    )

                finally:
                    progress.update(task1, advance=1)

        # Clean up
        progress.remove_task(task1)

        if n_duplicate > 0:
            progress.console.print(
                f"[bold yellow]{'Skipped':>12}[/] {n_duplicate} duplicate sequences"
            )

        if not data:
            progress.console.print(
                f"[bold red]{'Warning':>12}[/] No sequences were extracted from the input files"
            )
        else:
            progress.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(data)} systems from {len(df)} input files"
            )
        
        # Return DataFrame in the same format as GenBankDataset.extract_sequences
        return pandas.DataFrame(
            data=data, columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")

    def _process_existing_faa_files(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        representatives: typing.Optional[typing.Container[str]] = None
    ) -> typing.Dict[str, int]:
        """Process existing FAA files from defense finder output.

        Args:
            progress: Progress bar for tracking.
            df: DataFrame containing paths to FAA files.
            output: Output path for combined sequences.
            representatives: Set of representative cluster IDs.

        Returns:
            Dictionary mapping protein IDs to their sequence lengths.
        """
        column_name = "protein_file"
        protein_sizes = {}
        
        # Filter out rows with missing files
        df = df[df[column_name].notna()].copy()
        if df.empty:
            progress.console.print(
                f"[bold yellow]{'Warning':>12}[/] No valid FAA files found in input"
            )
            return {}

        # Filter to only include representatives if provided
        if representatives and len(representatives) > 0:
            original_len = len(df)
            
            # Convert representatives to strings for consistent comparison
            str_representatives = set(str(r) for r in representatives)
            progress.console.print(
                f"[bold blue]{'Filtering':>12}[/] systems to {len(str_representatives)} representatives"
            )
            
            # Create a filtered dataframe with only systems that are in the representatives
            filtered_df = df[
                df["system_id"].astype(str).isin(str_representatives) | 
                df.get("system_id", pandas.Series("", index=df.index)).astype(str).isin(str_representatives)
            ].copy()
            
            # If we found matches, use the filtered dataframe
            if not filtered_df.empty:
                df = filtered_df
                progress.console.print(
                    f"[bold blue]{'Filtered':>12}[/] to {len(df)} representative systems out of {original_len}"
                )
            else:
                # If no matches found, fall back to using all systems
                progress.console.print(
                    f"[bold yellow]{'Warning':>12}[/] No representative matches found. Using all systems."
                )

        # Create a task for overall progress
        task1 = progress.add_task(
            f"[bold blue]{'Processing':>9}[/] FAA files", total=len(df)
        )

        with open(output, "w") as dst:
            # Process each file
            for _, row in df.iterrows():
                file_path = pathlib.Path(row[column_name])
                if not file_path.exists():
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] FAA file not found: {file_path}"
                    )
                    progress.update(task1, advance=1)
                    continue
                    
                # Use system_id or sys_id as cluster identifier
                sys_id = row.get("system_id", row.get("sys_id", "unknown"))
                
                # Skip if not in representatives (double-check after filtering dataframe)
                if representatives and str(sys_id) not in str_representatives:
                    progress.update(task1, advance=1)
                    continue
                    
                progress.update(
                    task1, description=f"[bold blue]{'Reading':>9}[/] {sys_id}"
                )

                try:
                    # Check if file is gzipped
                    is_gzipped = False
                    with open(file_path, "rb") as test_f:
                        first_bytes = test_f.read(2)
                        if first_bytes == _GZIP_MAGIC:
                            is_gzipped = True

                    # Open file based on compression
                    open_func = gzip.open if is_gzipped else open
                    with open_func(file_path, "rt") as src:
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
                                    sequence = "".join(current_seq)
                                    seq_length = len(sequence)
                                    
                                    # Add system ID as prefix to ensure uniqueness across systems
                                    # Format: system_id_protein_id
                                    protein_id = f"{sys_id}_{current_id}"
                                    if protein_id not in protein_sizes:
                                        self.write_fasta(dst, protein_id, sequence)
                                        protein_sizes[protein_id] = seq_length

                                # Start new sequence
                                current_id = line.lstrip(">").split()[0]
                                current_seq = []
                            else:
                                # Continue collecting sequence
                                current_seq.append(line)

                        # Don't forget the last sequence
                        if current_id is not None and current_seq:
                            sequence = "".join(current_seq)
                            seq_length = len(sequence)
                            
                            # Add system ID as prefix to ensure uniqueness
                            protein_id = f"{sys_id}_{current_id}"
                            if protein_id not in protein_sizes:
                                self.write_fasta(dst, protein_id, sequence)
                                protein_sizes[protein_id] = seq_length

                except Exception as e:
                    progress.console.print(
                        f"[bold red]{'Error':>12}[/] processing {file_path}: {e}"
                    )

                finally:
                    progress.update(task1, advance=1)

            # Clean up
            progress.remove_task(task1)

            if not protein_sizes:
                progress.console.print(
                    f"[bold red]{'Warning':>12}[/] No protein sequences were extracted from the input files"
                )
            else:
                progress.console.print(
                    f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(df)} representative systems"
                )
            
            return protein_sizes

    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Extracts nucleotide sequences from DefenseFinder output files."""
        if len(inputs) == 1 and str(inputs[0]).endswith(".tsv"):
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                # Check if the TSV contains the expected columns
                if "nucleotide_file" in df.columns:
                    progress.console.print(
                        f"[bold blue]{'Found':>12}[/] {len(df)} nucleotide files to process (old format)"
                    )
                    return self._process_existing_fna_files(progress, df, output)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] Input TSV is missing required columns. Expected 'nucleotide_file'"
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
        """Extracts protein sequences from DefenseFinder output files."""
        # Check if input is a TSV file containing paths to FAA files
        if len(inputs) == 1 and str(inputs[0]).endswith(".tsv"):
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                # Check if the TSV contains the expected columns
                if "protein_file" in df.columns:
                    progress.console.print(
                        f"[bold blue]{'Found':>12}[/] {len(df)} FAA files to process"
                    )
                    return self._process_existing_faa_files(progress, df, output, representatives)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] Input TSV is missing required columns. Expected 'protein_file'"
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
