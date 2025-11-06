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
from .defense_extractor import (
    GenomeContext,
    GenomeResources,
    DefenseSystem,
    DefenseSystemExtractor
)


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
        self.defense_metadata: typing.Optional[typing.Union[pathlib.Path, str]] = None
        self.verbose: bool = False
        self.activity_filter: str = "defense"
        self.gff_cache_dir: typing.Optional[pathlib.Path] = None

    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Extracts nucleotide sequences from defense systems."""
        
        extractor = DefenseSystemExtractor(progress=progress, verbose=self.verbose)
        
        progress.console.print(f"[bold blue]{'Using':>12}[/] defense metadata file: [magenta]{self.defense_metadata}[/]")
        
        try:
            df = pandas.read_csv(self.defense_metadata, sep="\t")
            return self._process_defense_files_from_tsv(progress, df, output, extractor)
        except Exception as e:
            progress.console.print(f"[bold red]{'Error':>12}[/] reading defense metadata: {e}")
            return pandas.DataFrame(columns=["cluster_id", "cluster_length", "filename"]).set_index("cluster_id")

    def _process_defense_files_from_tsv(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        extractor: DefenseSystemExtractor
    ) -> pandas.DataFrame:
        """Process defense systems from TSV file with file paths."""
        
        chunk_size = 5
        all_data = []

        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] defense systems", total=len(df))

            for chunk_start in range(0, len(df), chunk_size):
                chunk_end = min(chunk_start + chunk_size, len(df))
                df_chunk = df.iloc[chunk_start:chunk_end]

                chunk_data = []
                for _, row in df_chunk.iterrows():
                    genome_id = row.get("genome_id", None)
                    progress.update(task, description=f"[bold blue]{'Processing':>9}[/] strain: [bold cyan]{genome_id}")

                    context = GenomeContext(
                        genome_id=genome_id,
                        systems_tsv=pathlib.Path(row["systems_tsv"]),
                        genes_tsv=pathlib.Path(row["genes_tsv"]),
                        gff_file=pathlib.Path(row["gff_file"]),
                        genomic_fasta=pathlib.Path(row["genome_fasta_file"]),
                        protein_fasta=pathlib.Path(row["protein_fasta_file"]),
                        activity_filter=self.activity_filter,
                    )
                    
                    if not context.is_valid():
                        progress.console.print(
                            f"[bold yellow]{'Missing':>12}[/] files for {genome_id}: {', '.join(context.missing_files)}"
                        )
                        progress.update(task, advance=1)
                        continue

                    # extract genomic sequences for defense systems
                    try:
                        systems = extractor.extract_systems(
                            context=context,
                            output_file=dst,
                        )
                        chunk_data.extend(systems)
                        del systems
                    except Exception as e:
                        progress.console.print(f"[bold red]{'Error':>12}[/] processing {genome_id}: {e}")

                    progress.update(task, advance=1)
                
                all_data.extend(chunk_data)
                del chunk_data, df_chunk
                gc.collect()

            progress.remove_task(task)

        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(all_data)} systems in total from {df.shape[0]} strains/genomes"
        )

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
        """Extracts protein sequences from defense systems."""
        
        extractor = DefenseSystemExtractor(progress=progress, verbose=self.verbose)

        progress.console.print(f"[bold blue]{'Using':>12}[/] defense metadata file: [magenta]{self.defense_metadata}[/]")

        try:
            df = pandas.read_csv(self.defense_metadata, sep="\t")
            return self._extract_proteins_from_tsv(progress, df, output, representatives, extractor)
        except Exception as e:
            progress.console.print(f"[bold red]{'Error':>12}[/] reading defense metadata: {e}")
            return {}

    def _extract_proteins_from_tsv(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        representatives: typing.Container[str],
        extractor: DefenseSystemExtractor
    ) -> typing.Dict[str, int]:
        """Extract proteins from defense systems specified in TSV."""
        
        protein_sizes = {}
        
        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] protein sequences", total=len(df))

            for n, row in df.iterrows():
                genome_id = row.get("genome_id", None)
                if genome_id is None:
                    genome_id = f"genome_{n:07}"
                progress.update(task, description=f"[bold blue]{'Processing':>9}[/] strain: [bold cyan]{genome_id}")

                context = GenomeContext(
                    genome_id=genome_id,
                    systems_tsv=pathlib.Path(row["systems_tsv"]),
                    genes_tsv=pathlib.Path(row["genes_tsv"]),
                    gff_file=pathlib.Path(row["gff_file"]),
                    genomic_fasta=pathlib.Path(row["genome_fasta_file"]),
                    protein_fasta=pathlib.Path(row["protein_fasta_file"]),
                    activity_filter=self.activity_filter,
                )
                
                if not context.is_valid():
                    progress.console.print(
                        f"[bold yellow]{'Missing':>12}[/] files for {genome_id}: {', '.join(context.missing_files)}"
                    )
                    progress.update(task, advance=1)
                    continue
                # extract protein sequences for representative defense systems
                try:
                    proteins = extractor.extract_proteins(
                        context=context,
                        output_file=dst,
                        representatives=representatives
                    )
                    protein_sizes.update(proteins)
                except Exception as e:
                    progress.console.print(f"[bold red]{'Error':>12}[/] processing {genome_id}: {e}")

                progress.update(task, advance=1)

            progress.remove_task(task)

        rep_count = "all"
        if representatives:
            try:
                rep_count = str(len(representatives))
            except TypeError:
                rep_count = "specified"

        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from "
            f"{rep_count} representative systems"
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