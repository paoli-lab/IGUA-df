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
from .cluster_extractor import (
    GenomeContext,
    GeneClusterExtractor
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
        tmp_fasta = output_db_path.with_suffix(".fna")
        self.extract_sequences(progress, inputs, tmp_fasta)
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
        tmp_fasta = output_db_path.with_suffix(".faa")
        protein_sizes = self.extract_proteins(progress, inputs, tmp_fasta, representatives)
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
            f"[bold green]{'Extracted':>12}[/] {len(protein_sizes):,} proteins from {len(representatives):,} nucleotide representative"
        )
        return protein_sizes


class FastaGFFDataset(BaseDataset):
    """FastaGFF dataset class.
    This class is used to extract nucleotide and protein sequences from fasta and GFF files specifying gene clusters.
    """
    def __init__(self) -> None:
        """Initialize the FastaGFFDataset class."""
        self.cluster_metadata: typing.Optional[typing.Union[pathlib.Path, str]] = None
        self.verbose: bool = False
        self.activity_filter: str = "defense"
        self.gff_cache_dir: typing.Optional[pathlib.Path] = None

    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Extracts nucleotide sequences from gene clusters."""
        extractor = GeneClusterExtractor(progress=progress, verbose=self.verbose)
        progress.console.print(f"[bold blue]{'Using':>12}[/] cluster metadata file: [magenta]{self.cluster_metadata}[/]")
        
        try:
            df = pandas.read_csv(self.cluster_metadata, sep="\t")
            return self._process_genomes(progress, df, output, extractor, is_protein=False)
        except Exception as e:
            progress.console.print(f"[bold red]{'Error':>12}[/] reading cluster metadata: {e}")
            return pandas.DataFrame(columns=["cluster_id", "cluster_length", "filename"]).set_index("cluster_id")

    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
    ) -> typing.Dict[str, int]:
        """Extracts protein sequences from gene clusters."""
        extractor = GeneClusterExtractor(progress=progress, verbose=self.verbose)
        progress.console.print(f"[bold blue]{'Using':>12}[/] cluster metadata file: [magenta]{self.cluster_metadata}[/]")

        try:
            df = pandas.read_csv(self.cluster_metadata, sep="\t")
            return self._process_genomes(progress, df, output, extractor, is_protein=True, representatives=representatives)
        except Exception as e:
            progress.console.print(f"[bold red]{'Error':>12}[/] reading cluster metadata: {e}")
            return {}

    def _process_genomes(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        extractor: GeneClusterExtractor,
        is_protein: bool = False,
        representatives: typing.Optional[typing.Container[str]] = None
    ) -> typing.Union[pandas.DataFrame, typing.Dict[str, int]]:
        """Process all genomes for sequence or protein extraction."""
        
        task_name = "protein sequences" if is_protein else "gene clusters"
        results = {} if is_protein else []
        
        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] {task_name}", total=len(df))
            
            chunk_size = 5 if not is_protein else None
            chunks = self._chunk_dataframe(df, chunk_size) if chunk_size else [(0, df)]
            
            for chunk_idx, df_chunk in chunks:
                chunk_data = [] if not is_protein else None
                
                for idx, row in df_chunk.iterrows():
                    genome_id = row.get("genome_id") or f"genome_{idx:07}"
                    progress.update(task, description=f"[bold blue]{'Processing':>9}[/] strain: [bold cyan]{genome_id}")
                    
                    context = self._create_genome_context(row, genome_id)
                    
                    if not context.is_valid():
                        progress.console.print(f"[bold yellow]{'Missing':>12}[/] files for {genome_id}")
                        progress.update(task, advance=1)
                        continue
                    
                    try:
                        if is_protein:
                            proteins = extractor.extract_proteins(context, dst, representatives)
                            results.update(proteins)
                        else:
                            systems = extractor.extract_systems(context, dst)
                            chunk_data.extend(systems)
                    except Exception as e:
                        progress.console.print(f"[bold red]{'Error':>12}[/] processing {genome_id}: {e}")
                    
                    progress.update(task, advance=1)
                
                if not is_protein:
                    results.extend(chunk_data)
                    del chunk_data
                    gc.collect()
            
            progress.remove_task(task)
        
        if is_protein:
            self._log_protein_summary(progress, results, representatives)
            return results
        else:
            self._log_sequence_summary(progress, results, len(df))
            return self._create_result_dataframe(results)
    
    def _create_genome_context(self, row: pandas.Series, genome_id: str) -> GenomeContext:
        """Create GenomeContext from a dataframe row."""
        return GenomeContext(
            genome_id=genome_id,
            systems_tsv=pathlib.Path(row["systems_tsv"]),
            genes_tsv=pathlib.Path(row["genes_tsv"]),
            gff_file=pathlib.Path(row["gff_file"]),
            genomic_fasta=pathlib.Path(row["genome_fasta_file"]),
            protein_fasta=pathlib.Path(row["protein_fasta_file"]),
            activity_filter=self.activity_filter,
        )
    
    def _chunk_dataframe(self, df: pandas.DataFrame, chunk_size: int) -> typing.Generator[typing.Tuple[int, pandas.DataFrame], None, None]:
        """Yield dataframe chunks."""
        for start in range(0, len(df), chunk_size):
            end = min(start + chunk_size, len(df))
            yield start, df.iloc[start:end]
    
    def _create_result_dataframe(self, data: typing.List[typing.Tuple]) -> pandas.DataFrame:
        """Create result dataframe from sequence extraction data."""
        df = pandas.DataFrame(
            data=data,
            columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")
        del data
        gc.collect()
        return df
    
    def _log_sequence_summary(self, progress: rich.progress.Progress, results: typing.List, n_genomes: int):
        """Log summary for sequence extraction."""
        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(results):,} gene clusters in total from {n_genomes:,} strains/genomes"
        )
    
    def _log_protein_summary(self, progress: rich.progress.Progress, results: typing.Dict, representatives: typing.Optional[typing.Container[str]]):
        """Log summary for protein extraction."""
        rep_count = "all"
        if representatives:
            try:
                rep_count = str(len(representatives))
            except TypeError:
                rep_count = "specified"
        
        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(results):,} proteins from {rep_count} representative gene clusters"
        )

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