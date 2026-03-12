import pathlib
import sys
import typing
from dataclasses import dataclass
from contextlib import contextmanager

import pandas as pd
import rich.progress

from .base import BaseDataset, Cluster, Protein
from .fasta_gff import FastaGFFDataset, IDResolver

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class MemoryProfiler:
    """Memory profiling utility."""
    
    def __init__(self, enabled: bool = True, console=None):
        """Initialize profiler.
        
        Args:
            enabled: Whether profiling is active
            console: Optional console for output (uses print if None)
        """
        self.enabled = enabled and PSUTIL_AVAILABLE
        self.console = console
        self._checkpoints = {}
    
    def get_memory_mb(self) -> float:
        """Get current process memory usage in MB."""
        if not self.enabled:
            return 0.0
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def format_memory_mb(mb: float) -> str:
        """Format memory size in MB."""
        if mb >= 1024:
            return f"{mb/1024:.2f} GB"
        return f"{mb:.2f} MB"
    
    def checkpoint(self, name: str):
        """Record a memory checkpoint."""
        if self.enabled:
            self._checkpoints[name] = self.get_memory_mb()
    
    def log(self, message: str, style: str = "cyan"):
        """Log a profiling message."""
        if not self.enabled:
            return
        
        formatted = f"[{style}]{message}[/]" if style else message
        
        if self.console and hasattr(self.console, 'print'):
            self.console.print(formatted)
        else:
            print(formatted)
    
    @contextmanager
    def profile_block(self, name: str, log_start: bool = False):
        """Context manager for profiling a code block.
        
        Usage:
            with profiler.profile_block("Loading data"):
                data = load_data()
        """
        if not self.enabled:
            yield
            return
        
        mem_before = self.get_memory_mb()
        if log_start:
            self.log(f"{name}: {self.format_memory_mb(mem_before)}", "dim cyan")
        
        yield
        
        mem_after = self.get_memory_mb()
        delta = mem_after - mem_before
        self.log(
            f"{name}: {self.format_memory_mb(mem_after)} "
            f"(Δ{self.format_memory_mb(delta)})",
            "cyan"
        )
    
    def log_dataframe_size(self, df: pd.DataFrame, name: str):
        """Log the memory usage of a DataFrame."""
        if not self.enabled:
            return
        
        df_size = df.memory_usage(deep=True).sum() / 1024 / 1024
        current_mem = self.get_memory_mb()
        self.log(
            f"{name}: {self.format_memory_mb(df_size)} "
            f"(process: {self.format_memory_mb(current_mem)})",
            "cyan"
        )
    
    def log_summary(self, summary_dict: typing.Dict[str, typing.Any]):
        """Log a formatted summary."""
        if not self.enabled:
            return
        
        lines = ["[bold cyan]Memory Profile Summary:[/]"]
        for key, value in summary_dict.items():
            lines.append(f"  {key}: {value}")
        
        self.log("\n".join(lines), None)


class InMemoryClusterDataset(FastaGFFDataset):
    """Dataset with clusters stored in memory for a single genome."""

    def __init__(
        self,
        clusters_grouped: pd.core.groupby.DataFrameGroupBy,
        genome_id: str,
        gff_file: pathlib.Path,
        genome_fasta: pathlib.Path,
        protein_fasta: pathlib.Path,
        gff_resolver: typing.Optional[IDResolver] = None,
        gff_attributes: typing.Optional[typing.List[str]] = None,
        profiler: typing.Optional[MemoryProfiler] = None,
    ) -> None:
        """Initialize with grouped DataFrame reference.

        Args:
            clusters_grouped: Grouped clusters DataFrame (shared reference)
            genome_id: Genome identifier
            gff_file: Path to GFF file
            genome_fasta: Path to genome FASTA
            protein_fasta: Path to protein FASTA
            gff_resolver: Custom GFF ID resolver
            gff_attributes: GFF attributes to index
            profiler: Optional memory profiler instance
        """
        self._clusters_grouped = clusters_grouped
        self._genome_id_key = genome_id

        self.genome_id = genome_id
        self.gff_file = gff_file
        self.genome_fasta = genome_fasta
        self.protein_fasta = protein_fasta
        self.column_mapping = {
            "cluster_id": "sys_id",
            "genes_in_cluster": "protein_in_syst",
        }
        self._gff_resolver = gff_resolver
        self._gff_attributes = gff_attributes
        self.profiler = profiler or MemoryProfiler(enabled=False)

        self._cluster_df_cache = None
        self._protein_idx = None
        self._gff_db = None
        self._coordinates = None

    @property
    def cluster_df(self) -> pd.DataFrame:
        """Return view of the grouped DataFrame for this genome."""
        if self._cluster_df_cache is None:
            self._cluster_df_cache = self._clusters_grouped.get_group(self._genome_id_key)
        return self._cluster_df_cache

    def cleanup_indexes(self):
        """Free memory by clearing heavy index objects after extraction."""
        if self._gff_db is not None:
            del self._gff_db
            self._gff_db = None
        
        if self._coordinates is not None:
            del self._coordinates
            self._coordinates = None
        
        if self._protein_idx is not None:
            del self._protein_idx
            self._protein_idx = None
        
        if self._cluster_df_cache is not None:
            del self._cluster_df_cache
            self._cluster_df_cache = None

    def get_shallow_size_kb(self) -> float:
        """Get approximate shallow size of this dataset object in KB."""
        size = sys.getsizeof(self)
        size += sys.getsizeof(self.genome_id)
        size += sys.getsizeof(self.gff_file)
        size += sys.getsizeof(self.genome_fasta)
        size += sys.getsizeof(self.protein_fasta)
        return size / 1024


class CustomTSVDataset(BaseDataset):
    """Dataset that loads clusters once and distributes across genome-specific datasets."""

    def __init__(
        self,
        clusters_tsv: pathlib.Path,
        metadata_tsv: pathlib.Path,
        gff_resolver: typing.Optional[IDResolver] = None,
        gff_attributes: typing.Optional[typing.List[str]] = None,
        genome_id_column: str = "#genome",
        progress: typing.Optional[rich.progress.Progress] = None,
        profiler: typing.Optional[MemoryProfiler] = None,
    ):
        """Initialize with cluster and metadata TSV files.

        Args:
            clusters_tsv: Path to clusters TSV (e.g., DefenseFinder systems)
            metadata_tsv: Path to metadata TSV with genome file paths
            gff_resolver: Custom GFF ID resolver to apply to all datasets
            gff_attributes: GFF attributes to index
            genome_id_column: Column name for genome ID in clusters_tsv
            progress: Optional progress bar
            profiler: Optional memory profiler (defaults to enabled profiler)
        """
        super().__init__()

        self.clusters_tsv = clusters_tsv
        self.metadata_tsv = metadata_tsv
        self.gff_resolver = gff_resolver
        self.gff_attributes = gff_attributes
        self.genome_id_column = genome_id_column
        
        # Use provided profiler or create default enabled one
        self.profiler = profiler if profiler is not None else MemoryProfiler(
            enabled=True,
            console=progress.console if progress else None
        )

        self.profiler.log(
            f"Memory before loading: {self.profiler.format_memory_mb(self.profiler.get_memory_mb())}",
            "cyan"
        )

        with self.profiler.profile_block("Metadata loading"):
            self.metadata_df = pd.read_csv(metadata_tsv, sep="\t", usecols=['genome_id','genome_fasta_file', 'gff_file', 'protein_fasta_file'], dtype={'genome_id': 'str', 'gff_file': 'str', 'genome_fasta_file': 'str', 'protein_fasta_file': 'str'}).sort_values("genome_id")
            self.metadata_df['gff_path'] = self.metadata_df['gff_file'].apply(pathlib.Path)
            self.metadata_df['genome_fasta_path'] = self.metadata_df['genome_fasta_file'].apply(pathlib.Path)
            self.metadata_df['protein_fasta_path'] = self.metadata_df['protein_fasta_file'].apply(pathlib.Path)

        
        
        self.profiler.log_dataframe_size(self.metadata_df, "Metadata loaded")

        with self.profiler.profile_block("Clusters loading"):
            self.clusters_df = pd.read_csv(clusters_tsv, sep="\t", usecols=['#genome', 'sys_id', 'protein_in_syst'])
            self.clusters_grouped = self.clusters_df.groupby(genome_id_column)
        
        self.profiler.log_dataframe_size(self.clusters_df, "Clusters loaded")

        self.datasets = self._create_datasets(progress)

    def _create_datasets(
        self, progress: typing.Optional[rich.progress.Progress] = None
    ) -> typing.List[InMemoryClusterDataset]:
        """Create one dataset per genome with shared DataFrame."""
        
        n_genomes = len(self.metadata_df)
        datasets = [None] * n_genomes

        self.profiler.checkpoint("before_datasets")
        
        task_id = None
        if progress:
            task_id = progress.add_task("Creating datasets...", total=n_genomes)
        
        # sample indexes for logging intermediate memory usage during dataset creation
        sample_indices = []
        if self.profiler.enabled and len(self.metadata_df) >= 10:
            sample_indices = [int(i * len(self.metadata_df) / 10) for i in range(10)]

        for idx, row in enumerate(self.metadata_df.itertuples()):
            datasets[idx] = InMemoryClusterDataset(
                clusters_grouped=self.clusters_grouped,
                genome_id=row.genome_id,
                gff_file=row.gff_path,
                genome_fasta=row.genome_fasta_path,
                protein_fasta=row.protein_fasta_path,
                gff_resolver=self.gff_resolver,
                gff_attributes=self.gff_attributes,
                profiler=self.profiler,
            )
            if idx % 100000 == 0:
                print(f"Created {idx}/{n_genomes} datasets")
            
            if idx in sample_indices:
                current_mem = self.profiler.get_memory_mb()
                dataset_size = datasets[idx].get_shallow_size_kb()
                self.profiler.log(
                    f"  Dataset {idx+1}/{len(self.metadata_df)}: "
                    f"{dataset_size:.2f} KB, process: {self.profiler.format_memory_mb(current_mem)}",
                    "dim cyan"
                )

            if progress and task_id:
                progress.update(task_id, advance=1)
        
        if progress and task_id:
            progress.remove_task(task_id)

        # Summary
        if self.profiler.enabled:
            mem_after = self.profiler.get_memory_mb()
            mem_before = self.profiler._checkpoints.get("before_datasets", 0)
            avg_dataset_size = sum(d.get_shallow_size_kb() for d in datasets) / len(datasets) if datasets else 0
            total_dataset_size = sum(d.get_shallow_size_kb() for d in datasets) / 1024
            
            self.profiler.log_summary({
                "Datasets created": len(datasets),
                f"Average dataset size": f"{avg_dataset_size:.2f} KB (shallow)",
                f"Total datasets size": self.profiler.format_memory_mb(total_dataset_size) + " (shallow)",
                f"Process memory": self.profiler.format_memory_mb(mem_after),
                f"Memory increase": self.profiler.format_memory_mb(mem_after - mem_before)
            })

        return datasets

    def extract_clusters(
        self, progress: typing.Optional[rich.progress.Progress] = None
    ) -> typing.Iterable[Cluster]:
        """Extract clusters from all datasets."""
        with self.profiler.profile_block("Cluster extraction"):
            for dataset in self.datasets:
                yield from dataset.extract_clusters(progress=progress)
                dataset.cleanup_indexes()

    def extract_proteins(
        self,
        progress: typing.Optional[rich.progress.Progress] = None,
        cluster_ids: typing.Optional[typing.Collection[str]] = None,
    ) -> typing.Iterable[Protein]:
        """Extract proteins from all datasets."""
        with self.profiler.profile_block("Protein extraction"):
            for dataset in self.datasets:
                yield from dataset.extract_proteins(progress, cluster_ids)
                dataset.cleanup_indexes()