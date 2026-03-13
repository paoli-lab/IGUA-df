import pathlib
import typing
from dataclasses import dataclass

import pandas as pd
import rich.progress

from .base import BaseDataset, Cluster, Protein
from .fasta_gff import FastaGFFDataset, IDResolver


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
    ):
        """Initialize with cluster and metadata TSV files.

        Args:
            clusters_tsv: Path to clusters TSV (e.g., DefenseFinder systems)
            metadata_tsv: Path to metadata TSV with genome file paths
            gff_resolver: Custom GFF ID resolver to apply to all datasets
            gff_attributes: GFF attributes to index
            genome_id_column: Column name for genome ID in clusters_tsv
            progress: Optional progress bar
        """
        super().__init__()

        self.clusters_tsv = clusters_tsv
        self.metadata_tsv = metadata_tsv
        self.gff_resolver = gff_resolver
        self.gff_attributes = gff_attributes
        self.genome_id_column = genome_id_column

        self.metadata_df = pd.read_csv(metadata_tsv, sep="\t", usecols=['genome_id','genome_fasta_file', 'gff_file', 'protein_fasta_file'], dtype={'genome_id': 'str', 'gff_file': 'str', 'genome_fasta_file': 'str', 'protein_fasta_file': 'str'}).sort_values("genome_id")
        self.metadata_df['gff_path'] = self.metadata_df['gff_file'].apply(pathlib.Path)
        self.metadata_df['genome_fasta_path'] = self.metadata_df['genome_fasta_file'].apply(pathlib.Path)
        self.metadata_df['protein_fasta_path'] = self.metadata_df['protein_fasta_file'].apply(pathlib.Path)

        self.clusters_df = pd.read_csv(clusters_tsv, sep="\t", usecols=['#genome', 'sys_id', 'protein_in_syst'])
        
        self.clusters_grouped = self.clusters_df.groupby(genome_id_column)
        self.datasets = self._create_datasets(progress)

    def _create_datasets(
        self, progress: typing.Optional[rich.progress.Progress] = None
    ) -> typing.List[InMemoryClusterDataset]:
        """Create one dataset per genome with shared DataFrame."""
        
        n_genomes = len(self.metadata_df)
        datasets = [None] * n_genomes
        
        task_id = None
        if progress:
            task_id = progress.add_task("Creating datasets...", total=n_genomes)
        
        for idx, row in enumerate(self.metadata_df.itertuples()):
            datasets[idx] = InMemoryClusterDataset(
                clusters_grouped=self.clusters_grouped,
                genome_id=row.genome_id,
                gff_file=row.gff_path,
                genome_fasta=row.genome_fasta_path,
                protein_fasta=row.protein_fasta_path,
                gff_resolver=self.gff_resolver,
                gff_attributes=self.gff_attributes,
            )
            if idx % 100000 == 0:
                print(f"Created {idx}/{n_genomes} datasets")
            if progress and task_id:
                progress.update(task_id, advance=1)
        
        if progress and task_id:
            progress.remove_task(task_id)
        
        return datasets

    def extract_clusters(
        self, progress: typing.Optional[rich.progress.Progress] = None
    ) -> typing.Iterable[Cluster]:
        """Extract clusters from all datasets."""
        for dataset in self.datasets:
            yield from dataset.extract_clusters(progress=progress)
            dataset.cleanup_indexes()

    def extract_proteins(
        self,
        progress: typing.Optional[rich.progress.Progress] = None,
        cluster_ids: typing.Optional[typing.Collection[str]] = None,
    ) -> typing.Iterable[Protein]:
        """Extract proteins from all datasets."""
        for dataset in self.datasets:
            yield from dataset.extract_proteins(progress, cluster_ids)
            dataset.cleanup_indexes()