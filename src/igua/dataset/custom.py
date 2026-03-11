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
        clusters_df: pd.DataFrame,
        genome_id: str,
        genome_mask: pd.Series,
        gff_file: pathlib.Path,
        genome_fasta: pathlib.Path,
        protein_fasta: pathlib.Path,
        gff_resolver: typing.Optional[IDResolver] = None,
        gff_attributes: typing.Optional[typing.List[str]] = None,
    ) -> None:
        """Initialize with shared DataFrame and boolean mask.

        Args:
            clusters_df: The FULL clusters DataFrame (shared across all datasets)
            genome_id: Genome identifier
            genome_mask: Boolean mask selecting rows for this genome
            gff_file: Path to GFF file
            genome_fasta: Path to genome FASTA
            protein_fasta: Path to protein FASTA
            gff_resolver: Custom GFF ID resolver
            gff_attributes: GFF attributes to index
        """
        self._full_clusters_df = clusters_df
        self._genome_mask = genome_mask

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

        self._protein_idx = None
        self._gff_db = None
        self._coordinates = None

    @property
    def cluster_df(self) -> pd.DataFrame:
        """Return view of the shared DataFrame for this genome."""
        return self._full_clusters_df[self._genome_mask]


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

        self.metadata_df = pd.read_csv(metadata_tsv, sep="\t").sort_values("genome_id")
        self.clusters_df = pd.read_csv(clusters_tsv, sep="\t")

        self.datasets = self._create_datasets(progress)

    def _create_datasets(
        self, progress: typing.Optional[rich.progress.Progress] = None
    ) -> typing.List[InMemoryClusterDataset]:
        """Create one dataset per genome with shared DataFrame."""
        datasets = []

        task_id = None
        if progress:
            task_id = progress.add_task(
                "Creating datasets...", total=len(self.metadata_df)
            )

        for _, row in self.metadata_df.iterrows():
            genome_id = row["genome_id"]
            genome_mask = self.clusters_df[self.genome_id_column] == genome_id

            if not genome_mask.any():
                if progress and task_id:
                    progress.console.print(
                        f"[yellow]Warning:[/] No clusters found for {genome_id}, skipping"
                    )
                    progress.update(task_id, advance=1)
                continue

            dataset = InMemoryClusterDataset(
                clusters_df=self.clusters_df,
                genome_id=genome_id,
                genome_mask=genome_mask,
                gff_file=pathlib.Path(row["gff_file"]),
                genome_fasta=pathlib.Path(row["genome_fasta_file"]),
                protein_fasta=pathlib.Path(row["protein_fasta_file"]),
                gff_resolver=self.gff_resolver,
                gff_attributes=self.gff_attributes,
            )
            datasets.append(dataset)

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

    def extract_proteins(
        self,
        progress: typing.Optional[rich.progress.Progress] = None,
        cluster_ids: typing.Optional[typing.Collection[str]] = None,
    ) -> typing.Iterable[Protein]:
        """Extract proteins from all datasets."""
        for dataset in self.datasets:
            yield from dataset.extract_proteins(progress, cluster_ids)