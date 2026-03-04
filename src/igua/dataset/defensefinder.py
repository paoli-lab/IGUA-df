import logging
import pathlib
import typing

import pandas as pd
import rich.progress

from .base import BaseDataset, Cluster, Protein
from .fasta_gff import FastaGFFDataset


logger = logging.getLogger(__name__)


class DefenseFinderDataset(FastaGFFDataset):
    """DefenseFinder-specific dataset class."""

    def __init__(
        self,
        cluster_table=pathlib.Path,
        gff_file=pathlib.Path,
        genome_fasta=pathlib.Path,
        protein_fasta=pathlib.Path,
        activity_filter: str = "all",
        genome_id: str = None,
    ) -> None:
        """Initialize DefenseFinder dataset.

        Args:
            cluster_table: Path to the DefenseFinder systems TSV file.
            gff_file: Path to the GFF file.
            genome_fasta: Path to the genome FASTA file.
            protein_fasta: Path to the protein FASTA file.
            activity_filter: Filter by activity type. Options: 'all',
                'defense', 'antidefense'.
        """
        defensefinder_mapping = {
            "cluster_id": "sys_id",
            "genes_in_cluster": "protein_in_syst",
        }

        super().__init__(
            cluster_table=cluster_table,
            genome_fasta=genome_fasta,
            gff_file=gff_file,
            protein_fasta=protein_fasta,
            column_mapping=defensefinder_mapping,
        )
        self.genome_id = genome_id if genome_id else str(genome_fasta)
        self.activity_filter = activity_filter
        self._apply_activity_filter()

    def _apply_activity_filter(self):
        """Apply activity filtering to loaded clusters."""
        if self.activity_filter.lower() == "all":
            return

        df = self._cluster_df
        if df is None:
            return

        original_count = len(df)

        if "activity" in df.columns:
            self._cluster_df = df[
                df["activity"].str.lower() == self.activity_filter.lower()
            ]
            logger.info(
                f"Filtered {original_count} → {len(self._cluster_df)} systems "
                f"({self.activity_filter} only, genome: [bold cyan]{self.genome_id}[/])"
            )
        else:
            logger.warning(
                f"No 'activity' column found in {self.cluster_table} "
                f"(genome: {self.genome_id})"
            )


class DefenseFinderTSVDataset(BaseDataset):
    """Dataset for multiple DefenseFinder results from metadata table."""

    def __init__(
        self,
        cluster_metadata_table: pathlib.Path,
        activity_filter: str = "all",
        progress: rich.progress.Progress = None,
    ):
        self.cluster_metadata_table: pathlib.Path = cluster_metadata_table
        self.activity_filter = activity_filter

        self.cluster_metadata_df = pd.read_csv(self.cluster_metadata_table, sep="\t")

        logger.info(
            f"Using cluster metadata table: [magenta]{self.cluster_metadata_table}"
        )

        self.datasets = [
            DefenseFinderDataset(
                cluster_table=pathlib.Path(row["cluster_table"]),
                gff_file=pathlib.Path(row["gff_file"]),
                genome_fasta=pathlib.Path(row["genome_fasta_file"]),
                protein_fasta=pathlib.Path(row["protein_fasta_file"]),
                activity_filter=self.activity_filter,
                genome_id=row.get("genome_id", None),
            )
            for _, row in self.cluster_metadata_df.iterrows()
        ]

    def extract_clusters(self, progress=None) -> typing.Iterable[Cluster]:
        """Extract clusters from all datasets.

        Args:
            progress: Ignored, kept for compatibility.

        Yields:
            Cluster objects.
        """
        for dataset in self.datasets:
            yield from dataset.extract_clusters(progress=progress)

    def extract_proteins(
        self,
        progress=None,
        cluster_ids: typing.Collection[str] = None,
    ) -> typing.Iterable[Protein]:
        """Extract proteins from all datasets.

        Args:
            progress: Ignored, kept for compatibility.
            cluster_ids: Optional set of cluster IDs to filter.

        Yields:
            Protein objects.
        """
        for dataset in self.datasets:
            yield from dataset.extract_proteins(progress, cluster_ids)
