from asyncio.log import logger
import functools
import pathlib
import typing
import multiprocessing.pool
from dataclasses import dataclass

import pandas as pd
import rich.progress

from .base import BaseDataset, Cluster, Protein
from .fasta_gff import FastaGFFDataset, IDResolver, SystemCoordinates


class InMemoryClusterDataset(FastaGFFDataset):
    """Dataset with clusters stored in memory for a single genome."""

    def __init__(
        self,
        clusters_df: pd.DataFrame,
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

        self._memory_cluster_df = clusters_df

    @property
    def cluster_df(self) -> pd.DataFrame:
        """Override parent property to return in-memory DataFrame."""
        return self._memory_cluster_df


    def cleanup_indexes(self):
        """Free memory by clearing heavy index objects after extraction."""
        if hasattr(self, "gff_db"):
            del self.gff_db
        if hasattr(self, "coordinates"):
            del self.coordinates
        if hasattr(self, "protein_idx"):
            del self.protein_idx

    @functools.cached_property
    def coordinates(self) -> typing.List[SystemCoordinates]:
        """Override to use fast iteration and optimized, decoupled boundary parsing."""
        coords = []
        
        gff_get = self.gff_db.get
        genome_fasta_str = str(self.genome_fasta)
        genome_id = self.genome_id
        
        for row in self.cluster_df.itertuples(index=False):
            cluster_id = str(row.sys_id)
            
            raw_genes = str(row.protein_in_syst)
            gene_list = [g for g in (x.strip() for x in raw_genes.split(",")) if g]
            
            if not gene_list:
                coords.append(self._invalid_coord(cluster_id, [], "Empty gene list"))
                continue
            
            beg_gene = str(row.sys_beg).strip()
            end_gene = str(row.sys_end).strip()
            
            feat_beg = gff_get(beg_gene)
            feat_end = gff_get(end_gene)
            
            if not feat_beg or not feat_end:
                missing = [g for g, f in [(beg_gene, feat_beg), (end_gene, feat_end)] if not f]
                coords.append(self._invalid_coord(
                    cluster_id, 
                    gene_list, 
                    f"Boundary genes not found in GFF: {', '.join(missing)}"
                ))
                continue
                
            if feat_beg.seqid != feat_end.seqid:
                coords.append(self._invalid_coord(
                    cluster_id, 
                    gene_list, 
                    f"Genes span multiple contigs: {feat_beg.seqid} and {feat_end.seqid}"
                ))
                continue
                
            start = min(feat_beg.start, feat_beg.end, feat_end.start, feat_end.end)
            end = max(feat_beg.start, feat_beg.end, feat_end.start, feat_end.end)
            
            region_size = end - start + 1
            if region_size > 100000:
                logger.info(f"Cluster {cluster_id} unusually large: {region_size:,} bp (genome: [bold cyan]{genome_id}[/])")
            elif region_size < 50:
                logger.info(f"Cluster {cluster_id} unusually small: {region_size} bp (genome: [bold cyan]{genome_id}[/])")
                
            coords.append(SystemCoordinates(
                cluster_id=cluster_id,
                seq_id=feat_beg.seqid,
                start_coord=start,
                end_coord=end,
                strand=feat_beg.strand,
                genes=gene_list,
                fasta_file=genome_fasta_str,
                valid=True
            ))
            
        return coords


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
        threads: int = None,
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

        self.threads = threads # number of threads for parallel extraction of clusters

        self.clusters_tsv = clusters_tsv
        self.metadata_tsv = metadata_tsv
        self.gff_resolver = gff_resolver
        self.gff_attributes = gff_attributes
        self.genome_id_column = genome_id_column
        
        self.metadata_df = pd.read_csv(metadata_tsv, sep="\t", usecols=['genome_id','genome_fasta_file', 'gff_file', 'protein_fasta_file'], dtype={'genome_id': 'str', 'gff_file': 'str', 'genome_fasta_file': 'str', 'protein_fasta_file': 'str'}).sort_values("genome_id")


        self.clusters_df = pd.read_csv(clusters_tsv, sep="\t", usecols=['#genome', 'sys_id', 'protein_in_syst', 'sys_beg', 'sys_end']) 


        duplicate_mask = self.clusters_df.duplicated(subset=['sys_id'], keep="first")
        if duplicate_mask.any():
            duplicate_clusters = self.clusters_df[duplicate_mask]['sys_id'].tolist()
            n_duplicates = len(duplicate_clusters)

            logger.warning(
                f"{n_duplicates} duplicate cluster/s: "
                f"[cyan]{', '.join(duplicate_clusters[:5])}{'...' if n_duplicates > 5 else ''}[/]"
            )
            self.clusters_df = self.clusters_df.drop_duplicates(subset=['sys_id'], keep="first")

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
                clusters_df=self.clusters_grouped.get_group(row.genome_id),
                genome_id=row.genome_id,
                gff_file=row.gff_file,
                genome_fasta=row.genome_fasta_file,
                protein_fasta=row.protein_fasta_file,
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
        def process(dataset: InMemoryClusterDataset):
            clusters = list(dataset.extract_clusters(progress=progress))
            dataset.cleanup_indexes()
            return clusters

        with multiprocessing.pool.ThreadPool(self.threads) as pool:
            for clusters in pool.imap(process, self.datasets):
                yield from clusters

    def extract_proteins(
        self,
        progress: typing.Optional[rich.progress.Progress] = None,
        cluster_ids: typing.Optional[typing.Collection[str]] = None,
    ) -> typing.Iterable[Protein]:
        """Extract proteins from all datasets."""
        def process(dataset: InMemoryClusterDataset):
            clusters = list(dataset.extract_proteins(progress, cluster_ids))
            dataset.cleanup_indexes()
            return clusters

        with multiprocessing.pool.ThreadPool(self.threads) as pool:
            for clusters in pool.imap(process, self.datasets):
                yield from clusters
