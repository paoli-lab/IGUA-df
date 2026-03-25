import collections
import gzip
import io
import logging
import functools
from operator import ge
import pathlib
import re
import typing
import warnings
from pathlib import Path
from dataclasses import asdict, dataclass

import pandas as pd
from rich.logging import RichHandler
import rich.progress

from .base import BaseDataset, Cluster, Protein


_GZIP_MAGIC = b"\x1f\x8b"


logger = logging.getLogger()
logger.setLevel(logging.INFO)

console = RichHandler(rich_tracebacks=True, show_path=False, markup=True)
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(console)

file_handler = logging.FileHandler("extraction.log")
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(
    logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
)
logger.addHandler(file_handler)

# logging.getLogger("igua.dataset.fasta_gff").setLevel(logging.INFO)


def _parse_tar_path(
    path: pathlib.Path,
) -> typing.Union[tuple[pathlib.Path, str], tuple[None, None]]:
    """Parse a path that might point inside a tar archive.

    Args:
        path: Path that might contain a tar archive reference.

    Returns:
        Tuple of (tar_path, member_path) if tar archive found, else (None, None).
    """
    path_str = str(path)

    for ext in [".tar.gz", ".tar.bz2", ".tgz", ".tar"]:
        if ext in path_str:
            parts = path_str.split(ext, 1)
            tar_path = pathlib.Path(parts[0] + ext)
            if tar_path.exists():
                member_path = parts[1].lstrip("/")
                return tar_path, member_path

    return None, None


def smart_open(path: pathlib.Path, mode: str = "rb") -> typing.BinaryIO:
    """Open file, handling regular files and tar archives.

    Uses caching for tar members to avoid repeated extraction.
    Supports gzip compression for both file types.

    Args:
        path: Path to file (may be inside tar archive).
        mode: File mode ('rb' or 'rt').

    Returns:
        File-like object (binary mode).
    """
    tar_path, member_path = _parse_tar_path(path)

    if tar_path and member_path:
        #######
        member_data = None
        reader = io.BufferedReader(member_data)

        if reader.peek().startswith(_GZIP_MAGIC):
            reader = gzip.GzipFile(mode="rb", fileobj=reader)  # type: ignore

        return reader  # type: ignore
    else:
        reader = open(path, "rb")
        if reader.peek().startswith(_GZIP_MAGIC):
            reader = gzip.GzipFile(mode="rb", fileobj=reader)  # type: ignore
        return reader  # type: ignore


def read_fasta(file_path: pathlib.Path) -> typing.Iterable[typing.Tuple[str, str, str]]:
    """Stream FASTA records from file.

    Handles both plain and gzip-compressed FASTA files.

    Args:
        file_path: Path to FASTA file (.fasta, .fa, .fna, or .gz).

    Yields:
        Tuple of (sequence_id, full_header, sequence_string).
    """
    with smart_open(file_path) as reader:
        name = None
        full_header = None
        sequence = []

        for line in io.TextIOWrapper(reader, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue

            if line.startswith(">"):
                if name is not None:
                    yield name, full_header, "".join(sequence)
                full_header = line[1:]
                name = full_header.split()[0]
                sequence = []
            else:
                sequence.append(line)

        if name is not None:
            yield name, full_header, "".join(sequence)


@dataclass
class SystemCoordinates:
    """Genome coordinates for a gene cluster.

    Attributes:
        cluster_id: Cluster identifier.
        seq_id: Sequence/contig identifier.
        start_coord: Start coordinate on the sequence (1-based).
        end_coord: End coordinate on the sequence (1-based, inclusive).
        strand: Strand orientation ('+', '-', or '.').
        genes: List of gene identifiers in the cluster.
        fasta_file: Path to the genome FASTA file.
        valid: Whether the coordinates are valid.
        error_msg: Error message if coordinates are invalid.
    """

    cluster_id: str
    seq_id: str
    start_coord: int
    end_coord: int
    strand: str
    genes: typing.List[str]
    fasta_file: str
    valid: bool = True
    error_msg: typing.Optional[str] = None

    def to_dict(self) -> typing.Dict:
        """Convert to dictionary for serialization.

        Returns:
            Dictionary representation of the coordinates.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: typing.Dict) -> "SystemCoordinates":
        """Create from dictionary.

        Args:
            data: Dictionary containing coordinate data.

        Returns:
            SystemCoordinates instance.
        """
        return cls(**data)



class IDResolver(typing.Protocol):
    """Protocol for gene ID resolution strategies."""
    
    def __call__(self, gene_id: str, index: typing.Dict[str, typing.Dict]) -> typing.Optional[typing.Dict]:
        """Resolve gene_id to GFF feature.
        
        Args:
            gene_id: Gene identifier to resolve
            index: Raw GFF index dictionary
            
        Returns:
            Feature dict if found, None otherwise
        """
        ...


def default_resolver(gene_id: str, index: typing.Dict[str, typing.Dict]) -> typing.Optional[typing.Dict]:
    """Default resolution with common transformations."""
    if result := index.get(gene_id):
        return result
    
    for prefix in ["gene-", "cds-"]:
        if result := index.get(f"{prefix}{gene_id}"):
            return result
    
    for variant in [gene_id.replace("_", "~"), gene_id.replace("~", "_")]:
        if result := index.get(variant):
            return result


def strict_resolver(gene_id: str, index: typing.Dict[str, typing.Dict]) -> typing.Optional[typing.Dict]:
    """Strict resolver - only exact matches."""
    return index.get(gene_id)


def create_mapping_resolver(mapping: typing.Dict[str, str]) -> IDResolver:
    """Factory for mapping-based resolvers.
    
    Args:
        mapping: Dict mapping {fasta_id: gff_id}
        
    Returns:
        Resolver function that uses the mapping
    """
    def resolver(gene_id: str, index: typing.Dict[str, typing.Dict]) -> typing.Optional[typing.Dict]:
        gff_id = mapping.get(gene_id, gene_id)
        return index.get(gff_id)
    return resolver


def create_transform_resolver(
    transforms: typing.List[typing.Callable[[str], str]]
) -> IDResolver:
    """Factory for custom transformation resolvers.
    
    Args:
        transforms: List of transformation functions to try
        
    Returns:
        Resolver that tries each transformation in order
    """
    def resolver(gene_id: str, index: typing.Dict[str, typing.Dict]) -> typing.Optional[typing.Dict]:
        if result := index.get(gene_id):
            return result
        for transform in transforms:
            if result := index.get(transform(gene_id)):
                return result
        return None
    return resolver


class GFFIndex:
    """Fast, in-memory GFF index for gene feature lookup."""

    def __init__(
        self,
        gff_path: pathlib.Path,
        resolver: typing.Optional[IDResolver] = None,
        index_attributes: typing.Optional[typing.List[str]] = None,
    ):
        """Initialize GFF index.
        
        Args:
            gff_path: Path to GFF file
            resolver: Custom ID resolver (defaults to default_resolver)
            index_attributes: Which GFF attributes to index
        """
        self.path = gff_path
        self._index: typing.Dict[str, typing.Dict] = {}
        self.resolver = resolver or default_resolver
        self.index_attributes = set(index_attributes or [
            "ID", "locus_tag", "Name", "gene", "old_locus_tag", "protein_id"
        ])
        self._build_index()

    def _build_index(self):
        """Build index from GFF file."""
        with smart_open(self.path) as reader:
            for line in io.TextIOWrapper(reader, encoding="utf-8"):
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 9:
                    continue

                seqid, _, ftype, start, end, _, strand, _, attrs = parts
                attr_dict = dict(
                    item.split("=", 1) for item in attrs.split(";") if "=" in item
                )

                # create a dedicated dataclass/namedtuple for GFF features to facilitate type hints
                feature = {
                    "seqid": seqid,
                    "type": ftype,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "attributes": attr_dict,
                }

                for key in self.index_attributes:
                    if val := attr_dict.get(key):
                        self._index[val] = feature

    def get(self, gene_id: str) -> typing.Optional[typing.Dict]:
        """Get feature using configured resolver."""
        return self.resolver(gene_id, self._index)

    def __contains__(self, gene_id: str) -> bool:
        return self.get(gene_id) is not None



class ProteinIndex:
    """Lazy-loading protein sequence index for efficient lookup."""

    _ATTR_REGEX = {
        attr: re.compile(rf"\[{attr}=([^\]]+)\]")
        for attr in ["locus_tag", "ID", "Name", "gene", "protein_id"]
    }

    def __init__(self, protein_fasta: pathlib.Path):
        """Initialize protein index without loading sequences.

        Args:
            protein_fasta: Path to protein FASTA file.
        """
        self.path = protein_fasta
        self._sequences: typing.Dict[str, str] = {}
        self._loaded = False

    def _ensure_loaded(self, gene_ids: typing.Optional[set] = None):
        """Load protein sequences on demand.

        Args:
            gene_ids: Optional set of gene IDs to load. If None, loads all.
        """
        if self._loaded:
            return

        for seq_id, full_header, sequence in read_fasta(self.path):
            if gene_ids is None or seq_id in gene_ids:
                self._sequences[seq_id] = sequence

            if gene_ids:
                for regex in self._ATTR_REGEX.values():
                    if match := regex.search(full_header):
                        attr_value = match.group(1)
                        if attr_value in gene_ids:
                            self._sequences[attr_value] = sequence

        self._loaded = True

    def load_subset(self, gene_ids: set):
        """Pre-load only specific proteins by ID.

        Args:
            gene_ids: Set of gene IDs to load.
        """
        self._ensure_loaded(gene_ids)

    def get(self, protein_id: str) -> typing.Optional[str]:
        """Get protein sequence by ID.

        Args:
            protein_id: Protein identifier.

        Returns:
            Protein sequence if found, None otherwise.
        """
        if not self._loaded:
            self._ensure_loaded()
        return self._sequences.get(protein_id)


class FastaGFFDataset(BaseDataset):
    """Dataset for extracting sequences from FASTA/GFF files."""

    def __init__(
        self,
        cluster_table: pathlib.Path,
        gff_file: pathlib.Path,
        genome_fasta: pathlib.Path,
        protein_fasta: pathlib.Path,
        genome_id: typing.Optional[str] = None,
        column_mapping: typing.Optional[typing.Dict[str, str]] = None,
        gff_resolver: typing.Optional[typing.Callable[[str, typing.Dict[str, typing.Dict]], typing.Optional[typing.Dict]]] = None,
        gff_attributes: typing.Optional[typing.List[str]] = None, 
    ) -> None:
        """Initialize FastaGFFDataset.
        
        Args:
            cluster_table: Path to clusters table
            gff_file: Path to GFF file
            genome_fasta: Path to genome FASTA
            protein_fasta: Path to protein FASTA
            genome_id: Optional genome identifier
            column_mapping: Custom column mapping
            gff_resolver: Custom GFF ID resolver function
            gff_attributes: Which GFF attributes to index
        """
        super().__init__()
        self.genome_id = genome_id if genome_id else str(genome_fasta)
        self.cluster_table = cluster_table
        self.gff_file = gff_file
        self.genome_fasta = genome_fasta
        self.protein_fasta = protein_fasta
        self.column_mapping = column_mapping or {
            "cluster_id": "cluster_id",
            "genes_in_cluster": "genes_in_cluster",
        }
        self._gff_resolver = gff_resolver
        self._gff_attributes = gff_attributes


        self.is_valid = self._is_valid()
        if not self.is_valid:
            raise FileNotFoundError(f"Missing files for genome {self.genome_id}")

    def _is_valid(self) -> bool:
        """Check if all required files exist.

        Returns:
            True if all files are present, False otherwise.
        """
        self.missing_files = [
            f"{name}: {fpath}"
            for fpath, name in [
                (self.cluster_table, "cluster_table"),
                (self.gff_file, "gff_file"),
                (self.genome_fasta, "genome_fasta"),
                (self.protein_fasta, "protein_fasta"),
            ]
            if not fpath.exists() and "tar.gz" not in str(fpath)
        ]
        # logger.info(
        #     f"Validation for genome {self.genome_id}: "
        #     f"{'All files found' if not self.missing_files else 'Missing files: ' + ', '.join(self.missing_files)}"
        # )
        return len(self.missing_files) == 0

    def _load_and_filter_clusters(
        self, cluster_table_path: pathlib.Path, use_columns: typing.List[str]
    ) -> pd.DataFrame:
        """Load clusters TSV file.

        Args:
            cluster_table_path: Path to clusters TSV.
            use_columns: List of columns to use from the TSV.
            console: Rich console for logging.

        Returns:
            Pandas DataFrame with cluster data.
        """
        df = pd.read_csv(
            cluster_table_path, sep="\t", usecols=use_columns, low_memory=True
        )
        return df

    @property
    def cluster_df(self) -> pd.DataFrame:
        """Load and filter clusters TSV.

        Returns:
            Pandas DataFrame with filtered clusters.
        """

        use_columns = list(self.column_mapping.values())
        df = self._load_and_filter_clusters(self.cluster_table, use_columns)

        col_map = self.column_mapping
        cluster_id = col_map["cluster_id"]

        duplicate_mask = df.duplicated(subset=[cluster_id], keep="first")
        if duplicate_mask.any():
            duplicate_clusters = df[duplicate_mask][cluster_id].tolist()
            n_duplicates = len(duplicate_clusters)

            logger.warning(
                f"{n_duplicates} duplicate cluster/s in [bold cyan]{self.genome_id}[/]: "
                f"[cyan]{', '.join(duplicate_clusters[:5])}{'...' if n_duplicates > 5 else ''}[/]"
            )
            df = df.drop_duplicates(subset=[cluster_id], keep="first")

        self.cluster_df = df
        return self.cluster_df

    @functools.cached_property
    def protein_idx(self) -> ProteinIndex:
        """Get protein index.

        Returns:
            ProteinIndex instance.
        """
        return ProteinIndex(self.protein_fasta)

    @functools.cached_property
    def gff_db(self) -> GFFIndex:
        """Get GFF index with configured resolver."""
        return GFFIndex(
            self.gff_file,
            resolver=self._gff_resolver,
            index_attributes=self._gff_attributes,
        )

    @functools.cached_property
    def coordinates(self) -> typing.List[SystemCoordinates]:
        """Parse coordinates from cluster DataFrame and build cluster coordinates.

        Returns:
            List of SystemCoordinates for all clusters.
        """
        coordinates = []
        for _, row in self.cluster_df.iterrows():
            coord = self._parse_cluster_coordinates(row)
            coordinates.append(coord)
        return coordinates

    def _parse_cluster_coordinates(self, row: dict) -> SystemCoordinates:
        """Parse coordinates for a single cluster.

        Uses adapter's column mapping for format flexibility.

        Args:
            row: Dictionary containing cluster data from TSV row.

        Returns:
            SystemCoordinates instance.
        """
        col_map = self.column_mapping
        cluster_id = row[col_map["cluster_id"]]

        gene_list = [
            g.strip()
            for g in str(row[col_map["genes_in_cluster"]]).split(",")
            if g.strip()
        ]

        if not gene_list:
            return self._invalid_coord(cluster_id, [], "Empty gene list")

        features = [self.gff_db.get(gene_id) for gene_id in gene_list]

        if None in features:
            missing = [
                gene_id for gene_id, feat in zip(gene_list, features) if feat is None
            ]
            return self._invalid_coord(
                cluster_id,
                gene_list,
                f"Genes not found in GFF: {', '.join(missing[:3])}",
            )

        seq_ids = {feat["seqid"] for feat in features}
        if len(seq_ids) > 1:
            return self._invalid_coord(
                cluster_id, gene_list, f"Genes span multiple contigs: {seq_ids}"
            )

        seq_id = seq_ids.pop()

        start = min(min(feat["start"], feat["end"]) for feat in features)
        end = max(max(feat["start"], feat["end"]) for feat in features)

        strand = features[0]["strand"]

        region_size = end - start + 1
        if region_size > 1e5:
            logger.info(
                f"Cluster {cluster_id} unusually large: {region_size:,} bp (genome: [bold cyan]{self.genome_id}[/])"
            )

        elif region_size < 50:
            logger.info(
                f"Cluster {cluster_id} unusually small: {region_size} bp (genome: [bold cyan]{self.genome_id}[/])"
            )

        return SystemCoordinates(
            cluster_id=cluster_id,
            seq_id=seq_id,
            start_coord=start,
            end_coord=end,
            strand=strand,
            genes=gene_list,
            fasta_file=str(self.genome_fasta),
            valid=True,
        )

    def _invalid_coord(
        self, cluster_id: str, genes: typing.List[str], error: str, seq_id: str = ""
    ) -> SystemCoordinates:
        """Create an invalid SystemCoordinates object.

        Args:
            cluster_id: Cluster identifier.
            genes: List of gene identifiers.
            error: Error message describing the issue.
            seq_id: Sequence identifier (default: empty string).

        Returns:
            Invalid SystemCoordinates instance with error message.
        """

        # ? raise warning and contiune instead of returning empty coords?
        logger.warning(
            f"{error} for cluster {cluster_id} (genome: [bold cyan]{self.genome_id}[/])"
        )
        return SystemCoordinates(
            cluster_id=cluster_id,
            seq_id=seq_id,
            start_coord=0,
            end_coord=0,
            strand="",
            genes=genes,
            fasta_file=str(self.genome_fasta),
            valid=False,
            error_msg=error,
        )

    def extract_genome_sequences(
        self,
    ) -> typing.Iterator[Cluster]:
        """Extract nucleotide sequences for gene clusters.

        Streams FASTA file to minimize memory usage.

        Args:
            output: Output sink for writing records.

        Returns:
            List of (cluster_id, sequence_length, fasta_file) tuples.
        """
        # builds coordinates
        # iterates over rows of cluster_df
        # parses coordinates using gff_db
        # so gets genes + builds gff_df
        # validates coordinates (genes exist, on same contig, cluster size etc)
        coordinates = self.coordinates
        valid_coords = [c for c in coordinates if c.valid]

        if not valid_coords:
            logger.warning(f"No valid clusters to extract for {self.genome_id}")
            return []

        contig_groups: collections.defaultdict[str, typing.List[SystemCoordinates]] = (
            collections.defaultdict(list)
        )
        for coord in valid_coords:
            contig_groups.setdefault(coord.seq_id, []).append(coord)

        num_contigs = len(contig_groups)

        logger.info(
            f"[bold blue]Processing[/] {len(valid_coords)} clusters across {num_contigs} contigs (genome: [bold cyan]{self.genome_id}[/])"
        )

        for seq_id, _, sequence in read_fasta(self.genome_fasta):
            if seq_id not in contig_groups:
                continue

            logger.debug(
                f"Loading contig {seq_id} ({len(contig_groups[seq_id])} clusters, genome: [bold cyan]{self.genome_id}[/])"
            )

            for coord in contig_groups[seq_id]:
                subseq = sequence[coord.start_coord - 1 : coord.end_coord]
                yield Cluster(coord.cluster_id, subseq, source=coord.fasta_file)

            logger.debug(
                f"[bold green]Extracted[/] [cyan]{coord.cluster_id}[/] ({len(subseq)} bp, genome: [bold cyan]{self.genome_id}[/])"
            )

            del contig_groups[seq_id]

            if not contig_groups:
                break

        if contig_groups:
            for seq_id in contig_groups:
                logger.error(f"Contig {seq_id} not found in genome {self.genome_id}")

    def extract_proteins_from_coordinates(
        self,
        coordinates: typing.List[SystemCoordinates],
    ) -> typing.Iterable[Protein]:
        """Extract protein sequences from gene coordinates.

        Args:
            coordinates: List of cluster coordinates.
            output_file: Output file handle for writing sequences.

        Returns:
            Dictionary mapping protein_id to sequence length.
        """
        valid_coords = [c for c in coordinates if c.valid]

        if not valid_coords:
            logger.warning(f"No valid clusters for {self.genome_id}")
            return

        all_gene_ids = set()
        for coord in valid_coords:
            all_gene_ids.update(coord.genes)

        self.protein_idx.load_subset(all_gene_ids)

        total_genes = len(all_gene_ids)

        logger.info(
            f"[bold blue]Processing[/] {total_genes} proteins from {len(valid_coords)} clusters (genome: [bold cyan]{self.genome_id}[/])"
        )

        n_extracted = 0
        for coord in valid_coords:
            for gene_id in coord.genes:
                if seq := self.protein_idx.get(gene_id):
                    protein_id = f"{coord.cluster_id}__{gene_id}"
                    yield Protein(
                        protein_id,
                        seq,
                        cluster_id=coord.cluster_id,
                    )
                    n_extracted += 1
                else:
                    logger.warning(
                        f"Protein {gene_id} not found for cluster {coord.cluster_id} (genome: [bold cyan]{self.genome_id}[/])"
                    )

        logger.info(
            f"[bold green]Extracted[/] {n_extracted} proteins from {len(coordinates)} clusters (genome: [bold cyan]{self.genome_id}[/])"
        )

    def extract_clusters(
        self,
        progress: rich.progress.Progress,
    ) -> typing.Iterable[Cluster]:

        coordinates = self.coordinates
        valid_count = sum(1 for c in coordinates if c.valid)

        logger.info(
            f"[bold green]Validated[/] {valid_count}/{len(coordinates)} clusters (genome: [bold cyan]{self.genome_id}[/])"
        )

        n_extracted = 0
        for cluster in self.extract_genome_sequences():
            yield cluster
            n_extracted += 1

        contig_groups = collections.defaultdict(list)
        for coord in coordinates:
            if coord.valid:
                contig_groups[coord.seq_id].append(coord)
        num_contigs = len(contig_groups)

        logger.info(
            f"[bold green]Extracted[/] {n_extracted} gene clusters across {num_contigs} contigs (genome: [bold cyan]{self.genome_id}[/])"
        )

    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        cluster_ids: typing.Collection[str],
    ) -> typing.Iterable[Protein]:

        self.progress = progress

        coordinates = self.coordinates
        if cluster_ids:
            coordinates = [c for c in coordinates if c.cluster_id in cluster_ids]
        yield from self.extract_proteins_from_coordinates(
            coordinates,
        )


class MetadataTSVDataset(BaseDataset):

    def __init__(
        self,
        cluster_metadata_table: pathlib.Path,
        column_mapping: typing.Optional[typing.Dict[str, str]] = None,
        progress: rich.progress.Progress = None,
    ):
        self.cluster_metadata_table: pathlib.Path = cluster_metadata_table
        self.column_mapping = column_mapping
        self.progress = progress

        self.cluster_metadata_df = pd.read_csv(self.cluster_metadata_table, sep="\t")

        logger.info(
            f"Using cluster metadata table: [magenta]{str(self.cluster_metadata_table)}[/]"
        )

        self.datasets = [
            FastaGFFDataset(
                cluster_table=pathlib.Path(row["cluster_table"]),
                gff_file=pathlib.Path(row["gff_file"]),
                genome_fasta=pathlib.Path(row["genome_fasta_file"]),
                protein_fasta=pathlib.Path(row["protein_fasta_file"]),
                column_mapping=column_mapping,
                genome_id=row.get("genome_id", None),
            )
            for _, row in self.cluster_metadata_df.iterrows()
        ]
        pass

    def extract_clusters(
        self, progress: rich.progress.Progress
    ) -> typing.Iterable[Cluster]:

        for dataset in self.datasets:
            yield from dataset.extract_clusters(progress=progress)

    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        cluster_ids: typing.Collection[str],
    ) -> typing.Iterable[Protein]:

        for dataset in self.datasets:
            yield from dataset.extract_proteins(progress, cluster_ids)
