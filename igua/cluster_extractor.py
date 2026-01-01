import gc
import json
import re
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple
import gzip
import pickle

import polars as pl
import rich.progress
from rich.console import Console


_GZIP_MAGIC = b'\x1f\x8b'


@dataclass
class SystemCoordinates:
    """Genomic coordinates for a gene cluster."""

    sys_id: str
    seq_id: str
    start_coord: int
    end_coord: int
    strand: str
    genes: List[str]
    fasta_file: str
    valid: bool = True
    error_msg: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "SystemCoordinates":
        """Create from dictionary."""
        return cls(**data)


class GFFIndex:
    """Fast, in-memory GFF index."""

    def __init__(self, gff_path: Path):
        self.path = gff_path
        self._index: Dict[str, Dict] = {}
        self._build_index()

    def _build_index(self):
        """Build comprehensive ID index (runs once)."""
        with open(self.path, "r") as f:
            for line in f:
                if line.startswith("#") or not line.strip():
                    continue

                parts = line.strip().split("\t")
                if len(parts) < 9 or parts[2] not in ["gene", "CDS"]:
                    continue

                seqid, _, ftype, start, end, _, strand, _, attrs = parts
                attr_dict = dict(
                    item.split("=", 1) for item in attrs.split(";") if "=" in item
                )

                feature = {
                    "seqid": seqid,
                    "type": ftype,
                    "start": int(start),
                    "end": int(end),
                    "strand": strand,
                    "attributes": attr_dict,
                }

                for key in [
                    "ID",
                    "locus_tag",
                    "Name",
                    "gene",
                    "old_locus_tag",
                    "protein_id",
                ]:
                    if val := attr_dict.get(key):
                        for variant in [
                            val,
                            f"gene-{val}",
                            f"cds-{val}",
                            val.replace("_", "~"),
                            val.replace("~", "_"),
                        ]:
                            self._index[variant] = feature

    def get(self, gene_id: str) -> Optional[Dict]:
        """O(1) lookup."""
        return self._index.get(gene_id)

    def __contains__(self, gene_id: str) -> bool:
        return gene_id in self._index


class FastaReader:
    """Streaming FASTA reader."""

    @staticmethod
    def read_fasta(file_path: Path) -> Iterable[Tuple[str, str]]:
        """Stream FASTA records from file."""
        opener = gzip.open if file_path.suffix == '.gz' else open
        
        with opener(file_path, 'rt') as f:
            name = None
            sequence = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    if name is not None:
                        yield name, ''.join(sequence)
                    name = line[1:].split()[0]
                    sequence = []
                else:
                    sequence.append(line)
            
            if name is not None:
                yield name, ''.join(sequence)


class ProteinIndex:
    """In-memory protein index for fast lookup."""
    
    def __init__(self, protein_fasta: Path):
        self.path = protein_fasta
        self._sequences: Dict[str, str] = {}
        self._headers: Dict[str, str] = {}
        self._build_index()
    
    def _build_index(self):
        """Build protein sequence index."""
        opener = gzip.open if self.path.suffix == '.gz' else open
        
        with opener(self.path, 'rt') as f:
            seq_id = None
            full_header = None
            sequence = []
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if line.startswith('>'):
                    if seq_id is not None:
                        self._sequences[seq_id] = ''.join(sequence)
                        self._headers[seq_id] = full_header
                    
                    full_header = line[1:]
                    seq_id = full_header.split()[0]
                    sequence = []
                else:
                    sequence.append(line)
            
            if seq_id is not None:
                self._sequences[seq_id] = ''.join(sequence)
                self._headers[seq_id] = full_header
    
    def get(self, protein_id: str) -> Optional[str]:
        """Get protein sequence by ID."""
        return self._sequences.get(protein_id)
    
    def get_header(self, protein_id: str) -> Optional[str]:
        """Get full header by ID."""
        return self._headers.get(protein_id)
    
    def keys(self):
        """Return all protein IDs."""
        return self._sequences.keys()
    
    def __contains__(self, protein_id: str) -> bool:
        return protein_id in self._sequences


class GenomeContext:
    """Immutable data container for genome/strain file paths and metadata."""

    def __init__(
        self,
        genome_id: Optional[str],
        systems_tsv: Path,
        genes_tsv: Path,
        gff_file: Path,
        genomic_fasta: Path,
        protein_fasta: Path,
        activity_filter: str = "defense",
    ):
        self.genome_id = genome_id if genome_id else str(uuid.uuid4())[:8]
        self.systems_tsv = Path(systems_tsv)
        self.genes_tsv = Path(genes_tsv)
        self.gff_file = Path(gff_file)
        self.genomic_fasta = Path(genomic_fasta)
        self.protein_fasta = Path(protein_fasta)
        self.activity_filter = activity_filter

        self.missing_files = [
            f"{name}: {path}"
            for path, name in [
                (self.systems_tsv, "systems_tsv"),
                (self.genes_tsv, "genes_tsv"),
                (self.gff_file, "gff_file"),
                (self.genomic_fasta, "genomic_fasta"),
                (self.protein_fasta, "protein_fasta"),
            ]
            if not path.exists()
        ]

    def __repr__(self):
        return f"GenomeContext(genome_id='{self.genome_id}', files={5 - len(self.missing_files)}/5)"

    def is_valid(self) -> bool:
        return len(self.missing_files) == 0


class GenomeResources:
    """Manages lazy-loading and caching of genome resources."""

    def __init__(self, context: GenomeContext, console: Console):
        self.context = context
        self.console = console
        self._systems_df: Optional[pl.DataFrame] = None
        self._genes_df: Optional[pl.DataFrame] = None
        self._protein_idx: Optional[ProteinIndex] = None
        self._gff_db: Optional[GFFIndex] = None
        self._coordinates_cache: Optional[List[SystemCoordinates]] = None

    @property
    def systems_df(self) -> pl.DataFrame:
        """Load and filter clusters TSV (lazy-loaded, cached)"""
        if self._systems_df is not None:
            return self._systems_df

        df = pl.read_csv(self.context.systems_tsv, separator="\t")
        original_count = len(df)

        if self.context.activity_filter.lower() != "all":
            if "activity" in df.columns:
                df = df.filter(
                    pl.col("activity").str.to_lowercase()
                    == self.context.activity_filter.lower()
                )
                self.console.print(
                    f"[bold green]{'Filtered':>12}[/] {original_count} → {len(df)} "
                    f"([bold cyan]{self.context.activity_filter}[/] only)"
                )
            else:
                self.console.print(
                    f"[bold yellow]{'Warning':>12}[/] No 'activity' column found"
                )
        else:
            self.console.print(
                f"[bold blue]{'Processing':>12}[/] all {original_count} clusters"
            )

        if (
            n_dup := df.filter(pl.col("sys_id").is_duplicated())
            .select("sys_id")
            .unique()
            .height
        ) > 0:
            dup_ids = (
                df.filter(pl.col("sys_id").is_duplicated())
                .select("sys_id")
                .unique()
                .to_series()
                .to_list()
            )
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] {n_dup} duplicate system/s: "
                f"[cyan]{', '.join(dup_ids[:5])}{'...' if n_dup > 5 else ''}[/]"
            )
            df = df.unique(subset=["sys_id"], keep="first")

        self._systems_df = df
        return self._systems_df

    @property
    def genes_df(self) -> pl.DataFrame:
        """Load genes TSV (lazy-loaded, cached)"""
        if self._genes_df is not None:
            return self._genes_df

        try:
            self._genes_df = pl.read_csv(
                self.context.genes_tsv,
                separator="\t",
                columns=["sys_id", "hit_id", "hit_pos"],
            )
            self.console.print(
                f"[bold blue]{'Loaded':>12}[/] genes TSV with {len(self._genes_df)} genes"
            )
        except Exception as e:
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] Failed to load genes_tsv: {e}"
            )
            self._genes_df = pl.DataFrame()

        return self._genes_df

    @property
    def protein_idx(self) -> ProteinIndex:
        if self._protein_idx is None:
            self._protein_idx = ProteinIndex(self.context.protein_fasta)
        return self._protein_idx

    @property
    def gff_db(self) -> GFFIndex:
        if self._gff_db is None:
            self._gff_db = GFFIndex(self.context.gff_file)
        return self._gff_db

    @property
    def coordinates(self) -> List[SystemCoordinates]:
        """Build and cache system coordinates (lazy-loaded, cached)"""
        if self._coordinates_cache is None:
            self._coordinates_cache = self._build_coordinates()
        return self._coordinates_cache

    def _build_coordinates(self) -> List[SystemCoordinates]:
        """Build coordinates for all clusters."""
        coordinates = []

        for row in self.systems_df.iter_rows(named=True):
            coord = self._parse_system_coordinates(row)
            coordinates.append(coord)

        return coordinates

    def _parse_system_coordinates(self, row: dict) -> SystemCoordinates:
        """Parse coordinates for a single system."""
        sys_id = row["sys_id"]

        sys_beg_gene = row.get("sys_beg")
        sys_end_gene = row.get("sys_end")

        if sys_beg_gene is None or sys_end_gene is None:
            return self._invalid_coord(
                sys_id,
                [],
                "Missing sys_beg or sys_end columns in systems TSV",
            )

        try:
            gene_list = [
                g.strip() for g in row["protein_in_syst"].split(",") if g.strip()
            ]
        except (KeyError, AttributeError):
            gene_list = self._get_genes_from_genes_tsv(sys_id)
            if not gene_list:
                return self._invalid_coord(
                    sys_id,
                    [],
                    "Missing protein_in_syst column and no genes found in genes_tsv",
                )

        if not gene_list:
            return self._invalid_coord(sys_id, [], "Empty gene list")

        beg_feature = self.gff_db.get(sys_beg_gene)
        end_feature = self.gff_db.get(sys_end_gene)

        if beg_feature is None:
            return self._invalid_coord(
                sys_id, gene_list, f"Start gene '{sys_beg_gene}' not found in GFF"
            )

        if end_feature is None:
            return self._invalid_coord(
                sys_id, gene_list, f"End gene '{sys_end_gene}' not found in GFF"
            )

        seq_id_beg = beg_feature["seqid"]
        seq_id_end = end_feature["seqid"]

        if seq_id_beg != seq_id_end:
            return self._invalid_coord(
                sys_id,
                gene_list,
                f"Start and end genes on different contigs: {seq_id_beg} vs {seq_id_end}",
            )

        seq_id = seq_id_beg

        start = min(
            beg_feature["start"], beg_feature["end"],
            end_feature["start"], end_feature["end"]
        )
        end = max(
            beg_feature["start"], beg_feature["end"],
            end_feature["start"], end_feature["end"]
        )
        
        strand = beg_feature["strand"]

        region_size = end - start + 1
        if region_size > 1e5:
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] System [bold cyan]{sys_id}[/] unusually large: {region_size} bp"
            )
        elif region_size < 50:
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] System [bold cyan]{sys_id}[/] unusually small: {region_size} bp"
            )

        return SystemCoordinates(
            sys_id=sys_id,
            seq_id=seq_id,
            start_coord=start,
            end_coord=end,
            strand=strand,
            genes=gene_list,
            fasta_file=str(self.context.genomic_fasta),
            valid=True,
        )

    def _get_genes_from_genes_tsv(self, sys_id: str) -> List[str]:
        """
        Fallback method to extract gene list from genes_tsv file.
        Uses the cached genes_df property.
        """
        if self.genes_df.is_empty():
            return []

        genes = (
            self.genes_df.filter(pl.col("sys_id") == sys_id)
            .sort("hit_pos")
            .select("hit_id")
            .to_series()
            .to_list()
        )

        return genes

    def _invalid_coord(
        self, sys_id: str, genes: List[str], error: str, seq_id: str = ""
    ) -> SystemCoordinates:
        """Create an invalid SystemCoordinates object."""
        self.console.print(
            f"[bold yellow]{'Warning':>12}[/] {error} for system [bold cyan]{sys_id}[/]"
        )
        return SystemCoordinates(
            sys_id=sys_id,
            seq_id=seq_id,
            start_coord=0,
            end_coord=0,
            strand="",
            genes=genes,
            fasta_file=str(self.context.genomic_fasta),
            valid=False,
            error_msg=error,
        )

    def cleanup(self):
        """Clean up resources to enable garbage collection."""
        self._systems_df = None
        self._genes_df = None
        self._protein_idx = None
        self._gff_db = None
        self._coordinates_cache = None


class SequenceExtractor:
    """Handles extraction of genomic and protein sequences."""

    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self._protein_lookup_cache: Optional[Dict[str, str]] = None
        self._cached_protein_fasta: Optional[Path] = None

    def extract_genomic_sequences(
        self,
        coordinates: List[SystemCoordinates],
        genomic_fasta: Path,
        output_file: TextIO,
    ) -> List[Tuple[str, int, str]]:
        """Extract genomic sequences by streaming FASTA file."""
        valid_coords = [c for c in coordinates if c.valid]

        if not valid_coords:
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] No valid clusters to extract"
            )
            return []

        contig_groups: Dict[str, List[SystemCoordinates]] = {}
        for coord in valid_coords:
            contig_groups.setdefault(coord.seq_id, []).append(coord)

        self.console.print(
            f"[bold blue]{'Processing':>12}[/] {len(valid_coords)} clusters across {len(contig_groups)} contigs"
        )

        results = []
        
        for seq_id, sequence in FastaReader.read_fasta(genomic_fasta):
            if seq_id not in contig_groups:
                continue
            
            if self.verbose:
                self.console.print(
                    f"[bold blue]{'Loading':>12}[/] contig {seq_id} ({len(contig_groups[seq_id])} clusters)"
                )
            
            for coord in contig_groups[seq_id]:
                try:
                    subseq = sequence[coord.start_coord - 1 : coord.end_coord]
                    output_file.write(f">{coord.sys_id}\n{subseq}\n")
                    
                    if self.verbose:
                        self.console.print(
                            f"[bold green]{'Extracted':>12}[/] {coord.sys_id} ({len(subseq)} bp)"
                        )
                    
                    results.append((coord.sys_id, len(subseq), coord.fasta_file))
                except Exception as e:
                    self.console.print(
                        f"[bold red]{'Error':>12}[/] Failed to extract {coord.sys_id}: {e}"
                    )
            
            del contig_groups[seq_id]
            
            if not contig_groups:
                break
        
        if contig_groups:
            for seq_id in contig_groups:
                self.console.print(
                    f"[bold red]{'Error':>12}[/] Contig {seq_id} not found in genome"
                )

        return results

    def extract_proteins_from_gene_list(
        self,
        system_genes: Dict[str, List[str]],
        protein_idx: ProteinIndex,
        protein_fasta_path: Path,
        output_file: TextIO,
    ) -> Dict[str, int]:
        """Extract protein sequences from a pre-computed gene list mapping.
        
        Args:
            system_genes: Dict mapping sys_id to list of gene IDs
            protein_idx: Protein sequence index
            protein_fasta_path: Path to protein fasta file (for cache tracking)
            output_file: Output file handle
            
        Returns:
            Dict mapping protein_id to sequence length
        """
        if self._cached_protein_fasta != protein_fasta_path:
            self._protein_lookup_cache = None
            self._cached_protein_fasta = protein_fasta_path
        
        total_genes = sum(len(genes) for genes in system_genes.values())
        self.console.print(
            f"[bold blue]{'Processing':>12}[/] {total_genes} proteins from {len(system_genes)} systems"
        )

        protein_sizes = {}
        for sys_id, gene_list in system_genes.items():
            for gene_id in gene_list:
                if seq := self._get_protein_sequence(gene_id, protein_idx):
                    protein_id = f"{sys_id}__{gene_id}"
                    output_file.write(f">{protein_id}\n{seq}\n")
                    protein_sizes[protein_id] = len(seq)
                else:
                    self.console.print(
                        f"[bold yellow]{'Warning':>12}[/] Protein {gene_id} not found for system [bold cyan]{sys_id}[/]"
                    )

        return protein_sizes

    def _get_protein_sequence(self, gene_id: str, protein_idx: ProteinIndex) -> Optional[str]:
        """Get protein sequence with fallback search."""
        seq = protein_idx.get(gene_id)
        if seq:
            return seq
        return self._fallback_protein_search(gene_id, protein_idx)

    def _fallback_protein_search(
        self, gene_id: str, protein_idx: ProteinIndex
    ) -> Optional[str]:
        """Fallback search using header attributes."""
        if self._protein_lookup_cache is None:
            self._protein_lookup_cache = {
                rec_id: protein_idx.get_header(rec_id) for rec_id in protein_idx.keys()
            }

        for attr in ["locus_tag", "ID", "Name", "gene"]:
            for rec_id, header in self._protein_lookup_cache.items():
                if header and re.search(rf"\[{attr}=({re.escape(gene_id)})\]", header):
                    return protein_idx.get(rec_id)

        return None

    def cleanup(self):
        """Clean up extractor's internal caches."""
        self._protein_lookup_cache = None
        self._cached_protein_fasta = None


class ClusterMetadataCache:
    """Manages caching of cluster metadata to avoid redundant processing."""
    
    def __init__(self, cache_path: Path):
        self.cache_path = cache_path
        self._metadata: Optional[Dict] = None
    
    def save(self, metadata: Dict) -> None:
        """Save metadata to pickle file."""
        with open(self.cache_path, 'wb') as f:
            pickle.dump(metadata, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    def load(self) -> Dict:
        """Load metadata from pickle file."""
        if self._metadata is None:
            with open(self.cache_path, 'rb') as f:
                self._metadata = pickle.load(f)
        return self._metadata
    
    def exists(self) -> bool:
        """Check if cache file exists."""
        return self.cache_path.exists()
    
    def clear(self) -> None:
        """Clear cache file and memory."""
        if self.cache_path.exists():
            self.cache_path.unlink()
        self._metadata = None


class GeneClusterExtractor:
    """High-level orchestrator for gene cluster extraction pipeline."""

    def __init__(
        self, progress: Optional[rich.progress.Progress] = None, verbose: bool = False
    ):
        self.progress = progress
        self.console = progress.console if progress else Console()
        self.verbose = verbose
        self._extractor = SequenceExtractor(self.console, verbose)

    def extract_systems(
        self,
        context: GenomeContext,
        output_file: TextIO,
    ) -> List[SystemCoordinates]:
        """Extract genomic sequences and return coordinates with metadata."""
        if not context.is_valid():
            self._log_missing_files(context)
            return []

        resources = GenomeResources(context, self.console)
        
        try:
            self.console.print(f"[bold blue]{'Building':>12}[/] cluster coordinates")
            coordinates = resources.coordinates

            valid_count = sum(1 for c in coordinates if c.valid)
            self.console.print(
                f"[bold green]{'Validated':>12}[/] {valid_count}/{len(coordinates)} clusters"
            )

            self.console.print(f"[bold blue]{'Extracting':>12}[/] cluster sequences")
            results = self._extractor.extract_genomic_sequences(
                coordinates, context.genomic_fasta, output_file
            )

            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(results)} gene clusters for [bold cyan]{context.genome_id}[/]"
            )
            
            return coordinates

        except Exception as e:
            self.console.print(
                f"[bold red]{'Error':>12}[/] Fatal error for {context.genome_id}: {e}"
            )
            import traceback
            traceback.print_exc()
            return []
        finally:
            resources.cleanup()
            gc.collect()

    def extract_proteins_from_metadata(
        self,
        metadata: Dict,
        output_file: TextIO,
        representatives: Optional[Iterable[str]] = None,
    ) -> Dict[str, int]:
        """Extract protein sequences using pre-computed metadata.
        
        Args:
            metadata: Dictionary containing genome_id, protein_fasta, and coordinates
            output_file: Output file handle
            representatives: Optional set of representative system IDs
            
        Returns:
            Dict mapping protein_id to sequence length
        """
        genome_id = metadata["genome_id"]
        protein_fasta = Path(metadata["protein_fasta"])
        coordinates = [SystemCoordinates.from_dict(c) for c in metadata["coordinates"]]
        
        if representatives:
            rep_set = set(representatives) if not isinstance(representatives, set) else representatives
            coordinates = [c for c in coordinates if c.sys_id in rep_set]
        
        valid_coords = [c for c in coordinates if c.valid]
        
        if not valid_coords:
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] No valid clusters for {genome_id}"
            )
            return {}
        
        system_genes = {c.sys_id: c.genes for c in valid_coords}
        
        try:
            protein_idx = ProteinIndex(protein_fasta)
            protein_sizes = self._extractor.extract_proteins_from_gene_list(
                system_genes, protein_idx, protein_fasta, output_file
            )
            
            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from "
                f"{len(valid_coords)} clusters for [bold cyan]{genome_id}[/]"
            )
            
            return protein_sizes
            
        except Exception as e:
            self.console.print(
                f"[bold red]{'Error':>12}[/] Fatal error for {genome_id}: {e}"
            )
            import traceback
            traceback.print_exc()
            return {}

    def _log_missing_files(self, context: GenomeContext):
        """Log missing files for a context."""
        self.console.print(
            f"[bold red]Error:[/] Missing files for {context.genome_id}:"
        )
        for missing in context.missing_files:
            self.console.print(f"  - {missing}")