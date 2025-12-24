import gc
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple

import polars as pl
import rich.progress
from pyfaidx import Fasta
from rich.console import Console


@dataclass
class SystemCoordinates:
    """Genomic coordinates for a gene cluster."""
    sys_id: str
    seq_id: str
    start_coord: int
    end_coord: int
    strand: str
    genes: List[str]
    fasta_file: Path
    valid: bool = True
    error_msg: Optional[str] = None


class GFFIndex:
    """Fast, in-memory GFF index."""
    
    def __init__(self, gff_path: Path):
        self.path = gff_path
        self._index: Dict[str, Dict] = {}
        self._build_index()
    
    def _build_index(self):
        """Build comprehensive ID index (runs once)."""
        with open(self.path, 'r') as f:
            for line in f:
                if line.startswith('#') or not line.strip():
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9 or parts[2] not in ['gene', 'CDS']:
                    continue
                
                seqid, _, ftype, start, end, _, strand, _, attrs = parts
                attr_dict = dict(item.split('=', 1) for item in attrs.split(';') if '=' in item)
                
                feature = {
                    'seqid': seqid,
                    'type': ftype,
                    'start': int(start),
                    'end': int(end),
                    'strand': strand,
                    'attributes': attr_dict
                }
                
                for key in ['ID', 'locus_tag', 'Name', 'gene', 'old_locus_tag', 'protein_id']:
                    if val := attr_dict.get(key):
                        for variant in [val, f"gene-{val}", f"cds-{val}", 
                                       val.replace('_', '~'), val.replace('~', '_')]:
                            self._index[variant] = feature
    
    def get(self, gene_id: str) -> Optional[Dict]:
        """O(1) lookup."""
        return self._index.get(gene_id)
    
    def get_batch(self, gene_ids: List[str]) -> Dict[str, Optional[Dict]]:
        """Batch lookup for multiple genes."""
        return {gene_id: self._index.get(gene_id) for gene_id in gene_ids}
    
    def __getitem__(self, gene_id: str) -> Dict:
        if (feature := self._index.get(gene_id)) is None:
            raise KeyError(f"Gene ID '{gene_id}' not found")
        return feature
    
    def __contains__(self, gene_id: str) -> bool:
        return gene_id in self._index


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
            ] if not path.exists()
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
        self._genome_idx: Optional[Fasta] = None
        self._protein_idx: Optional[Fasta] = None
        self._gff_db: Optional[GFFIndex] = None
        self._coordinates_cache: Optional[List[SystemCoordinates]] = None

    @property
    def systems_df(self) -> pl.DataFrame:
        """Load and filter systems TSV (lazy-loaded, cached)"""
        if self._systems_df is not None:
            return self._systems_df
            
        df = pl.read_csv(self.context.systems_tsv, separator="\t")
        original_count = len(df)

        if self.context.activity_filter.lower() != "all":
            if "activity" in df.columns:
                df = df.filter(pl.col("activity").str.to_lowercase() == self.context.activity_filter.lower())
                self.console.print(
                    f"[bold green]{'Filtered':>12}[/] {original_count} → {len(df)} "
                    f"([bold cyan]{self.context.activity_filter}[/] only)"
                )
            else:
                self.console.print(f"[bold yellow]{'Warning':>12}[/] No 'activity' column found")
        else:
            self.console.print(f"[bold blue]{'Processing':>12}[/] all {original_count} systems")

        if (n_dup := df.filter(pl.col("sys_id").is_duplicated()).select("sys_id").unique()).height > 0:
            dup_ids = df.filter(pl.col("sys_id").is_duplicated()).select("sys_id").unique().to_series().to_list()
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] {n_dup} duplicate system/s: "
                f"[cyan]{', '.join(dup_ids[:5])}{'...' if n_dup > 5 else ''}[/]"
            )
            df = df.unique(subset=["sys_id"], keep="first")

        self._systems_df = df
        return self._systems_df

    def filter_systems_by_representatives(self, representatives: Iterable[str]):
        """Filter systems to only include representatives"""
        if self._systems_df is None:
            _ = self.systems_df
        
        rep_list = list(representatives) if not isinstance(representatives, list) else representatives
        self._systems_df = self._systems_df.filter(pl.col("sys_id").is_in(rep_list))
        self._coordinates_cache = None

    @property
    def genome_idx(self) -> Fasta:
        if self._genome_idx is None:
            self._genome_idx = Fasta(str(self.context.genomic_fasta))
        return self._genome_idx

    @property
    def protein_idx(self) -> Fasta:
        if self._protein_idx is None:
            self._protein_idx = Fasta(str(self.context.protein_fasta))
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
        """Build coordinates for all systems."""
        coordinates = []
        
        for row in self.systems_df.iter_rows(named=True):
            coord = self._parse_system_coordinates(row)
            coordinates.append(coord)
        
        return coordinates
    
    def _parse_system_coordinates(self, row: dict) -> SystemCoordinates:
        """Parse coordinates for a single system."""
        sys_id = row['sys_id']
        
        try:
            gene_list = [g.strip() for g in row['protein_in_syst'].split(',') if g.strip()]
        except (KeyError, AttributeError):
            return self._invalid_coord(sys_id, [], "Missing protein_in_syst column")
        
        if not gene_list:
            return self._invalid_coord(sys_id, [], "Empty gene list")
        
        found_features = {
            gid: feat for gid, feat in self.gff_db.get_batch(gene_list).items() 
            if feat is not None
        }
        
        if not found_features:
            return self._invalid_coord(sys_id, gene_list, "No genes found in GFF")
        
        seq_ids = set(feat['seqid'] for feat in found_features.values())
        if len(seq_ids) > 1:
            return self._invalid_coord(sys_id, gene_list, f"Spans multiple contigs/sequences: {seq_ids}")
        
        seq_id = list(seq_ids)[0]
        
        if seq_id not in self.genome_idx:
            return self._invalid_coord(sys_id, gene_list, f"Contig/sequence {seq_id} not in genome", seq_id)
        
        start = min(min(f['start'], f['end']) for f in found_features.values())
        end = max(max(f['start'], f['end']) for f in found_features.values())
        strand = list(found_features.values())[0]['strand']
        
        region_size = end - start + 1
        if region_size > 1e5:
            self.console.print(f"[bold yellow]{'Warning':>12}[/] System [bold cyan]{sys_id}[/] unusually large: {region_size} bp")
        elif region_size < 50:
            self.console.print(f"[bold yellow]{'Warning':>12}[/] System [bold cyan]{sys_id}[/] unusually small: {region_size} bp")
        
        return SystemCoordinates(
            sys_id=sys_id,
            seq_id=seq_id,
            start_coord=start,
            end_coord=end,
            strand=strand,
            genes=gene_list,
            fasta_file=self.context.genomic_fasta,
            valid=True
        )
    
    def _invalid_coord(self, sys_id: str, genes: List[str], error: str, seq_id: str = "") -> SystemCoordinates:
        """Create an invalid SystemCoordinates object."""
        self.console.print(f"[bold yellow]{'Warning':>12}[/] {error} for system [bold cyan]{sys_id}[/]")
        return SystemCoordinates(
            sys_id=sys_id,
            seq_id=seq_id,
            start_coord=0,
            end_coord=0,
            strand="",
            genes=genes,
            fasta_file=self.context.genomic_fasta,
            valid=False,
            error_msg=error
        )

    def cleanup(self):
        """Clean up resources to enable garbage collection."""
        self._systems_df = None
        self._genome_idx = None
        self._protein_idx = None
        self._gff_db = None
        self._coordinates_cache = None


class SequenceExtractor:
    """Handles extraction of genomic and protein sequences."""
    
    def __init__(self, console: Console, verbose: bool = False):
        self.console = console
        self.verbose = verbose
        self._protein_lookup_cache: Optional[Dict[str, str]] = None
    
    def extract_genomic_sequences(
        self, 
        coordinates: List[SystemCoordinates],
        genome_idx: Fasta,
        output_file: TextIO
    ) -> List[Tuple[str, int, str]]:
        """Extract genomic sequences grouped by contig for efficiency."""
        valid_coords = [c for c in coordinates if c.valid]
        
        if not valid_coords:
            self.console.print(f"[bold yellow]{'Warning':>12}[/] No valid systems to extract")
            return []
        
        contig_groups = {}
        for coord in valid_coords:
            contig_groups.setdefault(coord.seq_id, []).append(coord)
        
        self.console.print(
            f"[bold blue]{'Processing':>12}[/] {len(valid_coords)} systems across {len(contig_groups)} contigs"
        )
        
        results = []
        for seq_id, systems in contig_groups.items():
            results.extend(self._extract_from_contig(seq_id, systems, genome_idx, output_file))
        
        return results
    
    def _extract_from_contig(
        self,
        seq_id: str,
        systems: List[SystemCoordinates],
        genome_idx: Fasta,
        output_file: TextIO
    ) -> List[Tuple[str, int, str]]:
        """Extract all systems from a single contig."""
        if self.verbose:
            self.console.print(f"[bold blue]{'Loading':>12}[/] contig {seq_id} ({len(systems)} systems)")
        
        results = []
        try:
            contig_seq = genome_idx[seq_id]
            
            for coord in systems:
                try:
                    seq = str(contig_seq[coord.start_coord - 1:coord.end_coord])
                    output_file.write(f">{coord.sys_id}\n{seq}\n")
                    
                    if self.verbose:
                        self.console.print(f"[bold green]{'Extracted':>12}[/] {coord.sys_id} ({len(seq)} bp)")
                    
                    results.append((coord.sys_id, len(seq), str(coord.fasta_file)))
                except Exception as e:
                    self.console.print(f"[bold red]{'Error':>12}[/] Failed to extract {coord.sys_id}: {e}")
            
            del contig_seq
            
        except KeyError:
            self.console.print(f"[bold red]{'Error':>12}[/] Contig {seq_id} not found in genome")
        except Exception as e:
            self.console.print(f"[bold red]{'Error':>12}[/] Failed to load contig {seq_id}: {e}")
        
        return results
    
    def extract_proteins(
        self,
        coordinates: List[SystemCoordinates],
        protein_idx: Fasta,
        output_file: TextIO
    ) -> Dict[str, int]:
        """Extract protein sequences for all systems."""
        valid_coords = [c for c in coordinates if c.valid]
        
        if not valid_coords:
            self.console.print(f"[bold yellow]{'Warning':>12}[/] No valid systems for protein extraction")
            return {}
        
        total_genes = sum(len(c.genes) for c in valid_coords)
        self.console.print(f"[bold blue]{'Processing':>12}[/] {total_genes} proteins from {len(valid_coords)} systems")
        
        protein_sizes = {}
        for coord in valid_coords:
            for gene_id in coord.genes:
                if seq := self._get_protein_sequence(gene_id, protein_idx):
                    protein_id = f"{coord.sys_id}__{gene_id}"
                    output_file.write(f">{protein_id}\n{seq}\n")
                    protein_sizes[protein_id] = len(seq)
                else:
                    self.console.print(
                        f"[bold yellow]{'Warning':>12}[/] Protein {gene_id} not found for system [bold cyan]{coord.sys_id}[/]"
                    )
        
        return protein_sizes
    
    def _get_protein_sequence(self, gene_id: str, protein_idx: Fasta) -> Optional[str]:
        """Get protein sequence with fallback search."""
        try:
            return str(protein_idx[gene_id])
        except KeyError:
            return self._fallback_protein_search(gene_id, protein_idx)
    
    def _fallback_protein_search(self, gene_id: str, protein_idx: Fasta) -> Optional[str]:
        """Fallback search using header attributes."""
        if self._protein_lookup_cache is None:
            self.console.print(f"[bold yellow]{'Building':>12}[/] fallback protein lookup map")
            self._protein_lookup_cache = {
                rec_id: protein_idx[rec_id].long_name 
                for rec_id in protein_idx.keys()
            }
        
        for attr in ['locus_tag', 'ID', 'Name', 'gene']:
            for rec_id, header in self._protein_lookup_cache.items():
                if re.search(rf'\[{attr}=({re.escape(gene_id)})\]', header):
                    return str(protein_idx[rec_id])
        
        return None


class GeneClusterExtractor:
    """High-level orchestrator for gene cluster extraction pipeline."""
    
    def __init__(self, progress: Optional[rich.progress.Progress] = None, verbose: bool = False):
        self.progress = progress
        self.console = progress.console if progress else Console()
        self.verbose = verbose
        self._cached_resources: Optional[GenomeResources] = None
        self._extractor = SequenceExtractor(self.console, verbose)
    
    def extract_systems(
        self,
        context: GenomeContext,
        output_file: TextIO,
        representatives: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, int, str]]:
        """Extract genomic sequences for gene clusters."""
        if not context.is_valid():
            self._log_missing_files(context)
            return []
        
        self._cached_resources = GenomeResources(context, self.console)
        
        try:
            if representatives:
                self._cached_resources.filter_systems_by_representatives(representatives)
            
            self.console.print(f"[bold blue]{'Building':>12}[/] system coordinates")
            coordinates = self._cached_resources.coordinates
            
            valid_count = sum(1 for c in coordinates if c.valid)
            self.console.print(f"[bold green]{'Validated':>12}[/] {valid_count}/{len(coordinates)} gene clusters")
            
            self.console.print(f"[bold blue]{'Extracting':>12}[/] cluster sequences")
            results = self._extractor.extract_genomic_sequences(
                coordinates, self._cached_resources.genome_idx, output_file
            )
            
            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(results)} gene clusters for [bold cyan]{context.genome_id}[/]"
            )
            return results
            
        except Exception as e:
            self.console.print(f"[bold red]{'Error':>12}[/] Fatal error for {context.genome_id}: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            if not self._should_keep_cache():
                self._cleanup_cached_resources()
    
    def extract_proteins(
        self,
        context: GenomeContext,
        output_file: TextIO,
        representatives: Optional[Iterable[str]] = None
    ) -> Dict[str, int]:
        """Extract protein sequences for gene clusters."""
        resources = self._get_or_create_resources(context)
        if resources is None:
            return {}
        
        try:
            if representatives and not self._cached_resources:
                resources.filter_systems_by_representatives(representatives)
            
            self.console.print(f"[bold blue]{'Loading':>12}[/] coordinates")
            coordinates = resources.coordinates
            
            self.console.print(f"[bold blue]{'Extracting':>12}[/] protein sequences")
            protein_sizes = self._extractor.extract_proteins(
                coordinates, resources.protein_idx, output_file
            )
            
            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from "
                f"{sum(1 for c in coordinates if c.valid)} gene clusters for [bold cyan]{context.genome_id}[/]"
            )
            return protein_sizes
            
        except Exception as e:
            self.console.print(f"[bold red]{'Error':>12}[/] Fatal error for {context.genome_id}: {e}")
            import traceback
            traceback.print_exc()
            return {}
        finally:
            self._cleanup_cached_resources()
    
    def _get_or_create_resources(self, context: GenomeContext) -> Optional[GenomeResources]:
        """Get cached resources or create new ones."""
        if self._cached_resources and self._cached_resources.context.genome_id == context.genome_id:
            self.console.print(f"[bold blue]{'Reusing':>12}[/] cached genome resources")
            return self._cached_resources
        
        if not context.is_valid():
            self._log_missing_files(context)
            return None
        
        return GenomeResources(context, self.console)
    
    def _should_keep_cache(self) -> bool:
        """Determine if cache should be kept for next operation."""
        return self._cached_resources is not None
    
    def _cleanup_cached_resources(self):
        """Clean up cached resources and trigger garbage collection."""
        if self._cached_resources:
            self._cached_resources.cleanup()
            self._cached_resources = None
        gc.collect()
    
    def _log_missing_files(self, context: GenomeContext):
        """Log missing files for a context."""
        self.console.print(f"[bold red]Error:[/] Missing files for {context.genome_id}:")
        for missing in context.missing_files:
            self.console.print(f"  - {missing}")