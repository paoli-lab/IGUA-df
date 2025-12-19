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
                if line.startswith('#'):
                    continue
                
                parts = line.strip().split('\t')
                if len(parts) < 9:
                    continue
                
                seqid, source, ftype, start, end, score, strand, phase, attrs = parts
                
                if ftype not in ['gene', 'CDS']:
                    continue
                
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
                    val = attr_dict.get(key)
                    if val:
                        self._index[val] = feature
                        self._index[f"gene-{val}"] = feature
                        self._index[f"cds-{val}"] = feature
                        self._index[val.replace('_', '~')] = feature
                        self._index[val.replace('~', '_')] = feature
    
    def get(self, gene_id: str) -> Optional[Dict]:
        """O(1) lookup."""
        return self._index.get(gene_id)
    
    def get_batch(self, gene_ids: List[str]) -> Dict[str, Optional[Dict]]:
        """Batch lookup for multiple genes."""
        return {gene_id: self._index.get(gene_id) for gene_id in gene_ids}
    
    def __getitem__(self, gene_id: str) -> Dict:
        """Dict-like access."""
        feature = self._index.get(gene_id)
        if feature is None:
            raise KeyError(f"Gene ID '{gene_id}' not found")
        return feature
    
    def __contains__(self, gene_id: str) -> bool:
        """Check if gene exists."""
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
        
        self.missing_files = []
        for file_path, name in [
            (self.systems_tsv, "systems_tsv"),
            (self.genes_tsv, "genes_tsv"),
            (self.gff_file, "gff_file"),
            (self.genomic_fasta, "genomic_fasta"),
            (self.protein_fasta, "protein_fasta"),
        ]:
            if not file_path.exists():
                self.missing_files.append(f"{name}: {file_path}")
    
    def __repr__(self):
        return f"GenomeContext(genome_id='{self.genome_id}', files={5 - len(self.missing_files)}/5)"

    def is_valid(self) -> bool:
        """Check if all required files exist"""
        return len(self.missing_files) == 0


class GenomeResources:
    """Manages lazy-loading and caching of genome resources."""

    def __init__(
        self, 
        context: GenomeContext, 
        console: Console,
    ):
        self.context = context
        self.console = console

        self._systems_df: Optional[pl.DataFrame] = None
        self._genome_idx: Optional[Fasta] = None
        self._protein_idx: Optional[Fasta] = None
        self._gff_db: Optional[GFFIndex] = None

    @property
    def systems_df(self) -> pl.DataFrame:
        """Load and filter systems TSV (lazy-loaded, cached)"""
        if self._systems_df is None:
            df = pl.read_csv(self.context.systems_tsv, separator="\t")
            original_count = len(df)

            if self.context.activity_filter.lower() != "all":
                if "activity" in df.columns:
                    df = df.filter(
                        pl.col("activity").str.to_lowercase() == self.context.activity_filter.lower()
                    )
                    self.console.print(
                        f"[bold green]{'Filtered':>12}[/] {original_count} systems to {len(df)} "
                        f"([bold cyan]{self.context.activity_filter}[/] systems only)"
                    )
                else:
                    self.console.print(
                        f"[bold yellow]{'Warning':>12}[/] No 'activity' column found, extracting all systems"
                    )
            else:
                self.console.print(
                    f"[bold blue]{'Processing':>12}[/] all {original_count} systems (no activity filter)"
                )

            ###### here get the names only of the duplicated systems instead of all the systems that appear 1+ times 
            n_duplicates = df.filter(pl.col("sys_id").is_duplicated()).height
            if n_duplicates > 0:
                duplicate_systems = (
                    df.filter(pl.col("sys_id").is_duplicated())
                    .select("sys_id")
                    .unique()
                    .to_series()
                    .to_list()
                )
                self.console.print(
                    f"[bold yellow]{'Warning':>12}[/] {n_duplicates} duplicate system/s in strain "
                    f"[bold cyan]{self.context.genome_id}[/]: "
                    f"[cyan]{', '.join(duplicate_systems[:5])}{'' if n_duplicates > 5 else ''}[/]"
                )
                df = df.unique(subset=["sys_id"], keep="first")

            self._systems_df = df

        return self._systems_df

    def filter_systems_by_representatives(self, representatives: Iterable[str]):
        """Filter systems to only include representatives"""
        if self._systems_df is None:
            _ = self.systems_df

        representatives_list = list(representatives) if not isinstance(representatives, list) else representatives
        self._systems_df = self._systems_df.filter(
            pl.col("sys_id").is_in(representatives_list)
        )

    @property
    def genome_idx(self) -> Fasta:
        """Load genome FASTA index with pyfaidx (lazy-loaded, cached)"""
        if self._genome_idx is None:
            self._genome_idx = Fasta(str(self.context.genomic_fasta))
        return self._genome_idx

    @property
    def protein_idx(self) -> Fasta:
        """Load protein FASTA index with pyfaidx (lazy-loaded, cached)"""
        if self._protein_idx is None:
            self._protein_idx = Fasta(str(self.context.protein_fasta))
        return self._protein_idx

    @property
    def gff_db(self) -> GFFIndex:
        """Load GFF index (lazy-loaded, cached)"""
        if self._gff_db is None:
            self._gff_db = GFFIndex(self.context.gff_file)
        return self._gff_db

    def cleanup(self):
        """Clean up resources to enable garbage collection."""
        self._systems_df = None
        self._genome_idx = None
        self._protein_idx = None
        self._gff_db = None

    def _get_seq_by_attribute(self, attr_name, attr_value):
        """Get sequence by attribute specified in brackets."""
        for record_id in self._protein_idx.keys():
            header = self._protein_idx[record_id].long_name  
            match = re.search(rf'\[{attr_name}=({attr_value})\]', header)
            if match:
                return self._protein_idx[record_id]
        return None

    def get_protein_sequence(self, hit_id: str) -> Optional[str]:
        """Get protein sequence with fallback substring matching."""
        if self._protein_idx is None:
            self._protein_idx = self.protein_idx

        try:
            return str(self._protein_idx[hit_id])
        except KeyError:
            pass
        
        search_attributes = ['locus_tag', 'ID', 'Name', 'gene']
        for attr in search_attributes:
            seq = self._get_seq_by_attribute(attr, hit_id)
            if seq:
                return str(seq)

        return None


def build_coordinates_dataframe(
    systems_df: pl.DataFrame,
    gff_index: GFFIndex,
    genome_idx: Fasta,
    genomic_fasta_path: Path,
    console: Console,
    verbose: bool = False
) -> List[SystemCoordinates]:
    """
    Parse all systems once, query GFF once, return coordinates.    
    """
    coordinates_list = []
    
    for row in systems_df.iter_rows(named=True):
        sys_id = row['sys_id']
        
        try:
            gene_list = row['protein_in_syst'].split(',')
            gene_list = [g.strip() for g in gene_list if g.strip()]
        except (KeyError, AttributeError):
            console.print(
                f"[bold yellow]{'Warning':>12}[/] No 'protein_in_syst' for system [bold cyan]{sys_id}[/]"
            )
            coordinates_list.append(
                SystemCoordinates(
                    sys_id=sys_id,
                    seq_id="",
                    start_coord=0,
                    end_coord=0,
                    strand="",
                    genes=[],
                    fasta_file=genomic_fasta_path,
                    valid=False,
                    error_msg="Missing protein_in_syst column"
                )
            )
            continue
        
        if not gene_list:
            console.print(
                f"[bold yellow]{'Warning':>12}[/] Empty gene list for system [bold cyan]{sys_id}[/]"
            )
            coordinates_list.append(
                SystemCoordinates(
                    sys_id=sys_id,
                    seq_id="",
                    start_coord=0,
                    end_coord=0,
                    strand="",
                    genes=[],
                    fasta_file=genomic_fasta_path,
                    valid=False,
                    error_msg="Empty gene list"
                )
            )
            continue
        
        gene_features = gff_index.get_batch(gene_list)
        
        found_features = {
            gene_id: feature 
            for gene_id, feature in gene_features.items() 
            if feature is not None
        }
        
        if not found_features:
            console.print(
                f"[bold yellow]{'Warning':>12}[/] No genes found in GFF for system [bold cyan]{sys_id}[/]"
            )
            coordinates_list.append(
                SystemCoordinates(
                    sys_id=sys_id,
                    seq_id="",
                    start_coord=0,
                    end_coord=0,
                    strand="",
                    genes=gene_list,
                    fasta_file=genomic_fasta_path,
                    valid=False,
                    error_msg="No genes found in GFF"
                )
            )
            continue
        
        seq_ids = set(feature['seqid'] for feature in found_features.values())
        if len(seq_ids) > 1:
            console.print(
                f"[bold yellow]{'Warning':>12}[/] System [bold cyan]{sys_id}[/] spans multiple contigs/sequences: {seq_ids}"
            )
            coordinates_list.append(
                SystemCoordinates(
                    sys_id=sys_id,
                    seq_id="",
                    start_coord=0,
                    end_coord=0,
                    strand="",
                    genes=gene_list,
                    fasta_file=genomic_fasta_path,
                    valid=False,
                    error_msg=f"Spans multiple contigs: {seq_ids}"
                )
            )
            continue
        
        seq_id = list(seq_ids)[0]
        
        if seq_id not in genome_idx:
            console.print(
                f"[bold red]{'Error':>12}[/] Contig/sequence {seq_id} not found in genome for system [bold cyan]{sys_id}[/]"
            )
            coordinates_list.append(
                SystemCoordinates(
                    sys_id=sys_id,
                    seq_id=seq_id,
                    start_coord=0,
                    end_coord=0,
                    strand="",
                    genes=gene_list,
                    fasta_file=genomic_fasta_path,
                    valid=False,
                    error_msg=f"Contig {seq_id} not in genome"
                )
            )
            continue
        
        ####### is there a neater way to do this????? 
        start_coord = min(min(feature['start'], feature['end']) for feature in found_features.values())
        end_coord = max(max(feature['start'], feature['end']) for feature in found_features.values())
        
        strand = list(found_features.values())[0]['strand']
        
        region_size = end_coord - start_coord + 1
        if region_size > 1e5:
            console.print(
                f"[bold yellow]{'Warning':>12}[/] System [bold cyan]{sys_id}[/] unusually large: {region_size} bp"
            )
        elif region_size < 50:
            console.print(
                f"[bold yellow]{'Warning':>12}[/] System [bold cyan]{sys_id}[/] unusually small: {region_size} bp"
            )
        
        if verbose:
            console.print(
                f"[bold blue]{'Parsed':>12}[/] {sys_id}: {seq_id}:{start_coord}-{end_coord} "
                f"({len(found_features)}/{len(gene_list)} genes)"
            )
        
        coordinates_list.append(
            SystemCoordinates(
                sys_id=sys_id,
                seq_id=seq_id,
                start_coord=start_coord,
                end_coord=end_coord,
                strand=strand,
                genes=gene_list,
                fasta_file=genomic_fasta_path,
                valid=True,
                error_msg=None
            )
        )
    
    return coordinates_list


def extract_by_contig(
    coordinates: List[SystemCoordinates],
    genome_idx: Fasta,
    output_file: TextIO,
    console: Console,
    verbose: bool = False
) -> List[Tuple[str, int, str]]:
    """
    Group by seq_id, load one contig at a time, extract all systems.
    """
    results = []
    
    valid_coords = [c for c in coordinates if c.valid]
    
    if not valid_coords:
        console.print(f"[bold yellow]{'Warning':>12}[/] No valid systems to extract")
        return results
    
    contig_groups = {}
    for coord in valid_coords:
        if coord.seq_id not in contig_groups:
            contig_groups[coord.seq_id] = []
        contig_groups[coord.seq_id].append(coord)
    
    console.print(
        f"[bold blue]{'Processing':>12}[/] {len(valid_coords)} systems across "
        f"{len(contig_groups)} contigs"
    )
    
    for seq_id, systems in contig_groups.items():
        if verbose:
            console.print(
                f"[bold blue]{'Loading':>12}[/] contig {seq_id} ({len(systems)} systems)"
            )
        
        try:
            contig_seq = genome_idx[seq_id]
            
            for coord in systems:
                try:
                    sequence = contig_seq[coord.start_coord - 1:coord.end_coord]
                    sequence_str = str(sequence)
                    sequence_length = len(sequence_str)
                    
                    output_file.write(f">{coord.sys_id}\n")
                    output_file.write(sequence_str)
                    output_file.write("\n")
                    
                    if verbose:
                        console.print(
                            f"[bold green]{'Extracted':>12}[/] {coord.sys_id} ({sequence_length} bp)"
                        )
                    
                    results.append((
                        coord.sys_id,
                        sequence_length,
                        str(coord.fasta_file)
                    ))
                    
                except Exception as e:
                    console.print(
                        f"[bold red]{'Error':>12}[/] Failed to extract {coord.sys_id}: {e}"
                    )
            
            del contig_seq
            
        except KeyError:
            console.print(
                f"[bold red]{'Error':>12}[/] Contig/sequence {seq_id} not found in genome"
            )
            continue
        except Exception as e:
            console.print(
                f"[bold red]{'Error':>12}[/] Failed to load contig/sequence {seq_id}: {e}"
            )
            continue
    
    return results


def extract_proteins_batch(
    coordinates: List[SystemCoordinates],
    protein_idx: Fasta,
    output_file: TextIO,
    console: Console,
    verbose: bool = False
) -> Dict[str, int]:
    """
    Extract protein sequences for all systems in batch.
    """
    protein_sizes = {}
    
    valid_coords = [c for c in coordinates if c.valid]
    
    if not valid_coords:
        console.print(f"[bold yellow]{'Warning':>12}[/] No valid systems for protein extraction")
        return protein_sizes
    
    total_genes = sum(len(coord.genes) for coord in valid_coords)
    console.print(
        f"[bold blue]{'Processing':>12}[/] {total_genes} proteins from "
        f"{len(valid_coords)} systems"
    )
    
    for coord in valid_coords:
        for gene_id in coord.genes:
            try:
                sequence_str = str(protein_idx[gene_id])
            except KeyError:
                sequence_str = None
                for attr in ['locus_tag', 'ID', 'Name', 'gene']:
                    for record_id in protein_idx.keys():
                        header = protein_idx[record_id].long_name
                        match = re.search(rf'\[{attr}=({gene_id})\]', header)
                        if match:
                            sequence_str = str(protein_idx[record_id])
                            break
                    if sequence_str:
                        break
                
                if sequence_str is None:
                    console.print(
                        f"[bold yellow]{'Warning':>12}[/] Protein {gene_id} not found "
                        f"for system [bold cyan]{coord.sys_id}[/]"
                    )
                    continue
            
            # composite protein ID: system_id__gene_id
            protein_id = f"{coord.sys_id}__{gene_id}"
            output_file.write(f">{protein_id}\n")
            output_file.write(sequence_str)
            output_file.write("\n")
            
            protein_sizes[protein_id] = len(sequence_str)
    
    return protein_sizes


class GeneClusterExtractor:
    """High-level orchestrator for gene cluster extraction pipeline."""
    
    def __init__(
        self,
        progress: Optional[rich.progress.Progress] = None,
        verbose: bool = False,
    ):
        self.progress = progress
        self.console = progress.console if progress else Console()
        self.verbose = verbose
    
    def extract_systems(
        self,
        context: GenomeContext,
        output_file: TextIO,
        representatives: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, int, str]]:
        """Extract genomic sequences for gene clusters."""
        if not context.is_valid():
            self.console.print(f"[bold red]Error:[/] Missing files for {context.genome_id}:")
            for missing in context.missing_files:
                self.console.print(f"  - {missing}")
            return []
        
        resources = GenomeResources(context, self.console)
        
        results = []
        
        try:
            systems_df = resources.systems_df
            
            if representatives:
                resources.filter_systems_by_representatives(representatives)
                systems_df = resources.systems_df
            
            self.console.print(
                f"[bold blue]{'Building':>12}[/] system coordinates"
            )
            coordinates = build_coordinates_dataframe(
                systems_df=systems_df,
                gff_index=resources.gff_db,
                genome_idx=resources.genome_idx,
                genomic_fasta_path=context.genomic_fasta,
                console=self.console,
                verbose=self.verbose
            )
            
            valid_count = sum(1 for c in coordinates if c.valid)
            invalid_count = len(coordinates) - valid_count
            
            self.console.print(
                f"[bold green]{'Validated':>12}[/] {valid_count} systems "
                f"({invalid_count} invalid)"
            )
            
            self.console.print(
                f"[bold blue]{'Extracting':>12}[/] cluster sequences"
            )
            results = extract_by_contig(
                coordinates=coordinates,
                genome_idx=resources.genome_idx,
                output_file=output_file,
                console=self.console,
                verbose=self.verbose
            )
            
            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(results)} gene clusters "
                f"for [bold cyan]{context.genome_id}[/]"
            )
            
        except Exception as e:
            self.console.print(
                f"[bold red]{'Error':>12}[/] Fatal error during extraction for "
                f"{context.genome_id}: {e}"
            )
            import traceback
            traceback.print_exc()
            
        finally:
            resources.cleanup()
            gc.collect()
        
        return results
    
    def extract_proteins(
        self,
        context: GenomeContext,
        output_file: TextIO,
        representatives: Optional[Iterable[str]] = None
    ) -> Dict[str, int]:
        """Extract protein sequences for gene clusters."""
        if not context.is_valid():
            self.console.print(
                f"[bold red]Error:[/] Missing files for {context.genome_id}:"
            )
            for missing in context.missing_files:
                self.console.print(f"  - {missing}")
            return {}
        
        resources = GenomeResources(context, self.console)
        
        all_proteins = {}
        
        try:
            systems_df = resources.systems_df
            
            if representatives:
                resources.filter_systems_by_representatives(representatives)
                systems_df = resources.systems_df
            
            self.console.print(
                f"[bold blue]{'Parsing':>12}[/] system gene lists"
            )
            coordinates = build_coordinates_dataframe(
                systems_df=systems_df,
                gff_index=resources.gff_db,
                genome_idx=resources.genome_idx,
                genomic_fasta_path=context.genomic_fasta,
                console=self.console,
                verbose=self.verbose
            )
            
            self.console.print(
                f"[bold blue]{'Extracting':>12}[/] protein sequences"
            )
            all_proteins = extract_proteins_batch(
                coordinates=coordinates,
                protein_idx=resources.protein_idx,
                output_file=output_file,
                console=self.console,
                verbose=self.verbose
            )
            
            total_proteins = len(all_proteins)
            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {total_proteins} proteins from "
                f"{len(systems_df)} systems for [bold cyan]{context.genome_id}[/]"
            )
            
        except Exception as e:
            self.console.print(
                f"[bold red]{'Error':>12}[/] Fatal error during extraction for "
                f"{context.genome_id}: {e}"
            )
            import traceback
            traceback.print_exc()
            
        finally:
            resources.cleanup()
            gc.collect()
        
        return all_proteins