import gc
import re
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple
import pandas as pd
import rich.progress
from pyfaidx import Fasta
from rich.console import Console



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
                
                # parse attributes
                attr_dict = dict(item.split('=', 1) for item in attrs.split(';') if '=' in item)
                
                feature = {
                    'seqid': seqid,
                    'type': ftype,
                    'start': int(start),
                    'end': int(end),
                    'strand': strand,
                    'attributes': attr_dict
                }
                
                # index by multiple possible keys
                for key in ['ID', 'locus_tag', 'Name', 'gene', 'old_locus_tag', 'protein_id']:
                    val = attr_dict.get(key)
                    if val:
                        self._index[val] = feature
                        # add common variations
                        self._index[f"gene-{val}"] = feature
                        self._index[f"cds-{val}"] = feature
                        self._index[val.replace('_', '~')] = feature
                        self._index[val.replace('~', '_')] = feature
    
    def get(self, gene_id: str) -> Optional[Dict]:
        """O(1) lookup."""
        return self._index.get(gene_id)
    
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
        strain_id: Optional[str],
        systems_tsv: Path,
        genes_tsv: Path,
        gff_file: Path,
        genomic_fasta: Path,
        protein_fasta: Path,
        activity_filter: str = "defense",
    ):
        self.strain_id = strain_id if strain_id else str(uuid.uuid4())[:8]
        
        # file paths 
        self.systems_tsv = Path(systems_tsv)
        self.genes_tsv = Path(genes_tsv)
        self.gff_file = Path(gff_file)
        self.genomic_fasta = Path(genomic_fasta)
        self.protein_fasta = Path(protein_fasta)
        
        # defense system filtering by activity 
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
        return f"GenomeContext(strain_id='{self.strain_id}', files={5 - len(self.missing_files)}/5)"

    def is_valid(self) -> bool:
        """Check if all required files exist"""
        return len(self.missing_files) == 0


class GenomeResources:
    """Manages lazy-loading and caching of genome resources.
    Resources loaded on-demand and cached for reuse.
    Automatic cleanup of temporary files.
    """

    def __init__(
        self, 
        context: GenomeContext, 
        console: Console,
    ):
        self.context = context
        self.console = console

        self._systems_df: Optional[pd.DataFrame] = None
        self._genes_df: Optional[pd.DataFrame] = None
        self._genome_idx: Optional[Fasta] = None
        self._protein_idx: Optional[Fasta] = None
        self._gff_db: Optional[GFFIndex] = None


    @property
    def systems_df(self) -> pd.DataFrame:
        """Load and filter systems TSV (lazy-loaded, cached)"""
        if self._systems_df is None:
            df = pd.read_csv(self.context.systems_tsv, sep="\t")
            original_count = len(df)

            # filter systems by activity if applicable
            if self.context.activity_filter.lower() != "all":
                if "activity" in df.columns:
                    df = df[df["activity"].str.lower() == self.context.activity_filter.lower()]
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

            # check for duplicate systems after activity filtering
            duplicate_mask = df.duplicated(subset=["sys_id"], keep="first")
            if duplicate_mask.any():
                duplicate_systems = df[duplicate_mask]["sys_id"].tolist()
                n_duplicates = len(duplicate_systems)
                # log warning about duplicates
                self.console.print(
                    f"[bold yellow]{'Warning':>12}[/] {n_duplicates} duplicate system/s in strain "
                    f"[bold cyan]{self.context.strain_id}[/]: "
                    f"[cyan]{', '.join(duplicate_systems[:5])}{'...' if n_duplicates > 5 else ''}[/]"
                )
                # keep first occurrence
                df = df.drop_duplicates(subset=["sys_id"], keep="first")

            self._systems_df = df

        return self._systems_df

    def filter_systems_by_representatives(self, representatives: Iterable[str]):
        """Filter systems to only include representatives"""
        if self._systems_df is None:
            _ = self.systems_df

        try:
            # try iterable first
            self._systems_df = self._systems_df[
                self._systems_df['sys_id'].isin(representatives)
            ]
        except TypeError:
            # __contains__ method for non-iterable containers
            mask = self._systems_df['sys_id'].apply(lambda x: x in representatives)
            self._systems_df = self._systems_df[mask]


    @property
    def genes_df(self) -> pd.DataFrame:
        """Load genes TSV (lazy-loaded, cached)"""
        if self._genes_df is None:
            self._genes_df = pd.read_csv(self.context.genes_tsv, sep="\t")
        return self._genes_df

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
    
    def find_gene_feature(self, gene_id: str) -> Optional[Dict]:
        """O(1) lookup with built-in fallbacks."""
        return self.gff_db.get(gene_id)


    def cleanup(self):
        """Clean up resources to enable garbage collection."""
        self._systems_df = None
        self._genes_df = None
        self._genome_idx = None
        self._protein_idx = None
        self._gff_db = None



    def _get_seq_by_attribute(self, attr_name, attr_value):
        """Get sequence by attribute specified in brackets."""
        for record_id in self._protein_idx.keys():
            # get full header line 
            header = self._protein_idx[record_id].long_name  
            match = re.search(rf'\[{attr_name}=({attr_value})\]', header)
            if match:
                return self._protein_idx[record_id]
        return None



    def get_protein_sequence(self, hit_id: str) -> Optional[str]:
        """Get protein sequence with fallback substring matching.
        """
        if self._protein_idx is None:
            self._protein_idx = self.protein_idx

        # try direct lookup first
        try:
            return str(self._protein_idx[hit_id])
        except KeyError:
            pass
        
        # substring search through attributes in header
        search_attributes= ['locus_tag', 'ID', 'Name', 'gene']
        for attr in search_attributes:
            seq = self._get_seq_by_attribute(attr, hit_id)
            if seq:
                return str(seq)

        return None

class DefenseSystem:
    """Represents a single defense system with extraction logic."""
    
    def __init__(
        self,
        system_row: pd.Series,
        resources: GenomeResources,
        console: Console,
        verbose: bool = False,
    ):
        # system data 
        self.sys_id: str = system_row['sys_id']
        self.sys_beg_hit_id: str = system_row['sys_beg']
        self.sys_end_hit_id: str = system_row['sys_end']

        self._system_row = system_row
        self._sys_genes_df: Optional[pd.DataFrame] = None
        self._sys_genes_hit_ids: Optional[List[str]] = None

        self.resources = resources
        self.console = console
        self.verbose = verbose

        # genomic coordinates 
        self.start_coord: Optional[int] = None
        self.end_coord: Optional[int] = None
        self.seq_id: Optional[str] = None
    
    @property
    def sys_genes_df(self) -> pd.DataFrame:
        """Lazy load system genes from resources.genes_df"""
        if self._sys_genes_df is None:
            self._sys_genes_df = self.resources.genes_df[
                self.resources.genes_df['sys_id'] == self.sys_id
            ].sort_values('hit_pos')
        return self._sys_genes_df
    
    @property
    def sys_genes_hit_ids(self) -> List[str]:
        """Lazy load gene hit IDs from sys_genes_df"""
        if self._sys_genes_hit_ids is None:
            self._sys_genes_hit_ids = self.sys_genes_df['hit_id'].tolist()
            # sys genes also specified in the system row as comma-separated string
            # sanity check if systems tsv and genes tsv are consistent
            try: 
                protein_hit_ids = self._system_row['protein_in_syst'].split(",")
                genes_count = self._system_row['genes_count']
                assert len(protein_hit_ids) == genes_count, \
                    f"Protein count mismatch for [bold cyan]{self.sys_id}[/]"
            except (KeyError, AssertionError) as e:
                self._log_warning(
                    f"failed to retrieve protein hit IDs or gene count from systems_tsv for [bold cyan]{self.sys_id}[/]: {e}"
                )
                pass
        return self._sys_genes_hit_ids

    def find_boundaries(self) -> bool:
        """Find and validate genomic boundaries using flexible lookup strategies."""
        start_feature = self.resources.find_gene_feature(self.sys_beg_hit_id)
        if not start_feature:
            self._log_warning(
                f"Start gene '{self.sys_beg_hit_id}' not found in GFF for system [bold cyan]{self.sys_id}[/]"
            )
            return False
        
        end_feature = self.resources.find_gene_feature(self.sys_end_hit_id)
        if not end_feature:
            self._log_warning(
                f"End gene '{self.sys_end_hit_id}' not found in GFF for system [bold cyan]{self.sys_id}[/]"
            )
            return False
        
        # sequences must be on the same contig/chromosome
        if start_feature['seqid'] != end_feature['seqid']:
            self._log_warning(
                f"System [bold cyan]{self.sys_id}[/] spans multiple sequences"
            )
            return False
        
        # sequence must exist in genome index
        if start_feature['seqid'] not in self.resources.genome_idx:
            self._log_error(
                f"Sequence {start_feature['seqid']} not found in genome"
            )
            return False
        
        self.start_coord = start_feature['start']
        self.end_coord = end_feature['end']
        self.seq_id = start_feature['seqid']
        
        # region size warnings 
        region_size = self.end_coord - self.start_coord + 1
        if region_size > 1e4:
            self._log_warning(f"System [bold cyan]{self.sys_id}[/] region unusually large: {region_size} bp")
        if region_size < 1e2:
            self._log_warning(f"System [bold cyan]{self.sys_id}[/] region unusually small: {region_size} bp")

        return True
        
    def extract_genomic_sequence(
        self,
        output_file: TextIO
    ) -> Optional[Tuple[str, int, str]]:
        """Extract genomic sequence for this defense system.
        
        Args:
            output_file: Open file handle to write FASTA sequence
            
        Returns:
            Tuple of (sys_id, sequence_length, source_file) or None on error
        """
        if not all([self.start_coord, self.end_coord, self.seq_id]):
            self._log_error(f"Cannot extract sequence: boundaries not set for [bold cyan]{self.sys_id}[/]")
            return None
        
        try:
            genome_seq = self.resources.genome_idx[self.seq_id]
            sequence = genome_seq[self.start_coord - 1:self.end_coord]  # 0-based indexing
            sequence_str = str(sequence)
            sequence_length = len(sequence_str)
            
            self._write_fasta(output_file, self.sys_id, sequence_str)
            
            if self.verbose:
                self.console.print(
                    f"[bold blue]{'Extracted':>22}[/] genomic sequence for "
                    f"[bold cyan]{self.sys_id}[/] ({sequence_length} bp)"
                )
            
            return (
                self.sys_id,                                # cluster_id
                sequence_length,                            # cluster_length
                str(self.resources.context.genomic_fasta)   # filename
            )
            
        except Exception as e:
            self._log_error(f"Failed to extract sequence for [bold cyan]{self.sys_id}[/]: {e}")
            return None
    

    def extract_protein_sequences(
        self,
        output_file: TextIO
    ) -> Dict[str, int]:
        """Extract protein sequences for all genes in this system.
        
        Args:
            output_file: Open file handle to write FASTA sequences
            
        Returns:
            Dict mapping protein_id to sequence_length
        """
        protein_sizes = {}
        
        for hit_id in self.sys_genes_hit_ids:
            sequence_str = self.resources.get_protein_sequence(hit_id)
            
            if sequence_str is None:
                self._log_warning(
                    f"Protein {hit_id} not found in FASTA for system [bold cyan]{self.sys_id}[/]"
                )
                continue
            # composite protein ID: system_id__gene_id
            # because using "_" as in original IGUA would break downstream processing,
            # since system IDs and gene IDs are separated by "_"
            protein_id = f"{self.sys_id}__{hit_id}"
            self._write_fasta(output_file, protein_id, sequence_str)
            protein_sizes[protein_id] = len(sequence_str)
        
        return protein_sizes


    def _write_fasta(self, file: TextIO, name: str, sequence: str):
        """Write a FASTA record to file"""
        file.write(">{}\n".format(name))
        file.write(sequence)
        file.write("\n")
        return None
    
    def _log_warning(self, message: str):
        """Log a warning message"""
        self.console.print(f"[bold yellow]{'Warning':>12}[/] {message}")
    
    def _log_error(self, message: str):
        """Log an error message"""
        self.console.print(f"[bold red]{'Error':>12}[/] {message}")


class DefenseSystemExtractor:
    """High-level orchestrator for defense system extraction pipeline."""
    
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
        """Extract genomic sequences for defense systems."""
        if not context.is_valid():
            self.console.print(f"[bold red]Error:[/] Missing files for {context.strain_id}:")
            for missing in context.missing_files:
                self.console.print(f"  - {missing}")
            return []
        
        # initialize resources
        resources = GenomeResources(context, self.console)
        using_external_progress = self.progress is not None
        
        if not using_external_progress:
            self.progress = rich.progress.Progress(
                rich.progress.SpinnerColumn(),
                rich.progress.TextColumn("[bold blue]{task.description}"),
                rich.progress.BarColumn(),
                rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
            )
            self.progress.start()
        
        results = []
        
        try:
            systems_df = resources.systems_df
            
            if representatives:
                resources.filter_systems_by_representatives(representatives)
                systems_df = resources.systems_df
            
            systems_processed = 0
            for _, system_row in systems_df.iterrows():
                system = DefenseSystem(
                    system_row,
                    resources,
                    self.console,
                    self.verbose,
                )
                
                if not system.find_boundaries():
                    continue
                
                result = system.extract_genomic_sequence(output_file)
                if result:
                    results.append(result)
                
                systems_processed += 1
                
                if systems_processed % 10 == 0:
                    gc.collect()
            
            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(results)} defense systems "
                f"for [bold cyan]{context.strain_id}[/]"
            )
            
        except Exception as e:
            self._log_error(
                f"Fatal error during extraction for for {context.strain_id}: {e}",
            )
            
        finally:
            resources.cleanup()
            
            if not using_external_progress and self.progress:
                self.progress.stop()
            
            gc.collect()
        
        return results
    

    def extract_proteins(
        self,
        context: GenomeContext,
        output_file: TextIO,
        representatives: Optional[Iterable[str]] = None
    ) -> Dict[str, int]:
        """Extract protein sequences for defense systems.
        
        Args:
            context: Genome context with file paths and metadata
            output_file: Open file handle for writing FASTA sequences
            representatives: Optional set of system IDs to extract (for filtering)
            
        Returns:
            Dict mapping protein_id to sequence_length
        """
        if not context.is_valid():
            self.console.print(
                f"[bold red]Error:[/] Missing files for {context.strain_id}:"
            )
            for missing in context.missing_files:
                self.console.print(f"  - {missing}")
            return {}
        
        # initialize resources 
        resources = GenomeResources(context, self.console)
        
        using_external_progress = self.progress is not None
        if not using_external_progress:
            self.progress = rich.progress.Progress(
                rich.progress.SpinnerColumn(),
                rich.progress.TextColumn("[bold blue]{task.description}"),
                rich.progress.BarColumn(),
                rich.progress.TimeElapsedColumn(),
                rich.progress.TimeRemainingColumn(),
                rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
            )
            self.progress.start()
        
        all_proteins = {}
        
        try:
            systems_df = resources.systems_df
            
            if representatives:
                resources.filter_systems_by_representatives(representatives)
                systems_df = resources.systems_df
            
            systems_processed = 0
            for _, system_row in systems_df.iterrows():
                system = DefenseSystem(
                    system_row,
                    resources,
                    self.console,
                    self.verbose
                )
                
                proteins = system.extract_protein_sequences(output_file)
                all_proteins.update(proteins)
                
                systems_processed += 1
            
            total_proteins = len(all_proteins)
            self.console.print(
                f"[bold green]{'Extracted':>12}[/] {total_proteins} proteins from "
                f"{len(systems_df)} systems for [bold cyan]{context.strain_id}[/]"
            )
            
        except Exception as e:
            self._log_error(
                f"Fatal error during extraction for for {context.strain_id}: {e}",
            )
            
        finally:
            resources.cleanup()
            
            if not using_external_progress and self.progress:
                self.progress.stop()
            
            gc.collect()
        
        return all_proteins
    

    def _log_error(self, message: str):
        """Log an error message"""
        self.console.print(f"[bold red]{'Error':>12}[/] {message}")