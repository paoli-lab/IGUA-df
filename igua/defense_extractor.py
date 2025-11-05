from abc import ABC, abstractmethod
import gc
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO, Tuple
import gffutils
import pandas as pd
import rich.progress
from pyfaidx import Fasta
from rich.console import Console


# [ ] cleanup and logging 
# [ ] still problems with GFF and fasta lookup when substring

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
        gff_cache_dir: Optional[Path] = None
    ):
        self.context = context
        self.console = console
        self.gff_cache_dir = gff_cache_dir

        self._systems_df: Optional[pd.DataFrame] = None
        self._genes_df: Optional[pd.DataFrame] = None
        self._genome_idx: Optional[Fasta] = None
        self._protein_idx: Optional[Fasta] = None
        self._gff_db: Optional[gffutils.FeatureDB] = None
        self._gff_db_path: Optional[str] = None
        self._gene_lookup_cache: Optional[Dict[str, str]] = None

        # # track what needs cleanup
        # self._temp_files = []


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

    # @property
    # def gff_db(self) -> gffutils.FeatureDB:
    #     """Create GFF database (lazy-loaded, cached)"""
    #     if self._gff_db is None:
    #         unique_id = str(uuid.uuid4())[:8]

    #         if self.gff_cache_dir:
    #             os.makedirs(self.gff_cache_dir, exist_ok=True)
    #             db_path = os.path.join(
    #                 str(self.gff_cache_dir),
    #                 f"{os.path.basename(str(self.context.gff_file))}_{unique_id}.db"
    #             )
    #         else:
    #             db_path = os.path.join(
    #                 tempfile.gettempdir(),
    #                 f"gff_temp_{unique_id}.db"
    #             )

    #         try:
    #             # disk-based database first
    #             self._gff_db = gffutils.create_db(
    #                 str(self.context.gff_file),
    #                 dbfn=db_path,
    #                 force=True,
    #                 merge_strategy='create_unique',
    #                 id_spec=['ID', 'Name', 'gbkey', 'gene', 'gene_biotype', 'locus_tag', 'old_locus_tag']
    #             )
    #             self._gff_db_path = db_path
    #             # self._temp_files.append(db_path)

    #         except Exception as e:
    #             self.console.print(
    #                 f"[bold red]Error:[/] Failed to create GFF database: {str(e)}"
    #             )
    #             self.console.print("[yellow]Using in-memory database[/]")

    #             # fallback to in-memory
    #             self._gff_db = gffutils.create_db(
    #                 str(self.context.gff_file),
    #                 dbfn=":memory:",
    #                 force=True,
    #                 merge_strategy='create_unique',
    #                 id_spec=['ID', 'Name', 'gbkey', 'gene', 'gene_biotype', 'locus_tag', 'old_locus_tag']
    #             )
    #             self._gff_db_path = ":memory:"

    #     return self._gff_db

    @property
    def gff_db(self) -> gffutils.FeatureDB:
        """Create GFF database (lazy-loaded, cached)"""
        if self._gff_db is None:
            unique_id = str(uuid.uuid4())[:8]
            
            if self.gff_cache_dir:
                os.makedirs(self.gff_cache_dir, exist_ok=True)
                db_path = os.path.join(
                    str(self.gff_cache_dir),
                    f"{self.context.gff_file.stem}_{unique_id}.db"
                )
            else:
                db_path = os.path.join(
                    tempfile.gettempdir(),
                    f"gff_temp_{unique_id}.db"
                )
            
            try:
                self._gff_db = gffutils.create_db(
                    str(self.context.gff_file),
                    dbfn=db_path,
                    force=True,
                    merge_strategy='create_unique',
                    id_spec={
                        'gene': ['ID', 'gene_id', 'locus_tag', 'Name'],
                        'CDS': ['ID', 'protein_id', 'locus_tag', 'Name'],
                        'mRNA': ['ID', 'transcript_id', 'locus_tag', 'Name']
                    },
                    disable_infer_genes=True,
                    disable_infer_transcripts=True,
                )
                self._gff_db_path = db_path
                
            except Exception as e:
                self.console.print(f"[bold red]Error:[/] Failed to create GFF database: {e}")
                self.console.print("[yellow]Using in-memory database[/]")
                
                self._gff_db = gffutils.create_db(
                    str(self.context.gff_file),
                    dbfn=":memory:",
                    force=True,
                    merge_strategy='create_unique',
                    id_spec={
                        'gene': ['ID', 'gene_id', 'locus_tag', 'Name'],
                        'CDS': ['ID', 'protein_id', 'locus_tag', 'Name'],
                        'mRNA': ['ID', 'transcript_id', 'locus_tag', 'Name']
                    },
                    disable_infer_genes=True,
                    disable_infer_transcripts=True,
                )
                self._gff_db_path = ":memory:"
        
        if self._gene_lookup_cache is None:
            self._build_gene_lookup_cache()
        
        return self._gff_db


    def _build_gene_lookup_cache(self):
        """Build a mapping from various gene ID formats to feature IDs"""
        if self._gene_lookup_cache is not None:
            return
        
        self._gene_lookup_cache = {}
        db = self.gff_db
        
        # Iterate through all features once
        for feature in db.all_features(featuretype=['gene', 'CDS', 'mRNA']):
            # Add direct ID
            self._gene_lookup_cache[feature.id] = feature.id
            
            # Add formatted variants
            for attr in ['ID', 'Name', 'locus_tag', 'gene', 'old_locus_tag']:
                if attr in feature.attributes:
                    for value in feature.attributes[attr]:
                        self._gene_lookup_cache[value] = feature.id
                        # MAG format variants
                        self._gene_lookup_cache[f"gene-{value}"] = feature.id
                        self._gene_lookup_cache[value.replace('~', '_')] = feature.id
    
    def cleanup(self):
        """Clean up temporary files and resources"""
        # # clean up temporary GFF database files
        # for temp_file in self._temp_files:
        #     if os.path.exists(temp_file):
        #         try:
        #             os.unlink(temp_file)
        #         except Exception as e:
        #             self.console.print(
        #                 f"[yellow]Warning: Could not delete temp file {temp_file}: {e}[/]"
                    # )

        self._systems_df = None
        self._genes_df = None
        self._genome_idx = None
        self._protein_idx = None
        self._gff_db = None


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
        # self.protein_hit_ids: list = system_row['protein_in_syst'].split(",")
        # self.genes_count: int = system_row['genes_count']

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
        
        # assert len(self.protein_hit_ids) == self.genes_count, \
        #     f"Protein count mismatch for {self.sys_id}"
    
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
        return self._sys_genes_hit_ids
    
    # @property
    # def protein_hit_ids(self) -> List[str]:
    #     """Get protein hit IDs from system row"""
    #     return self._system_row['protein_in_syst'].split(",")
    
    @property
    def genes_count(self) -> int:
        """Get gene count from system row"""
        return self._system_row['genes_count']

    def find_boundaries(self) -> bool:
        """Find and validate genomic boundaries using flexible lookup strategies.
        
        Returns:
            True if boundaries found and valid, False otherwise
        """
        db = self.resources.gff_db
        if not hasattr(self.resources, '_gene_lookup_cache'):
            self.resources._build_gene_lookup_cache()
        
        cache = self.resources._gene_lookup_cache
        
        # direct lookup from cache
        start_feature_id = cache.get(self.sys_beg_hit_id)
        if not start_feature_id:
            self._log_warning(
                f"Start gene '{self.sys_beg_hit_id}' not found in GFF for system {self.sys_id}"
            )
            return False
        
        end_feature_id = cache.get(self.sys_end_hit_id)
        if not end_feature_id:
            self._log_warning(
                f"End gene '{self.sys_end_hit_id}' not found in GFF for system {self.sys_id}"
            )
            return False
        
        try:
            start_feature = db[start_feature_id]
            end_feature = db[end_feature_id]
        except KeyError as e:
            self._log_error(f"Feature lookup failed: {e}")
            return False
        
        if start_feature.seqid != end_feature.seqid:
            self._log_warning(
                f"System {self.sys_id} spans multiple sequences: "
                f"{start_feature.seqid} and {end_feature.seqid}"
            )
            return False
        
        if start_feature.seqid not in self.resources.genome_idx:
            self._log_error(
                f"Sequence {start_feature.seqid} not found in genome for system {self.sys_id}"
            )
            return False
        
        # Set coordinates
        self.start_coord = start_feature.start
        self.end_coord = end_feature.end
        self.seq_id = start_feature.seqid
        
        # Region size warnings
        region_size = self.end_coord - self.start_coord + 1
        if region_size > 1e4:
            self._log_warning(f"System {self.sys_id} region unusually large: {region_size} bp")
        if region_size < 1e2:
            self._log_warning(f"System {self.sys_id} region unusually small: {region_size} bp")
        
        return True
        
        # start_gene_start_coord, start_gene_end_coord, start_gene_seq_id = start_gene_result
        # end_gene_start_coord, end_gene_end_coord, end_gene_seq_id = end_gene_result

        # # genes must be on same sequence
        # if start_gene_seq_id != end_gene_seq_id:
        #     self._log_warning(
        #         f"System {self.sys_id} spans multiple sequences: "
        #         f"{start_gene_seq_id} and {end_gene_seq_id}"
        #     )
        #     return False

        # # sequence must exist in genome
        # if start_gene_seq_id not in self.resources.genome_idx:
        #     self._log_error(
        #         f"Sequence {start_gene_seq_id} not found in genome for system {self.sys_id}"
        #     )
        #     return False
        
        # self.start_coord = start_gene_start_coord
        # self.end_coord = end_gene_end_coord  
        # self.seq_id = start_gene_seq_id

        # # region size warnings 
        # region_size = self.end_coord - self.start_coord + 1
        
        # if region_size > 1e4:
        #     self._log_warning(
        #         f"System {self.sys_id} region unusually large: {region_size} bp"
        #     )
        # if region_size < 1e2:
        #     self._log_warning(
        #         f"System {self.sys_id} region unusually small: {region_size} bp"
        #     )
        
        # return True
    

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
            self._log_error(f"Cannot extract sequence: boundaries not set for {self.sys_id}")
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
                    f"[cyan]{self.sys_id}[/] ({sequence_length} bp)"
                )
            
            return (
                self.sys_id,  # cluster_id
                sequence_length,  # cluster_length
                str(self.resources.context.genomic_fasta)  # filename
            )
            
        except Exception as e:
            self._log_error(f"Failed to extract sequence for {self.sys_id}: {e}")
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
            try:
                sequence = self.resources.protein_idx[hit_id]
                sequence_str = str(sequence)
                sequence_length = len(sequence_str)
                
                # composite protein ID: system_id__gene_id
                # because using "_" as in original IGUA would break downstream processing,
                # since system IDs and gene IDs are separated by "_"
                protein_id = f"{self.sys_id}__{hit_id}"
                
                self._write_fasta(output_file, protein_id, sequence_str)
                
                protein_sizes[protein_id] = sequence_length
                
            except KeyError:
                self._log_warning(
                    f"Protein {hit_id} not found in FASTA for system {self.sys_id}"
                )
            except Exception as e:
                self._log_error(
                    f"Error extracting protein {hit_id} for system {self.sys_id}: {e}"
                )
        
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
        gff_cache_dir: Optional[Path] = None,
        representatives: Optional[Iterable[str]] = None
    ) -> List[Tuple[str, int, str]]:
        """Extract genomic sequences for defense systems."""
        if not context.is_valid():
            self.console.print(f"[bold red]Error:[/] Missing files for {context.strain_id}:")
            for missing in context.missing_files:
                self.console.print(f"  - {missing}")
            return []
        
        # initialize resources 
        resources = GenomeResources(context, self.console, gff_cache_dir)
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
                "EXTRACTION_FATAL_ERROR",
                f"Uncaught exception in extraction process: {e}",
                strain_id=context.strain_id
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
        resources = GenomeResources(context, self.console, gff_cache_dir=None)
        
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
                "PROTEIN_EXTRACTION_ERROR",
                f"Failed to extract protein sequences: {e}",
                strain_id=context.strain_id
            )
            
        finally:
            resources.cleanup()
            
            if not using_external_progress and self.progress:
                self.progress.stop()
            
            gc.collect()
        
        return all_proteins
    

    def _log_error(
        self,
        error_type: str,
        message: str,
        strain_id: Optional[str] = None,
        system_id: Optional[str] = None,
        files_dict: Optional[Dict] = None,
        exception: Optional[Exception] = None
    ):
        """Log error information to file"""
        pass
        # log_file = "defense_extraction_errors.log"
        
        # timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        # log_message = f"[{timestamp}] {error_type}: {message}\n"
        
        # if strain_id:
        #     log_message += f"  Strain: {strain_id}\n"
        # if system_id:
        #     log_message += f"  System: {system_id}\n"
        # if files_dict:
        #     log_message += "  Files:\n"
        #     for key, path in files_dict.items():
        #         if path:
        #             log_message += f"    {key}: {path}\n"
        # if exception:
        #     log_message += f"  Exception: {str(exception)}\n"
        
        # log_message += "-" * 80 + "\n"
        
        # try:
        #     with open(log_file, "a") as f:
        #         f.write(log_message)
        # except Exception as e:
        #     self.console.print(f"[red]Failed to write to log file: {e}[/]")
