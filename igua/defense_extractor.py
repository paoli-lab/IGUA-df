import os
import pathlib
import pandas as pd
import re
import gc
import time
import tempfile
import uuid
import gffutils
import io
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional, Union, Any, Container, Iterable
import traceback
import tempfile


import rich.progress
from rich.console import Console


class DefenseExtractor:
    """
    Handles extraction of defense systems from genomic data.
    
    This class provides methods to extract defense system sequences from:
    1. DefenseFinder TSV files (systems.tsv and genes.tsv)
    2. GFF annotation files
    3. Genome FASTA files
    """
    
    def __init__(
        self, 
        progress: Optional[rich.progress.Progress] = None,
        output_base_dir: Optional[pathlib.Path] = None,
        write_output: bool = False,
        verbose: bool = False,
        extract_nucleotides: bool = False,  # nucleotide seqs not extracted by default, useful for debugging
    ):
        self.progress = progress
        self.output_base_dir = output_base_dir
        self.write_output = write_output
        self.console = progress.console if progress else Console()
        self.verbose = verbose
        self.extract_nucleotides = extract_nucleotides

    def _get_unique_strain_id(self, strain_id: Optional[str]) -> str:
        """Get or create a unique strain identifier"""
        if strain_id:
            # clean strain_id to avoid filesystem issues
            clean_strain = strain_id.replace('/', '_').replace('\\', '_').replace(' ', '_')

            return clean_strain
        
        # if None strain_id, generate a unique identifier
        unique_ref = str(uuid.uuid4())[:8]
        return unique_ref

    def _load_and_filter_systems(
        self, 
        systems_tsv_file: pathlib.Path, 
        genes_tsv_file: pathlib.Path,
        activity_filter: str,
        strain_id: Optional[str],
        files_dict: Dict[str, Any]
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
        """Load and filter TSV files"""
        try:
            systems_df = pd.read_csv(systems_tsv_file, sep='\t')
            
            # filter systems by activity if applicable
            original_count = len(systems_df)
            if activity_filter.lower() != "all":
                if 'activity' in systems_df.columns:
                    systems_df = systems_df[systems_df['activity'].str.lower() == activity_filter.lower()]
                    
            # check for duplicate systems after activity filtering
            duplicate_mask = systems_df.duplicated(keep='first')
            if duplicate_mask.any():
                duplicate_systems = systems_df[duplicate_mask]['sys_id'].tolist()
                n_duplicates = len(duplicate_systems)
                
                self.console.print(
                    f"[bold yellow]{'Warning':>12}[/] {n_duplicates} duplicate system/s in strain [bold cyan]{strain_id}[/]: [cyan]{', '.join(duplicate_systems[:5])}{'...' if n_duplicates > 5 else ''}[/]"
                )
                
                self._log_error(
                    "DUPLICATE_SYSTEMS_WARNING", 
                    f"Found {n_duplicates} duplicate system IDs in systems TSV", 
                    strain_id=strain_id, 
                    files=files_dict,
                    exception={
                        "duplicate_sys_ids": duplicate_systems,
                        "systems_tsv_file": str(systems_tsv_file),
                        "total_systems": len(systems_df),
                        "unique_systems": len(systems_df['sys_id'].unique())
                    }
                )
                
                # keep first occurrence
                systems_df = systems_df.drop_duplicates(keep='first')
            
            if activity_filter.lower() != "all":
                if 'activity' in systems_df.columns:
                    self.console.print(
                        f"[bold green]{'Filtered':>12}[/] {original_count} systems to {len(systems_df)} "
                        f"([bold cyan]{activity_filter}[/] systems only)"
                    )
                else:
                    self.console.print(
                        f"[bold yellow]{'Warning':>12}[/] No 'activity' column found, extracting all systems"
                    )
            else:
                self.console.print(f"[bold blue]{'Processing':>12}[/] all {original_count} systems (no activity filter)")

            genes_df = pd.read_csv(genes_tsv_file, sep='\t')
            return systems_df, genes_df
            
        except Exception as e:
            self._log_error(
                "TSV_ERROR", "Failed to read TSV files", 
                strain_id=strain_id, files=files_dict, exception=e
            )
            self.console.print(f"[bold red]Error:[/] Failed to read TSV files: {str(e)}")
            return None, None

    def _setup_gff_database(
        self, 
        gff_file: pathlib.Path, 
        gff_cache_dir: Optional[pathlib.Path],
        unique_id: str,
        strain_id: Optional[str],
        files_dict: Dict[str, Any]
    ) -> Tuple[Any, str]:
        """Set up GFF database with in-memory fallback"""
        db_path = ":memory:"
        
        try:
            if gff_cache_dir:
                os.makedirs(gff_cache_dir, exist_ok=True)
                db_path = os.path.join(str(gff_cache_dir), f"{os.path.basename(str(gff_file))}_{unique_id}.db")
            else:
                db_path = os.path.join(tempfile.gettempdir(), f"gff_temp_{unique_id}.db")
                
            db = gffutils.create_db(
                str(gff_file),  
                dbfn=db_path, 
                force=True, 
                merge_strategy='create_unique',
                    # widen id_spec to include more attributes for robust matching
                id_spec=['ID', 'Name', 'gbkey', 'gene', 'gene_biotype', 'locus_tag', 'old_locus_tag']
            )
            return db, db_path
            
        except Exception as e:
            self._log_error(
                "GFF_ERROR", "Failed to create GFF database, falling back to in-memory", 
                strain_id=strain_id, files=files_dict, exception=e
            )
            self.console.print(f"[bold blue]{'Using':>12}[/] in-memory database")
            db_path = ":memory:"
            
            try:
                db = gffutils.create_db(
                    str(gff_file),
                    dbfn=":memory:", 
                    merge_strategy='create_unique',
                        # widen id_spec to include more attributes for robust matching
                    id_spec=['ID', 'Name', 'gbkey', 'gene', 'gene_biotype', 'locus_tag', 'old_locus_tag']
                )
                return db, db_path
                
            except Exception as e2:
                self._log_error(
                    "GFF_FATAL_ERROR", "Failed to create in-memory GFF database", 
                    strain_id=strain_id, files=files_dict, exception=e2
                )
                self.console.print(f"[bold red]Error:[/] Failed to create GFF database: {str(e2)}")
                return None, db_path

    def _extract_single_system(
        self,
        system: pd.Series,
        genes_df: pd.DataFrame,
        genome_dict: Dict[str, Any],
        db: Any,
        output_dir: Optional[pathlib.Path],
        strain_id: Optional[str],
        files_dict: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Extract a single defense system"""
        sys_id = system['sys_id']
        clean_strain = self._get_unique_strain_id(strain_id)
        unique_sys_id = f"{clean_strain}@{sys_id}"
        system_files = {**files_dict, "system_id": sys_id}
        
        system_genes = genes_df[genes_df['sys_id'] == sys_id].sort_values('hit_pos') 
        if system_genes.empty:
            self._log_error(
                "NO_GENES_ERROR", "No genes found for system", 
                strain_id=strain_id, system_id=sys_id, files=system_files
            )
            self.console.print(f"[bold yellow]{'Warning':>12}[/] No genes found for system {sys_id}")
            return None
        
        # beginning and ending gene
        try:
            beg_gene_mask = system_genes['hit_id'] == system['sys_beg']
            end_gene_mask = system_genes['hit_id'] == system['sys_end']
            
            if not beg_gene_mask.any() or not end_gene_mask.any():
                self._log_error(
                    "GENE_NOT_FOUND", f"Beginning or ending gene not found in system genes", 
                    strain_id=strain_id, system_id=sys_id, files=system_files,
                    exception={
                        "sys_beg": system['sys_beg'], 
                        "sys_end": system['sys_end'],
                        "available_hit_ids": list(system_genes['hit_id'])
                    }
                )
                self.console.print(f"[bold red]Error:[/] Could not find beginning or ending gene for system {sys_id}")
                return None
            
            beg_hit_id = system['sys_beg']
            end_hit_id = system['sys_end']
            
            # get genomic coordinates from GFF database
            start_coord, start_end, start_seq_id = self._find_gene_coordinates(db, beg_hit_id)
            end_coord, end_end, end_seq_id = self._find_gene_coordinates(db, end_hit_id)

            system_start = start_coord
            system_end = end_end
            
        except Exception as e:
            self._log_error(
                "GENE_DATA_ERROR", "Failed to extract gene information", 
                strain_id=strain_id, system_id=sys_id, files=system_files, exception=e
            )
            self.console.print(f"[bold red]Error:[/] Failed to get gene data for system {sys_id}: {str(e)}")
            return None
        
        # validate coordinates
        if system_start is None or system_end is None or start_seq_id is None or end_seq_id is None:
            self._log_error(
                "COORD_ERROR", "Missing coordinates for system", 
                strain_id=strain_id, system_id=sys_id, files=system_files,
                exception={
                    "system_start": system_start, "system_end": system_end, 
                    "start_seq_id": start_seq_id, "end_seq_id": end_seq_id,
                    "beg_hit_id": beg_hit_id, "end_hit_id": end_hit_id
                }
            )
            self.console.print(f"[bold yellow]{'Warning':>12}[/] Missing coordinates for system {sys_id}")
            return None
            
        if start_seq_id != end_seq_id:
            self._log_error(
                "SEQUENCE_SPAN_ERROR", "System spans multiple sequences", 
                strain_id=strain_id, system_id=sys_id, files=system_files,
                exception={"start_seq_id": start_seq_id, "end_seq_id": end_seq_id}
            )
            self.console.print(f"[bold yellow]{'Warning':>12}[/] System {sys_id} spans multiple sequences")
            return None
            
        seq_id = start_seq_id
        
        # swap coordinate order if needed
        if system_start > system_end:
            system_start, system_end = system_end, system_start
            self.console.print(f"[bold yellow]{'Warning':>12}[/] System {sys_id} coordinates swapped: start > end.")

        # raise warning for suspiciously large regions >1e4 
        region_size = system_end - system_start + 1
        if region_size > 1e4:
            self._log_error(
                "LARGE_REGION_WARNING", f"System region too large: {region_size} bp", 
                strain_id=strain_id, system_id=sys_id, files=system_files
            )
            self.console.print(f"[bold yellow]{'Warning':>12}[/] System [cyan]{sys_id}[/] region too large: {region_size} bp.")
        
        # extract genomic sequence, store and write if needed
        if seq_id not in genome_dict:
            self._log_error(
                "SEQUENCE_ID_ERROR", f"Sequence ID {seq_id} not found in genome", 
                strain_id=strain_id, system_id=sys_id, files=system_files
            )
            self.console.print(f"[bold red]{'Error':>12}[/] Sequence {seq_id} not found in genome")
            return None

        try:
            genome_seq = genome_dict[seq_id].seq
            sequence = genome_seq[max(0, system_start-1):min(len(genome_seq), system_end)]
            sequence_length = len(sequence)
            
            result = {
                "sequence": str(sequence),
                "length": sequence_length,
                "seq_id": seq_id,
                "start": system_start,
                "end": system_end,
                "type": system['type'],
                "subtype": system['subtype'] if 'subtype' in system and pd.notna(system['subtype']) else None,
                "strain_id": strain_id,
                "sys_id": sys_id,
                "unique_sys_id": unique_sys_id
            }
            
            if self.write_output and output_dir:
                output_file = os.path.join(str(output_dir), f"{unique_sys_id}.fasta")
                with open(output_file, "w") as out:
                    out.write(f">{unique_sys_id} OriginalID:{sys_id} Strain:{strain_id} Type:{system['type']} Subtype:{system['subtype'] if 'subtype' in system and pd.notna(system['subtype']) else 'NA'} Length:{sequence_length}bp\n")
                    out.write(str(sequence) + "\n")
                result["file_path"] = output_file
                
            if self.verbose: 
                self.console.print(f"[bold blue]{'Extracted':>22}[/] genomic sequence for [cyan]{sys_id}[/] system ({sequence_length} bp)")
            

            del sequence, genome_seq
            return result
            
        except Exception as e:
            self._log_error(
                "SEQUENCE_EXTRACTION_ERROR", "Failed to extract or write sequence", 
                strain_id=strain_id, system_id=sys_id, files=system_files, exception=e
            )
            self.console.print(f"[bold red]Error:[/] Failed to extract sequence for {sys_id}: {str(e)}")
            return None

    def extract_systems(
        self,
        systems_tsv_file: pathlib.Path, 
        genes_tsv_file: pathlib.Path, 
        gff_file: pathlib.Path, 
        fasta_file: pathlib.Path,
        output_dir: Optional[pathlib.Path] = None,
        gff_cache_dir: Optional[pathlib.Path] = None,
        strain_id: Optional[str] = None,
        activity_filter: str = "defense",
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract defense systems' sequences from genomic data
        
        Args:
            systems_tsv_file: Path to DefenseFinder systems TSV
            genes_tsv_file: Path to DefenseFinder genes TSV
            gff_file: Path to GFF annotation file
            fasta_file: Path to genome FASTA file
            output_dir: Directory to write output files (if write_output=True)
            gff_cache_dir: Directory to cache GFF databases
            strain_id: Optional strain identifier
            
        Returns:
            Dictionary mapping system_id to sequence data and metadata
        """
        if self.write_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        unique_id = str(uuid.uuid4())
        
        # Setup file dictionary for error logging
        files_dict = {
            "systems_tsv": systems_tsv_file,
            "genes_tsv": genes_tsv_file,
            "gff_file": gff_file,
            "fasta_file": fasta_file
        }
        
        if output_dir:
            files_dict["output_dir"] = output_dir
        
        using_external_progress = self.progress is not None
        progress = self.progress or rich.progress.Progress(
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
        )
        
        results = {}
        systems_df = None
        genes_df = None
        genome_dict = None
        db = None
        db_path = ":memory:"
        
        try:
            if not using_external_progress:
                progress.start()
            
            # load and filter TSV files
            systems_df, genes_df = self._load_and_filter_systems(
                systems_tsv_file, genes_tsv_file, activity_filter, strain_id, files_dict
            )
            
            if systems_df is None or genes_df is None:
                return {}
            
            # load genome
            try:
                genome_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
            except Exception as e:
                self._log_error(
                    "FASTA_ERROR", "Failed to load genome FASTA", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to load genome: {str(e)}")
                return {}
            
            # setup GFF database
            db, db_path = self._setup_gff_database(
                gff_file, gff_cache_dir, unique_id, strain_id, files_dict
            )
            
            if db is None:
                return {}
            
            # process each system
            systems_processed = 0
            for _, system in systems_df.iterrows():
                result = self._extract_single_system(
                    system, genes_df, genome_dict, db, output_dir, strain_id, files_dict
                )
                
                if result:
                    results[result["unique_sys_id"]] = result
                
                systems_processed += 1
                
                # Periodic garbage collection
                if systems_processed % 10 == 0:
                    gc.collect()

            self.console.print(f"[bold green]{'Extracted':>12}[/] {len(results)} defense systems for [bold cyan]{strain_id}[/]")
            
        except Exception as e:
            self._log_error(
                "EXTRACTION_FATAL_ERROR", "Uncaught exception in extraction process", 
                strain_id=strain_id, files=files_dict, exception=e
            )
            self.console.print(f"[bold red]Fatal Error:[/] {str(e)}")
            
        finally:
            # cleanup
            if systems_df is not None:
                del systems_df
            if genes_df is not None:
                del genes_df  
            if genome_dict is not None:
                del genome_dict
            if db is not None:
                del db
            
            # clean up temp database
            if db_path != ":memory:" and os.path.exists(db_path) and unique_id in str(db_path):
                try:
                    os.unlink(db_path)
                except Exception as e:
                    self._log_error(
                        "CLEANUP_ERROR", "Failed to clean up temporary database", 
                        strain_id=strain_id, files={"db_path": db_path}, exception=e
                    )
                
            if not using_external_progress:
                progress.stop()
                
            gc.collect()
                
        return results
        
    def extract_gene_sequences(
        self,
        systems_tsv_file: Union[pathlib.Path, str], 
        genes_tsv_file: Union[pathlib.Path, str], 
        faa_file: Union[pathlib.Path, str], 
        fna_file: Union[pathlib.Path, str],
        output_dir: Optional[Union[pathlib.Path, str]] = None,
        strain_id: Optional[str] = None,
        activity_filter: str = "defense",
        representatives: Optional[Container[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract gene sequences (protein and nucleotide) for defense systems
        
        Args:
            systems_tsv_file: Path to DefenseFinder systems TSV
            genes_tsv_file: Path to DefenseFinder genes TSV
            faa_file: Path to protein FASTA file
            fna_file: Path to nucleotide FASTA file
            output_dir: Directory to write output files (if write_output=True)
            strain_id: Optional strain identifier
            activity_filter: Filter systems by activity type (default: "defense")
            representatives: Optional set of representative cluster IDs to extract (for efficiency)
            
        Returns:
            Dictionary mapping system_id to gene sequence data
        """
        if self.write_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # files for error logging
        files_dict = {
            "systems_tsv": systems_tsv_file,
            "genes_tsv": genes_tsv_file,
            "faa_file": faa_file,
            "fna_file": fna_file
        }
        
        if output_dir:
            files_dict["output_dir"] = output_dir
            
        using_external_progress = self.progress is not None
        progress = self.progress or rich.progress.Progress(
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TimeElapsedColumn(),
            rich.progress.TimeRemainingColumn(),
            rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
        )
        
        results = {}
        
        try:
            if not using_external_progress:
                progress.start()
                
            try:
                systems_df = pd.read_csv(systems_tsv_file, sep='\t')

                # filter systems by activity if applicable 
                if activity_filter.lower() != "all":
                    if 'activity' in systems_df.columns:
                        systems_df = systems_df[systems_df['activity'].str.lower() == activity_filter.lower()]
                
                # check for duplicate systems after activity filtering (silent removal)
                duplicate_mask = systems_df.duplicated(keep='first')
                if duplicate_mask.any():
                    duplicate_systems = systems_df[duplicate_mask]['sys_id'].tolist()
                    n_duplicates = len(duplicate_systems)
                    
                    # duplicates: keep first occurrence
                    # warnings have already been logged
                    systems_df = systems_df.drop_duplicates(keep='first')

                # representatives filtering (for computational efficiency)
                if representatives is not None:
                    clean_strain = self._get_unique_strain_id(strain_id)
                    # unique system IDs to match against representatives
                    systems_df['unique_sys_id'] = clean_strain + '@' + systems_df['sys_id']
                    original_count = len(systems_df)
                    
                    try:
                        # try iterable first
                        systems_df = systems_df[systems_df['unique_sys_id'].isin(representatives)]
                    except TypeError:
                        # __contains__ method for non-iterable containers
                        mask = systems_df['unique_sys_id'].apply(lambda x: x in representatives)
                        systems_df = systems_df[mask]
                    
                    filtered_count = len(systems_df)
                    
                    if self.verbose and self.progress:
                        self.progress.console.print(
                            f"[bold cyan]{'Filtered':>12}[/] {original_count} systems to {filtered_count} "
                            f"representatives for {strain_id}"
                        )

                genes_df = pd.read_csv(genes_tsv_file, sep='\t')
            except Exception as e:
                self._log_error(
                    "TSV_ERROR", "Failed to read TSV files for gene extraction", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to read TSV files: {str(e)}")
                return {}
                
            faa_index = None
            fna_index = None
            try:
                if os.path.exists(faa_file) and os.path.isfile(faa_file): 
                    faa_index = self._create_fasta_index(faa_file)
                # only create nucleotide index if extraction is enabled
                # useful for debugging but not functional 
                if self.extract_nucleotides and os.path.exists(fna_file) and os.path.isfile(fna_file): 
                    fna_index = self._create_fasta_index(fna_file)
                    
                # at least protein index must be created
                if faa_index is None:
                    self.console.print(f"[bold red]Error:[/] No valid protein FASTA file provided")
                    return {}
                    
            except Exception as e:
                self._log_error(
                    "INDEX_ERROR", "Failed to create sequence indices", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to create sequence indices: {str(e)}")
                return {}
                
            
            for _, system in systems_df.iterrows():
                sys_id = system['sys_id']
                clean_strain = self._get_unique_strain_id(strain_id)
                unique_sys_id = f"{clean_strain}@{sys_id}"
    
                result = self._extract_sequences_for_system(
                    unique_sys_id, genes_df, faa_index, fna_index, output_dir, strain_id, sys_id, progress
                )
                
                if result:
                    results[unique_sys_id] = result
                
        except Exception as e:
            self._log_error(
                "GENE_EXTRACTION_ERROR", "Failed to extract gene sequences", 
                strain_id=strain_id, files=files_dict, exception=e
            )
            self.console.print(f"[bold red]Error:[/] Failed to extract gene sequences: {str(e)}")
            
        finally:
            if not using_external_progress:
                progress.stop()

        total_proteins = sum(len(system.get('proteins', {})) for system in results.values())
        total_nucleotides = sum(len(system.get('nucleotides', {})) for system in results.values())
        
        if self.extract_nucleotides:
            progress.console.print(f"[bold green]{'Extracted':>12}[/] {total_proteins} proteins and {total_nucleotides} nucleotides from {len(results)} systems for [bold cyan]{strain_id}[/]")
        else:
            progress.console.print(f"[bold green]{'Extracted':>12}[/] {total_proteins} proteins from {len(results)} systems for [bold cyan]{strain_id}[/]")

        return results
    
    def _process_sequence_type(
        self, 
        index: Any, 
        matching_seq_ids: List[str], 
        hit_ids: set, 
        system_id: str, 
        sequence_type: str,
        output_file: Optional[str],
        result: Dict[str, Any],
        files_dict: Dict[str, Any],
        strain_id: Optional[str],
        progress: Optional[rich.progress.Progress]
    ) -> int:
        """Helper function to process protein or nucleotide sequences"""
        if index is None:
            return 0
            
        buffer = io.StringIO() if not self.write_output else None
        count = 0
        
        try:
            out_handle = open(output_file, "w") if output_file else buffer
            
            for seq_id in matching_seq_ids:
                record = index[seq_id]
                gene_id = self._extract_protein_id(seq_id, hit_ids)
                
                # ID with system for downstream processing
                modified_id = f"{system_id}@@{gene_id}"
                modified_desc = f"{modified_id} [system={system_id}] [{sequence_type}={gene_id}]"
                
                # store in result section
                result_key = "proteins" if sequence_type == "protein" else "nucleotides"
                id_key = "protein_id" if sequence_type == "protein" else "nucleotide_id"
                unique_key = "unique_protein_id" if sequence_type == "protein" else "unique_nucleotide_id"
                
                result[result_key][gene_id] = {
                    "sequence": str(record.seq),
                    "length": len(record.seq),
                    "system_id": system_id,
                    id_key: gene_id,
                    unique_key: modified_id
                }
                
                self._write_fasta(out_handle, modified_id, modified_desc, str(record.seq))
                count += 1
                
            if output_file and out_handle:
                out_handle.close()
            elif buffer:
                # store buffer content in result
                fasta_key = "protein_fasta" if sequence_type == "protein" else "nucleotide_fasta"
                result[fasta_key] = buffer.getvalue()
                buffer.close()
                
        except Exception as e:
            error_type = f"{sequence_type.upper()}_WRITE_ERROR"
            self._log_error(
                error_type, f"Failed to process {sequence_type} sequences for system {system_id}", 
                strain_id=strain_id, system_id=system_id, files=files_dict, exception=e
            )
            if progress:
                progress.console.print(f"[bold red]Error:[/] Failed to process {sequence_type} sequences for {system_id}: {str(e)}")
        
        return count

    def _extract_sequences_for_system(
        self,
        system_id: str, 
        genes_df: pd.DataFrame, 
        faa_index: Any, 
        fna_index: Any, 
        output_dir: Optional[Union[pathlib.Path, str]],
        strain_id: Optional[str],
        original_sys_id: str = "",
        progress: Optional[rich.progress.Progress] = None,
        verbose: bool = False, 
    ) -> Dict[str, Any]:
        """Extract gene sequences for a single defense system"""
        # for error logging
        files_dict: Dict[str, Any] = {"system_id": system_id}
        if output_dir:
            files_dict.update({
                "output_dir": output_dir,
                "faa_output": os.path.join(str(output_dir), f"{system_id}.faa") if self.write_output else None,
                "fna_output": os.path.join(str(output_dir), f"{system_id}.fna") if self.write_output else None
            })
        
        # filter only genes for this system
        lookup_sys_id = original_sys_id or system_id.split('@')[-1]
        system_genes = genes_df[genes_df['sys_id'] == lookup_sys_id] 
        
        if system_genes.empty:
            self._log_error(
                "NO_SYSTEM_GENES_ERROR", f"No genes found for system {system_id}", 
                strain_id=strain_id, system_id=system_id, files=files_dict
            )
            if progress:
                progress.console.print(f"[bold yellow]Warning:[/] No genes found for system {system_id}")
            return {}
        
        hit_ids = set(system_genes['hit_id'].unique())
        
        faa_out = os.path.join(str(output_dir), f"{system_id}.faa") if self.write_output and output_dir else None
        fna_out = os.path.join(str(output_dir), f"{system_id}.fna") if self.write_output and output_dir else None
        
        result = {
            "system_id": system_id,
            "proteins": {},
            "nucleotides": {}
        }
        
        try:
            faa_matching = []
            fna_matching = []
            
            if faa_index is not None:
                faa_matching = self._find_matching_sequences(faa_index, hit_ids, "protein")
                if self.verbose and progress:
                    progress.console.print(f"[bold blue]{'Found':>22}[/] {len(faa_matching)} protein matches for {len(hit_ids)} hit_ids in [cyan]{system_id}[/]")
            
            if fna_index is not None:
                fna_matching = self._find_matching_sequences(fna_index, hit_ids, "nucleotide")
                if self.verbose and progress:
                    progress.console.print(f"[bold blue]{'Found':>22}[/] {len(fna_matching)} nucleotide matches for {len(hit_ids)} hit_ids in [cyan]{system_id}[/]")
            
            faa_count = 0
            # process protein sequences only if faa_index exists
            if faa_index is not None:
                faa_buffer = io.StringIO() if not self.write_output else None
                
                try:
                    out_handle = open(faa_out, "w") if faa_out else faa_buffer
                    
                    for seq_id in faa_matching:
                        record = faa_index[seq_id]
                        
                        protein_id = self._extract_protein_id_from_record(record, seq_id, hit_ids)

                        # create modified ID with system for downstream processing
                        # retains system and protein information 
                        modified_id = f"{system_id}@@{protein_id}"
                        modified_desc = f"{modified_id} [system={system_id}] [protein={protein_id}]"
                        
                        result["proteins"][protein_id] = {
                            "sequence": str(record.seq),
                            "length": len(record.seq),
                            "system_id": system_id,
                            "protein_id": protein_id,
                            "unique_protein_id": modified_id
                        }
                        
                        self._write_fasta(out_handle, modified_id, modified_desc, str(record.seq))
                        faa_count += 1
                        
                    if faa_out and out_handle:
                        out_handle.close()
                    elif faa_buffer:
                        # store buffer content in result
                        result["protein_fasta"] = faa_buffer.getvalue()
                        faa_buffer.close()
                        
                except Exception as e:
                    self._log_error(
                        "PROTEIN_WRITE_ERROR", f"Failed to process protein sequences for system {system_id}", 
                        strain_id=strain_id, system_id=system_id, files=files_dict, exception=e
                    )
                    if progress:
                        progress.console.print(f"[bold red]Error:[/] Failed to process protein sequences for {system_id}: {str(e)}")
            
            # process nucleotide sequences only if extraction is enabled and fna_index exists
            fna_count = 0
            if self.extract_nucleotides and fna_index is not None:
                fna_buffer = io.StringIO() if not self.write_output else None
                
                try:
                    out_handle = open(fna_out, "w") if fna_out else fna_buffer
                    
                    for seq_id in fna_matching:
                        record = fna_index[seq_id]
                        nucleotide_id = self._extract_protein_id_from_record(record, seq_id, hit_ids)
                        
                        # create modified ID with system for downstream processing
                        modified_id = f"{system_id}@@{nucleotide_id}" 
                        modified_desc = f"{modified_id} [system={system_id}] [nucleotide={nucleotide_id}]"
                        
                        result["nucleotides"][nucleotide_id] = {
                            "sequence": str(record.seq),
                            "length": len(record.seq),
                            "system_id": system_id,
                            "nucleotide_id": nucleotide_id,
                            "unique_nucleotide_id": modified_id
                        }
                        
                        self._write_fasta(out_handle, modified_id, modified_desc, str(record.seq))
                        fna_count += 1
                        
                    if fna_out and out_handle:
                        out_handle.close()
                    elif fna_buffer:
                        # store buffer content in result
                        result["nucleotide_fasta"] = fna_buffer.getvalue()
                        fna_buffer.close()
                        
                except Exception as e:
                    self._log_error(
                        "NUCLEOTIDE_WRITE_ERROR", f"Failed to process nucleotide sequences for system {system_id}", 
                        strain_id=strain_id, system_id=system_id, files=files_dict, exception=e
                    )
                    if progress:
                        progress.console.print(f"[bold red]Error:[/] Failed to process nucleotide sequences for system {system_id}: {str(e)}")
            
            if self.write_output:
                if faa_out and faa_index is not None:
                    result["faa_file"] = faa_out
                if self.extract_nucleotides and fna_out and fna_index is not None:
                    result["fna_file"] = fna_out
            
            if progress and self.verbose:
                if faa_index is not None and self.extract_nucleotides and fna_index is not None:
                    progress.console.print(f"[bold blue]{'Extracted':>22}[/] {faa_count} proteins and {fna_count} nucleotides from [cyan]{system_id}[/] system")
                elif faa_index is not None:
                    progress.console.print(f"[bold green]{'Extracted':>22}[/] {faa_count} proteins from [cyan]{system_id}[/] system")
                elif self.extract_nucleotides and fna_index is not None:
                    progress.console.print(f"[bold green]{'Extracted':>22}[/] {fna_count} nucleotides from [cyan]{system_id}[/] system")
                else:
                    progress.console.print(f"[bold yellow]{'Warning':>22}[/] No sequences extracted for [cyan]{system_id}[/] system")

            return result
            
        except Exception as e:
            self._log_error(
                "SEQUENCE_EXTRACTION_ERROR", f"Uncaught error extracting sequences for system {system_id}", 
                strain_id=strain_id, system_id=system_id, files=files_dict, exception=e
            )
            if progress:
                progress.console.print(f"[bold red]Error:[/] Failed to extract sequences for {system_id}: {str(e)}")
            return {}
    
    def _find_matching_sequences(self, seq_index, hit_ids, seq_type="protein"):
        """
        Find matching sequences using multiple strategies for robust matching.
        
        Args:
            seq_index: FASTA sequence index (from SeqIO)
            hit_ids: Set of hit_ids (locus_tags) to match
            seq_type: Type of sequences ("protein" or "nucleotide") for logging
            
        Returns:
            List of matching sequence IDs from the index
        """
        matching_sequences = []
        hit_ids_found = set()
        
        for seq_id in seq_index:
            seq_record = seq_index[seq_id]
            
            # 1 - direct substring match
            for hit_id in hit_ids:
                if hit_id in seq_id:
                    matching_sequences.append(seq_id)
                    hit_ids_found.add(hit_id)
                    break
            else:
                # 2 - parse locus_tag from FASTA description/header
                # look for [locus_tag=XXXXX] pattern in sequence description
                if hasattr(seq_record, 'description'):
                    description = seq_record.description
                    locus_tag_match = re.search(r'\[locus_tag=([^\]]+)\]', description)
                    if locus_tag_match:
                        locus_tag = locus_tag_match.group(1)
                        if locus_tag in hit_ids:
                            matching_sequences.append(seq_id)
                            hit_ids_found.add(locus_tag)
                            continue

                # 3 - parse from complex sequence ID formats
                # handle formats like "lcl|NC_015593.1_prot_WP_013846339.1_1"
                # and extract potential locus_tags from embedded information
                for hit_id in hit_ids:
                    # check if hit_id appears anywhere in the full sequence record
                    full_record_text = f"{seq_id} {getattr(seq_record, 'description', '')}"
                    if hit_id in full_record_text:
                        matching_sequences.append(seq_id)
                        hit_ids_found.add(hit_id)
                        break
        
        # log missing hit_ids 
        missing_hit_ids = hit_ids - hit_ids_found
        if missing_hit_ids and self.verbose:
            missing_str = ', '.join(list(missing_hit_ids)[:3])
            if len(missing_hit_ids) > 3:
                missing_str += f" (and {len(missing_hit_ids) - 3} more)"

            if len(hit_ids_found) / len(hit_ids) < 0.9:  # less than 90% found
                if self.progress:
                    self.progress.console.print(
                        f"[bold yellow]{'Warning':>22}[/] Missing {seq_type} sequences for hit_ids: {missing_str}"
                    )
        
        return matching_sequences
    
    def _find_gene_coordinates(self, db, gene_id):
        """Find gene coordinates in GFF database using multiple lookup strategies"""
        
        # 1 - direct exact ID match
        try:
            feature = db[gene_id]
            return (feature.start, feature.end, feature.seqid) if feature.featuretype in ['gene', 'CDS', 'mRNA'] else (None, None, None)
        except Exception:
            pass

        # 2 - gene ID format (gene-LOCUS_TAG)
        try:
            gene_id_format = f"gene-{gene_id}"
            feature = db[gene_id_format]
            return (feature.start, feature.end, feature.seqid) if feature.featuretype in ['gene', 'CDS', 'mRNA', 'pseudogene'] else (None, None, None)
        except Exception:
            pass

        # 3 - enhanced MAG format parsing (MAGid~contigID_geneIndex)
        try:
            if '~' in gene_id and '_' in gene_id:
                # parse MAG format: MAGid~contigID_geneIndex
                # example: '20100900_E1D_4~c_000000000699_10'
                parts = gene_id.split('~')
                if len(parts) == 2:
                    mag_id = parts[0]  # '20100900_E1D_4'
                    contig_gene = parts[1]  # 'c_000000000699_10'
                    
                    if '_' in contig_gene:
                        # split from the right to handle contig names with underscores
                        contig_id, gene_index = contig_gene.rsplit('_', 1)  # 'c_000000000699', '10'
                        
                        # try multiple MAG-based ID patterns commonly found in GFF files
                        mag_patterns = [
                            f"{mag_id}~{contig_id}_{gene_index}",                            
                            # no MAG prefix (contig + gene only)
                            f"{contig_id}_{gene_index}",                            
                            # with gene- prefix (standard GFF format)
                            f"gene-{mag_id}~{contig_id}_{gene_index}",
                            f"gene-{contig_id}_{gene_index}",
                            # underscore separator instead of tilde
                            f"{mag_id}_{contig_id}_{gene_index}",
                            
                            # CDS format variations
                            f"cds-{mag_id}~{contig_id}_{gene_index}",
                            f"cds-{contig_id}_{gene_index}",
                            
                            # Prokka/PGAP style with sequence ID
                            f"{contig_id}_{gene_index:0>5}",  # Zero-padded gene index
                            f"{mag_id}_{contig_id}_{gene_index:0>5}",
                            
                            # alternative separators
                            f"{mag_id}-{contig_id}_{gene_index}",
                            f"{mag_id}.{contig_id}_{gene_index}",
                            
                            contig_id,
                            
                            # MAG-specific variations?
                            f"{mag_id}~{contig_id}_gene_{gene_index}",
                            f"{contig_id}_gene_{gene_index}",
                            f"gene_{gene_index}",
                        ]
                        
                        for pattern in mag_patterns:
                            try:
                                feature = db[pattern]
                                if feature.featuretype in ['gene', 'CDS', 'mRNA', 'pseudogene', 'tRNA', 'rRNA']:
                                    return (feature.start, feature.end, feature.seqid)
                            except Exception:
                                continue
        except Exception:
                pass

        # 4 - attribute-based lookup with MAG-aware search
        try:
            if '~' in gene_id and '_' in gene_id:
                # parse components for attribute-based search
                parts = gene_id.split('~')
                if len(parts) == 2:
                    mag_id = parts[0]
                    contig_gene = parts[1]
                    contig_id, gene_index = contig_gene.rsplit('_', 1)
                    
                    # search by various attribute combinations
                    for feature_type in ['gene', 'CDS', 'mRNA', 'pseudogene']:
                        for feature in db.features_of_type(feature_type):
                            attrs = feature.attributes
                            
                            # check various attribute patterns
                            search_patterns = [
                                gene_id,  # exact match
                                f"{contig_id}_{gene_index}",  # without MAG
                                f"{mag_id}_{contig_id}_{gene_index}",  # underscore format
                                contig_id,  # just contig
                                gene_index,  # just gene index
                            ]
                            
                            # check all attributes for matches
                            for attr_key, attr_values in attrs.items():
                                if isinstance(attr_values, list):
                                    attr_values = [str(v) for v in attr_values]
                                else:
                                    attr_values = [str(attr_values)]
                                
                                for pattern in search_patterns:
                                    if str(pattern) in attr_values:
                                        return (feature.start, feature.end, feature.seqid)
                                    
                            # special check for ID attribute containing our components
                            id_attr = attrs.get('ID', [''])[0]
                            if any(component in str(id_attr) for component in [mag_id, contig_id, gene_index]):
                                return (feature.start, feature.end, feature.seqid)
            else:
                # standard locus_tag attribute search for non-MAG formats
                for feature_type in ['gene', 'CDS']:
                    for feature in db.features_of_type(feature_type):
                        locus_tag_attr = feature.attributes.get('locus_tag', [None])[0]
                        if locus_tag_attr == gene_id:
                            return (feature.start, feature.end, feature.seqid)
        except Exception as e:
            self._log_error(
                "GENE_LOOKUP_ERROR", 
                f"Failed to find gene {gene_id} in GFF database using all strategies", 
                exception=e
            )

        # 5 - positional search for MAG contigs
        try:
            if '~' in gene_id and '_' in gene_id:
                parts = gene_id.split('~')
                if len(parts) == 2:
                    mag_id = parts[0]
                    contig_gene = parts[1]
                    contig_id, gene_index = contig_gene.rsplit('_', 1)
                    
                    try:
                        gene_index_int = int(gene_index)
                        
                        # find genes on the specific contig and get the nth gene
                        contig_genes = []
                        for feature_type in ['gene', 'CDS']:
                            for feature in db.features_of_type(feature_type):
                                # check if feature is on our contig
                                if (feature.seqid == contig_id or 
                                    contig_id in feature.seqid or 
                                    feature.seqid.endswith(contig_id)):
                                    contig_genes.append(feature)
                        
                        # sort by position and get the gene at the specified index
                        contig_genes.sort(key=lambda x: x.start)
                        if 1 <= gene_index_int <= len(contig_genes):
                            target_gene = contig_genes[gene_index_int - 1]  # 1-based indexing
                            return (target_gene.start, target_gene.end, target_gene.seqid)
                            
                    except ValueError:
                        pass  # gene_index is not an integer
        except Exception:
            pass
        
        return None, None, None
    
    def _create_fasta_index(self, fasta_file, reindex=False):
        """Create an in-memory index for a FASTA file"""
        try:
            return SeqIO.index(fasta_file, "fasta")
        except Exception as e:
            self._log_error(
                "INDEX_ERROR", f"Failed to create FASTA index for {fasta_file}", 
                exception=e
            )
            self.console.print(f"[bold red]Error:[/] Failed to index {os.path.basename(fasta_file)}: {str(e)}")
            return None

    def _extract_protein_id_from_record(self, record, seq_id, hit_ids):
        """Extract protein ID from a sequence record using the record's description"""
        # 1 - parse from FASTA description using regex
        # look for [locus_tag=XXXXX] pattern in the record description
        description = getattr(record, 'description', seq_id)
        
        locus_tag_match = re.search(r'\[locus_tag=([^\]]+)\]', description)
        if locus_tag_match:
            locus_tag = locus_tag_match.group(1)
            if locus_tag in hit_ids:
                return locus_tag
        
        # 2 - direct match with sequence ID
        if seq_id in hit_ids:
            return seq_id
        
        # 3 - substring match in sequence ID    
        for hit_id in hit_ids:
            if hit_id in seq_id:
                return hit_id
        
        # 4 - substring match in full description
        for hit_id in hit_ids:
            if hit_id in description:
                return hit_id
                
        # default: return the first hit_id as fallback (maintains functionality)
        # but log this as it might indicate a matching problem
        fallback_id = list(hit_ids)[0] if hit_ids else seq_id.split()[0]
        if self.verbose and self.progress:
            self.progress.console.print(
                f"[bold yellow]{'Warning':>22}[/] Could not match seq_id '{seq_id[:50]}...' with hit_ids, using fallback: {fallback_id}"
            )
        return fallback_id

    def _extract_protein_id(self, seq_id, hit_ids):
        """Extract the protein ID from a sequence ID using multiple matching strategies"""
        # 1 - direct match
        if seq_id in hit_ids:
            return seq_id

        # 2 - substring match
        for hit_id in hit_ids:
            if hit_id in seq_id:
                return hit_id
        
        # 3 - parse from FASTA header using regex
        # look for [locus_tag=XXXXX] pattern
        locus_tag_match = re.search(r'\[locus_tag=([^\]]+)\]', seq_id)
        if locus_tag_match:
            locus_tag = locus_tag_match.group(1)
            if locus_tag in hit_ids:
                return locus_tag
        
        # 4 - extract from complex sequence ID formats
        # handle formats like "lcl|NC_015593.1_prot_WP_013846339.1_1" 
        # try to find a hit_id that appears anywhere in the seq_id
        for hit_id in hit_ids:
            if hit_id in seq_id:
                return hit_id
                
        # default: return the first hit_id as fallback (maintains functionality)
        # but log this as it might indicate a matching problem
        fallback_id = list(hit_ids)[0] if hit_ids else seq_id
        if self.verbose and self.progress:
            self.progress.console.print(
                f"[bold yellow]{'Warning':>22}[/] Could not match seq_id '{seq_id[:50]}...' with hit_ids, using fallback: {fallback_id}"
            )
        return fallback_id


    def _write_fasta(self, handle, seq_id, description, sequence):
        """Write a sequence in FASTA format"""
        handle.write(f">{description}\n")
        handle.write(sequence)
        handle.write("\n")
        return None
    
    def _log_error(self, error_type, message, strain_id=None, system_id=None, files=None, exception=None):
        """Log error information"""
        
        # use workdir or temp directory if no output_base_dir set
        if not self.output_base_dir:
            log_dir = tempfile.gettempdir()
            log_file = os.path.join(log_dir, "igua_defense_extraction_errors.log")
        else:
            log_file = os.path.join(str(self.output_base_dir), "defense_extraction_errors.log")
        
            
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {error_type}: {message}\n"
        
        if strain_id:
            log_message += f"  Strain: {strain_id}\n"
        
        if system_id:
            log_message += f"  System: {system_id}\n"
        
        if files:
            log_message += "  Files:\n"
            for key, path in files.items():
                if path:
                    log_message += f"    {key}: {path}\n"
        
        if exception:
            log_message += f"  Exception: {str(exception)}\n"
            if hasattr(exception, "__traceback__"):
                log_message += f"  Traceback:\n{''.join(traceback.format_tb(exception.__traceback__))}\n"

        log_message += "-" * 80 + "\n"
                
        with open(log_file, "a") as f:
            f.write(log_message)
        
        return None