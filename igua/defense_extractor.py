import os
import pathlib
import pandas as pd
import tempfile
import uuid
import gffutils
import io
from Bio import SeqIO
from typing import Dict, List, Tuple, Optional, Union, Any
import rich.progress
from rich.console import Console

# from memory_profiler import profile
from .profiler import profiler

# TODO: deduplicate fna/faa sequence extraction 
# TODO: check memory usage of extractor 


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
    ):
        self.progress = progress
        self.output_base_dir = output_base_dir
        self.write_output = write_output
        self.console = progress.console if progress else Console()
        self.verbose = verbose

    def _get_unique_strain_id(self, strain_id: Optional[str]) -> str:
        """Get or create a unique strain identifier"""
        if strain_id:
            # clean strain_id to avoid filesystem issues
            clean_strain = strain_id.replace('/', '_').replace('\\', '_').replace(' ', '_')

            return clean_strain
        
        # if None strain_id, generate a unique identifier
        unique_ref = str(uuid.uuid4())[:8]
        return unique_ref

    @profiler.profile_function
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
        
        # set up database path
        db_path = ":memory:"
        unique_id = str(uuid.uuid4())
        
        # for error logging
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
        
        try:
            if not using_external_progress:
                progress.start()
            
            # load TSV files
            try:
                systems_df = pd.read_csv(systems_tsv_file, sep='\t')
                
                # filter systems by activity if applicable
                original_count = len(systems_df)
                if activity_filter.lower() != "all":
                    if 'activity' in systems_df.columns:
                        systems_df = systems_df[systems_df['activity'].str.lower() == activity_filter.lower()]
                        filtered_count = len(systems_df)
                        
                # check for duplicate systems after activity filtering
                # note: only complete duplicates are omitted
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
                    
                    # duplicates: keep first occurrence
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

            except Exception as e:
                self._log_error(
                    "TSV_ERROR", "Failed to read TSV files", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to read TSV files: {str(e)}")
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
            
            # set up GFF database with in-memory fallback
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
                    id_spec=['ID', 'Name', 'locus_tag', 'gene']
                )
            except Exception as e:
                self._log_error(
                    "GFF_ERROR", "Failed to create GFF database, falling back to in-memory", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold blue]{'Using':>12}[/] in-memory database")
                db_path = ":memory:"  # reset to in-memory in case of exception
                try:
                    db = gffutils.create_db(
                        str(gff_file),  # convert Path to string here
                        dbfn=":memory:", 
                        merge_strategy='create_unique',
                        id_spec=['ID', 'Name', 'locus_tag', 'gene']
                    )
                except Exception as e2:
                    self._log_error(
                        "GFF_FATAL_ERROR", "Failed to create in-memory GFF database", 
                        strain_id=strain_id, files=files_dict, exception=e2
                    )
                    self.console.print(f"[bold red]Error:[/] Failed to create GFF database: {str(e2)}")
                    return {}
            
            for _, system in systems_df.iterrows():
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
                    continue
                
                # beginning and ending genes
                try:
                    # directly access the start and end genes using 'sys_beg' and 'sys_end' from the systems tsv DefenseFinder output 
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
                        continue
                    
                    beg_hit_id = system['sys_beg']
                    end_hit_id = system['sys_end']
                    
                    # get genomic coordinates from the GFF database
                    start_coord, start_end, start_seq_id = self._find_gene_coordinates(db, beg_hit_id)
                    end_coord, end_end, end_seq_id = self._find_gene_coordinates(db, end_hit_id)

                    # for complete system genomic sequence, use start of first gene and end of last gene
                    system_start = start_coord
                    system_end = end_end
                    
                except Exception as e:
                    self._log_error(
                        "GENE_DATA_ERROR", "Failed to extract gene information", 
                        strain_id=strain_id, system_id=sys_id, files=system_files, exception=e
                    )
                    self.console.print(f"[bold red]Error:[/] Failed to get gene data for system {sys_id}: {str(e)}")
                    continue
                
                # skip if coordinates invalid
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
                    continue
                    
                if start_seq_id != end_seq_id:
                    self._log_error(
                        "SEQUENCE_SPAN_ERROR", "System spans multiple sequences", 
                        strain_id=strain_id, system_id=sys_id, files=system_files,
                        exception={"start_seq_id": start_seq_id, "end_seq_id": end_seq_id}
                    )
                    self.console.print(f"[bold yellow]{'Warning':>12}[/] System {sys_id} spans multiple sequences")
                    continue
                    
                seq_id = start_seq_id
                
                # valid coordinates - swap if needed
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
                if seq_id in genome_dict:
                    try:
                        genome_seq = genome_dict[seq_id].seq
                        sequence = genome_seq[max(0, system_start-1):min(len(genome_seq), system_end)]
                        sequence_length = len(sequence)
                        
                        results[unique_sys_id] = {
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
                            results[unique_sys_id]["file_path"] = output_file
                            
                        if self.verbose: 
                            self.console.print(f"[bold blue]{'Extracted':>22}[/] genomic sequence for [cyan]{sys_id}[/] system ({sequence_length} bp)")
                        
                    except Exception as e:
                        self._log_error(
                            "SEQUENCE_EXTRACTION_ERROR", "Failed to extract or write sequence", 
                            strain_id=strain_id, system_id=sys_id, files=system_files, exception=e
                        )
                        self.console.print(f"[bold red]Error:[/] Failed to extract sequence for {sys_id}: {str(e)}")
                else:
                    self._log_error(
                        "SEQUENCE_ID_ERROR", f"Sequence ID {seq_id} not found in genome", 
                        strain_id=strain_id, system_id=sys_id, files=system_files
                    )
                    self.console.print(f"[bold red]{'Error':>12}[/] Sequence {seq_id} not found in genome")
                    
                # progress.advance(task_systems)

            self.console.print(f"[bold green]{'Extracted':>12}[/] {len(results)} defense systems for [bold cyan]{strain_id}[/]")
            
        except Exception as e:
            self._log_error(
                "EXTRACTION_FATAL_ERROR", "Uncaught exception in extraction process", 
                strain_id=strain_id, files=files_dict, exception=e
            )
            self.console.print(f"[bold red]Fatal Error:[/] {str(e)}")
            
        finally:
            # clean up temp database
            if db_path != ":memory:" and os.path.exists(db_path) and (not gff_cache_dir or unique_id in db_path):
                try:
                    os.unlink(db_path)
                except Exception as e:
                    self._log_error(
                        "CLEANUP_ERROR", "Failed to clean up temporary database", 
                        strain_id=strain_id, files={"db_path": db_path}, exception=e
                    )
                
            if not using_external_progress:
                progress.stop()
                
        return results
        
    @profiler.profile_function
    def extract_gene_sequences(
        self,
        systems_tsv_file: Union[pathlib.Path, str], 
        genes_tsv_file: Union[pathlib.Path, str], 
        faa_file: Union[pathlib.Path, str], 
        fna_file: Union[pathlib.Path, str],
        output_dir: Optional[Union[pathlib.Path, str]] = None,
        strain_id: Optional[str] = None,
        activity_filter: str = "defense",
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
                if os.path.exists(fna_file) and os.path.isfile(fna_file): 
                    fna_index = self._create_fasta_index(fna_file)
                    
                # at least one index was created
                if faa_index is None and fna_index is None:
                    self.console.print(f"[bold red]Error:[/] No valid FASTA files provided")
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
        progress.console.print(f"[bold green]{'Extracted':>12}[/] {total_proteins} proteins and {total_nucleotides} nucleotides from {len(results)} systems for [bold cyan]{strain_id}[/]")

        return results
    
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
        # file paths for error logging
        files_dict = {"system_id": system_id}
        if output_dir:
            files_dict.update({
                "output_dir": output_dir,
                "faa_output": os.path.join(output_dir, f"{system_id}.faa") if self.write_output else None,
                "fna_output": os.path.join(output_dir, f"{system_id}.fna") if self.write_output else None
            }) # type: ignore
        
        # filter only genes for this system
        lookup_sys_id = original_sys_id or system_id.split('@')[-1] # original sys_id
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
            # find matching sequences
            faa_matching = []
            fna_matching = []
            
            if faa_index is not None:
                faa_matching = [
                    seq_id for seq_id in faa_index if any(hit_id in seq_id for hit_id in hit_ids)
                ]
            
            if fna_index is not None:
                fna_matching = [
                    seq_id for seq_id in fna_index if any(hit_id in seq_id for hit_id in hit_ids)
                ]
            
            faa_count = 0
            # process protein sequences only if faa_index exists
            if faa_index is not None:
                faa_buffer = io.StringIO() if not self.write_output else None
                
                try:
                    out_handle = open(faa_out, "w") if faa_out else faa_buffer
                    
                    for seq_id in faa_matching:
                        record = faa_index[seq_id]
                        
                        protein_id = self._extract_protein_id(seq_id, hit_ids)

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
            
            # process nucleotide sequences only if fna_index exists
            fna_count = 0
            if fna_index is not None:
                fna_buffer = io.StringIO() if not self.write_output else None
                
                try:
                    out_handle = open(fna_out, "w") if fna_out else fna_buffer
                    
                    for seq_id in fna_matching:
                        record = fna_index[seq_id]
                        nucleotide_id = self._extract_protein_id(seq_id, hit_ids)
                        
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
                if fna_out and fna_index is not None:
                    result["fna_file"] = fna_out
            
            if progress and self.verbose:
                if faa_index is not None and fna_index is not None:
                    progress.console.print(f"[bold blue]{'Extracted':>22}[/] {faa_count} proteins and {fna_count} nucleotides from [cyan]{system_id}[/] system")
                elif faa_index is not None:
                    progress.console.print(f"[bold green]{'Extracted':>22}[/] {faa_count} proteins from [cyan]{system_id}[/] system")
                elif fna_index is not None:
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
    
    def _find_gene_coordinates(self, db, gene_id):
        """Find gene coordinates in GFF database"""
        # direct exact ID match
        try:
            feature = db[gene_id]
            return (feature.start, feature.end, feature.seqid) if feature.featuretype in ['gene', 'CDS', 'mRNA'] else (None, None, None)
        except Exception as e:
            self._log_error(
                "GENE_LOOKUP_ERROR", 
                f"Failed to find gene {gene_id} in GFF database", 
                exception=e
            )
            pass
        
        return None, None, None
    
    def _create_fasta_index(self, fasta_file, reindex=False):
        """Create an index for a FASTA file"""
        index_file = f"{fasta_file}.idx"
        if reindex or not os.path.exists(index_file):
            return SeqIO.index_db(index_file, fasta_file, "fasta")
        return SeqIO.index_db(index_file)
    
    def _extract_protein_id(self, seq_id, hit_ids):
        """Extract the protein ID from a sequence ID using the hit IDs for matching"""
        # try direct match
        if seq_id in hit_ids:
            return seq_id
            
        # try to extract from sequence ID
        for hit_id in hit_ids:
            if hit_id in seq_id:
                return hit_id
                
        # Default to sequence ID if no match
        return seq_id
    
    def _write_fasta(self, handle, seq_id, description, sequence):
        """Write a sequence in FASTA format"""
        handle.write(f">{description}\n")
        handle.write(sequence)
        handle.write("\n")
        return None
    
    def _log_error(self, error_type, message, strain_id=None, system_id=None, files=None, exception=None):
        """Log error information"""
        import time
        import traceback
        
        # skip logging if no output directory
        if not self.output_base_dir:
            return None
            
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
        
        log_file = os.path.join(str(self.output_base_dir), "defense_extraction_errors.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        with open(log_file, "a") as f:
            f.write(log_message)
        
        return None