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

# TODO: fix progress bar integration


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
        write_output: bool = False
    ):
        self.progress = progress
        self.output_base_dir = output_base_dir
        self.write_output = write_output
        self.console = progress.console if progress else Console()
        
    def extract_systems(
        self,
        systems_tsv_file: pathlib.Path, 
        genes_tsv_file: pathlib.Path, 
        gff_file: pathlib.Path, 
        fasta_file: pathlib.Path,
        output_dir: Optional[pathlib.Path] = None,
        gff_cache_dir: Optional[pathlib.Path] = None,
        strain_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract defense systems from genomic data
        
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
        # Create output directory if needed and writing is enabled
        if self.write_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Set up database path
        db_path = ":memory:"
        unique_id = str(uuid.uuid4())
        
        # Track files for error logging
        files_dict = {
            "systems_tsv": systems_tsv_file,
            "genes_tsv": genes_tsv_file,
            "gff_file": gff_file,
            "fasta_file": fasta_file
        }
        
        if output_dir:
            files_dict["output_dir"] = output_dir
        
        # Create progress tracking
        using_external_progress = self.progress is not None
        progress = self.progress or rich.progress.Progress(
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
        )
        
        # Dictionary to store results
        results = {}
        
        try:
            if not using_external_progress:
                progress.start()
            
            # Load input data
            # task_setup = progress.add_task("[bold blue]Setting up...", total=3)
            
            # Load TSV files
            # progress.update(task_setup, description="Reading TSV files")
            try:
                systems_df = pd.read_csv(systems_tsv_file, sep='\t')
                genes_df = pd.read_csv(genes_tsv_file, sep='\t')
            except Exception as e:
                self._log_error(
                    "TSV_ERROR", "Failed to read TSV files", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to read TSV files: {str(e)}")
                return {}
            # progress.advance(task_setup)
            
            # Load genome
            # progress.update(task_setup, description="Loading genome")
            try:
                genome_dict = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
            except Exception as e:
                self._log_error(
                    "FASTA_ERROR", "Failed to load genome FASTA", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to load genome: {str(e)}")
                return {}
            # progress.advance(task_setup)
            
            # Set up GFF database with in-memory fallback
            # progress.update(task_setup, description="Creating GFF database")
            try:
                if gff_cache_dir:
                    os.makedirs(gff_cache_dir, exist_ok=True)
                    db_path = os.path.join(str(gff_cache_dir), f"{os.path.basename(str(gff_file))}_{unique_id}.db")
                else:
                    db_path = os.path.join(tempfile.gettempdir(), f"gff_temp_{unique_id}.db")
                    
                # Convert pathlib.Path to string before passing to gffutils
                db = gffutils.create_db(
                    str(gff_file),  # Convert Path to string here
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
                db_path = ":memory:"  # Reset to in-memory in case of exception
                try:
                    db = gffutils.create_db(
                        str(gff_file),  # Convert Path to string here
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
                
            # progress.advance(task_setup)
            
            # Process each system
            # task_systems = progress.add_task("[bold blue]Processing systems", total=len(systems_df))
            
            for _, system in systems_df.iterrows():
                sys_id = system['sys_id']
                # progress.update(task_systems, description=f"Processing: {sys_id}")
                
                system_files = {**files_dict, "system_id": sys_id}
                    
                # Get system genes
                system_genes = genes_df[genes_df['sys_id'] == sys_id].sort_values('hit_pos')
                if system_genes.empty:
                    self._log_error(
                        "NO_GENES_ERROR", "No genes found for system", 
                        strain_id=strain_id, system_id=sys_id, files=system_files
                    )
                    self.console.print(f"[bold yellow]{'Warning':>12}[/] No genes found for system {sys_id}")
                    # progress.advance(task_systems)
                    continue
                
                # Get beginning and ending genes
                try:
                    # Directly access the start and end genes using 'sys_beg' and 'sys_end' from the system
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
                        # progress.advance(task_systems)
                        continue
                    
                    # Extract the hit IDs
                    beg_hit_id = system['sys_beg']
                    end_hit_id = system['sys_end']
                    
                    # Get genomic coordinates from the GFF database
                    start_coord, start_end, start_seq_id = self._find_gene_coordinates(db, beg_hit_id)
                    end_coord, end_end, end_seq_id = self._find_gene_coordinates(db, end_hit_id)

                    # For the complete system, use start of first gene and end of last gene
                    system_start = start_coord
                    system_end = end_end
                    
                except Exception as e:
                    self._log_error(
                        "GENE_DATA_ERROR", "Failed to extract gene information", 
                        strain_id=strain_id, system_id=sys_id, files=system_files, exception=e
                    )
                    self.console.print(f"[bold red]Error:[/] Failed to get gene data for system {sys_id}: {str(e)}")
                    # progress.advance(task_systems)
                    continue
                
                # Skip if coordinates invalid
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
                    # progress.advance(task_systems)
                    continue
                    
                if start_seq_id != end_seq_id:
                    self._log_error(
                        "SEQUENCE_SPAN_ERROR", "System spans multiple sequences", 
                        strain_id=strain_id, system_id=sys_id, files=system_files,
                        exception={"start_seq_id": start_seq_id, "end_seq_id": end_seq_id}
                    )
                    self.console.print(f"[bold yellow]{'Warning':>12}[/] System {sys_id} spans multiple sequences")
                    # progress.advance(task_systems)
                    continue
                    
                seq_id = start_seq_id
                
                # Valid coordinates - swap if needed
                if system_start > system_end:
                    system_start, system_end = system_end, system_start
                    
                # Raise warning for suspiciously large regions
                region_size = system_end - system_start + 1
                if region_size > 1e4:
                    self._log_error(
                        "LARGE_REGION_WARNING", f"System region too large: {region_size} bp", 
                        strain_id=strain_id, system_id=sys_id, files=system_files
                    )
                    self.console.print(f"[bold yellow]{'Warning':>12}[/] System {sys_id} region too large: {region_size} bp.")
                
                # Extract sequence
                if seq_id in genome_dict:
                    try:
                        # Get sequence using system_start and system_end
                        genome_seq = genome_dict[seq_id].seq
                        sequence = genome_seq[max(0, system_start-1):min(len(genome_seq), system_end)]
                        sequence_length = len(sequence)
                        
                        # Store sequence data in results
                        results[sys_id] = {
                            "sequence": str(sequence),
                            "length": sequence_length,
                            "seq_id": seq_id,
                            "start": system_start,
                            "end": system_end,
                            "type": system['type'],
                            "subtype": system['subtype'] if 'subtype' in system and pd.notna(system['subtype']) else None,
                            "strain_id": strain_id
                        }
                        
                        # Write to file if requested
                        if self.write_output and output_dir:
                            output_file = os.path.join(str(output_dir), f"{sys_id}.fasta")
                            with open(output_file, "w") as out:
                                out.write(f">{sys_id} Type:{system['type']} Subtype:{system['subtype'] if 'subtype' in system and pd.notna(system['subtype']) else 'NA'} Length:{sequence_length}bp\n")
                                out.write(str(sequence) + "\n")
                            results[sys_id]["file_path"] = output_file
                            
                        # progress.update(task_systems, description=f"Processing: {sys_id} ({sequence_length} bp)")
                        self.console.print(f"[bold blue]{'Extracted':>12}[/] {sys_id} sequence ({sequence_length} bp)")
                        
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

            self.console.print(f"[bold green]{'Extracted':>12}[/] {len(results)} defense systems")
            
        except Exception as e:
            self._log_error(
                "EXTRACTION_FATAL_ERROR", "Uncaught exception in extraction process", 
                strain_id=strain_id, files=files_dict, exception=e
            )
            self.console.print(f"[bold red]Fatal Error:[/] {str(e)}")
            
        finally:
            # Clean up temp database
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
        
    def extract_gene_sequences(
        self,
        systems_tsv_file: pathlib.Path, 
        genes_tsv_file: pathlib.Path, 
        faa_file: pathlib.Path, 
        fna_file: pathlib.Path,
        output_dir: Optional[pathlib.Path] = None,
        strain_id: Optional[str] = None,
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
        # Create output directory if needed and writing is enabled
        if self.write_output and output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        # Track files for error logging
        files_dict = {
            "systems_tsv": systems_tsv_file,
            "genes_tsv": genes_tsv_file,
            "faa_file": faa_file,
            "fna_file": fna_file
        }
        
        if output_dir:
            files_dict["output_dir"] = output_dir
            
        # Create progress tracking
        using_external_progress = self.progress is not None
        progress = self.progress or rich.progress.Progress(
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TimeElapsedColumn(),
            rich.progress.TimeRemainingColumn(),
            rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
        )
        
        # Dictionary to store results
        results = {}
        
        try:
            if not using_external_progress:
                progress.start()
                
            # Read TSV files
            try:
                systems_df = pd.read_csv(systems_tsv_file, sep='\t')
                genes_df = pd.read_csv(genes_tsv_file, sep='\t')
            except Exception as e:
                self._log_error(
                    "TSV_ERROR", "Failed to read TSV files for gene extraction", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to read TSV files: {str(e)}")
                return {}
                
            # Create FASTA indices
            try:
                faa_index = self._create_fasta_index(faa_file)
                fna_index = self._create_fasta_index(fna_file)
            except Exception as e:
                self._log_error(
                    "INDEX_ERROR", "Failed to create sequence indices", 
                    strain_id=strain_id, files=files_dict, exception=e
                )
                self.console.print(f"[bold red]Error:[/] Failed to create sequence indices: {str(e)}")
                return {}
                
            # Process each system
            # task = progress.add_task(
            #     f"[bold cyan]Extracting[/] gene sequences", 
            #     total=len(systems_df)
            # )
            
            for _, system in systems_df.iterrows():
                sys_id = system['sys_id']
                # progress.update(task, description=f"[bold cyan]Extracting[/] genes for: {sys_id}")
                
                # Process system
                result = self._extract_sequences_for_system(
                    sys_id, genes_df, faa_index, fna_index, output_dir, strain_id, progress
                )
                
                if result:
                    results[sys_id] = result
                    
                # progress.update(task, advance=1)
                
        except Exception as e:
            self._log_error(
                "GENE_EXTRACTION_ERROR", "Failed to extract gene sequences", 
                strain_id=strain_id, files=files_dict, exception=e
            )
            self.console.print(f"[bold red]Error:[/] Failed to extract gene sequences: {str(e)}")
            
        finally:
            if not using_external_progress:
                progress.stop()
                
        return results
    
    def _extract_sequences_for_system(
        self,
        system_id: str, 
        genes_df: pd.DataFrame, 
        faa_index: Any, 
        fna_index: Any, 
        output_dir: Optional[pathlib.Path],
        strain_id: Optional[str],
        progress: Optional[rich.progress.Progress] = None
    ) -> Dict[str, Any]:
        """Extract gene sequences for a single defense system"""
        # Prepare file paths for error logging
        files_dict = {"system_id": system_id}
        if output_dir:
            files_dict.update({
                "output_dir": output_dir,
                "faa_output": os.path.join(output_dir, f"{system_id}.faa") if self.write_output else None,
                "fna_output": os.path.join(output_dir, f"{system_id}.fna") if self.write_output else None
            })
        
        # Filter genes for this system
        system_genes = genes_df[genes_df['sys_id'] == system_id]
        
        if system_genes.empty:
            self._log_error(
                "NO_SYSTEM_GENES_ERROR", f"No genes found for system {system_id}", 
                strain_id=strain_id, system_id=system_id, files=files_dict
            )
            if progress:
                progress.console.print(f"[bold yellow]Warning:[/] No genes found for system {system_id}")
            return {}
        
        # Get gene IDs
        hit_ids = set(system_genes['hit_id'].unique())
        
        # Prepare output file paths
        faa_out = os.path.join(str(output_dir), f"{system_id}.faa") if self.write_output and output_dir else None
        fna_out = os.path.join(str(output_dir), f"{system_id}.fna") if self.write_output and output_dir else None
        
        result = {
            "system_id": system_id,
            "proteins": {},
            "nucleotides": {}
        }
        
        try:
            # Find matching sequences
            faa_matching = [
                seq_id for seq_id in faa_index if any(hit_id in seq_id for hit_id in hit_ids)
            ]
            
            fna_matching = [
                seq_id for seq_id in fna_index if any(hit_id in seq_id for hit_id in hit_ids)
            ]
            
            # Process protein sequences
            faa_count = 0
            faa_buffer = io.StringIO() if not self.write_output else None
            
            try:
                # Open file for writing if needed
                out_handle = open(faa_out, "w") if faa_out else faa_buffer
                
                for seq_id in faa_matching:
                    # Get original record
                    record = faa_index[seq_id]
                    
                    # Extract protein ID from sequence ID
                    protein_id = self._extract_protein_id(seq_id, hit_ids)
                    
                    # Create modified ID with system
                    modified_id = f"{system_id}_{protein_id}"
                    modified_desc = f"{modified_id} [system={system_id}]"
                    
                    # Store in results
                    result["proteins"][protein_id] = {
                        "sequence": str(record.seq),
                        "length": len(record.seq),
                        "system_id": system_id,
                        "id": modified_id
                    }
                    
                    # Write to file/buffer
                    self._write_fasta(out_handle, modified_id, modified_desc, str(record.seq))
                    faa_count += 1
                    
                # Close file if it was opened
                if faa_out:
                    out_handle.close()
                else:
                    # Store buffer content in result
                    result["protein_fasta"] = faa_buffer.getvalue()
                    faa_buffer.close()
                    
            except Exception as e:
                self._log_error(
                    "PROTEIN_WRITE_ERROR", f"Failed to process protein sequences for system {system_id}", 
                    strain_id=strain_id, system_id=system_id, files=files_dict, exception=e
                )
                if progress:
                    progress.console.print(f"[bold red]Error:[/] Failed to process protein sequences for {system_id}: {str(e)}")
            
            # Process nucleotide sequences
            fna_count = 0
            fna_buffer = io.StringIO() if not self.write_output else None
            
            try:
                # Open file for writing if needed
                out_handle = open(fna_out, "w") if fna_out else fna_buffer
                
                for seq_id in fna_matching:
                    # Get original record
                    record = fna_index[seq_id]
                    
                    # Extract gene ID from sequence ID
                    gene_id = self._extract_protein_id(seq_id, hit_ids)
                    
                    # Create modified ID with system
                    modified_id = f"{system_id}_{gene_id}"
                    modified_desc = f"{modified_id} [system={system_id}]"
                    
                    # Store in results
                    result["nucleotides"][gene_id] = {
                        "sequence": str(record.seq),
                        "length": len(record.seq),
                        "system_id": system_id,
                        "id": modified_id
                    }
                    
                    # Write to file/buffer
                    self._write_fasta(out_handle, modified_id, modified_desc, str(record.seq))
                    fna_count += 1
                    
                # Close file if it was opened
                if fna_out:
                    out_handle.close()
                else:
                    # Store buffer content in result
                    result["nucleotide_fasta"] = fna_buffer.getvalue()
                    fna_buffer.close()
                    
            except Exception as e:
                self._log_error(
                    "NUCLEOTIDE_WRITE_ERROR", f"Failed to process nucleotide sequences for system {system_id}", 
                    strain_id=strain_id, system_id=system_id, files=files_dict, exception=e
                )
                if progress:
                    progress.console.print(f"[bold red]Error:[/] Failed to process nucleotide sequences for {system_id}: {str(e)}")
            
            # Add file paths to result if writing was enabled
            if self.write_output:
                if faa_out:
                    result["faa_file"] = faa_out
                if fna_out:
                    result["fna_file"] = fna_out
            
            # Report success
            if progress:
                progress.console.print(f"[bold green]Extracted[/] {faa_count} proteins and {fna_count} nucleotides for {system_id}")
            
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
        # Try direct exact ID match
        try:
            feature = db[gene_id]
            return (feature.start, feature.end, feature.seqid) if feature.featuretype in ['gene', 'CDS', 'mRNA'] else (None, None, None)
        except Exception:
            pass
        
        # Try searching for ID in attributes
        for feature_type in ['gene', 'CDS', 'mRNA']:
            try:
                features = list(db.features_of_type(feature_type))
                for feature in features:
                    for key in ['ID', 'Name', 'locus_tag', 'gene', 'product']:
                        if key not in feature.attributes:
                            continue
                        
                        for value in feature.attributes[key]:
                            if (value == gene_id or 
                                f"_{gene_id}_" in value or 
                                value.startswith(f"{gene_id}_") or 
                                value.endswith(f"_{gene_id}")):
                                return feature.start, feature.end, feature.seqid
            except Exception:
                continue
        
        # If no match found
        return None, None, None
    
    def _create_fasta_index(self, fasta_file, reindex=False):
        """Create an index for a FASTA file"""
        index_file = f"{fasta_file}.idx"
        if reindex or not os.path.exists(index_file):
            return SeqIO.index_db(index_file, fasta_file, "fasta")
        return SeqIO.index_db(index_file)
    
    def _extract_protein_id(self, seq_id, hit_ids):
        """Extract the protein ID from a sequence ID using the hit IDs for matching"""
        # First try direct match
        if seq_id in hit_ids:
            return seq_id
            
        # Try to extract from sequence ID
        for hit_id in hit_ids:
            if hit_id in seq_id:
                return hit_id
                
        # Default to sequence ID if no match
        return seq_id
    
    def _write_fasta(self, handle, seq_id, description, sequence):
        """Write a sequence in FASTA format"""
        handle.write(f">{description}\n")
        
        # Write sequence in blocks of 60 characters
        for i in range(0, len(sequence), 60):
            handle.write(sequence[i:i+60] + "\n")
            
        return None
    
    def _log_error(self, error_type, message, strain_id=None, system_id=None, files=None, exception=None):
        """Log error information"""
        import time
        import traceback
        
        # Skip logging if no output directory
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