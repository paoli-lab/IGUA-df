from fileinput import filename
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
from pyfaidx import Fasta 
from typing import Dict, List, Tuple, Optional, Union, Any, Container, Iterable, TextIO
import traceback
import tempfile


import rich.progress
from rich.console import Console


# maybe commit at this stage after cleaning up

# [ ] logging is missing
# [ ] implement cleaning up temp files and resources
# [ ] benchmark against old igua 

# gene IDs - protein IDs are still forced to be "__" 
# because using "_" as in original IGUA would break downstream processing, 
# since system IDs and gene IDs are separated by "_" 




class DefenseSystemDatabaseCollection(): 
    """
    Store defense systems metadata and enable extractor functionality.
    """

    def __init__(
        self,
        systems_tsv_file: pathlib.Path,
        genes_tsv_file: pathlib.Path,
        genomic_fasta_file: pathlib.Path,
        gff_file: pathlib.Path,
        protein_fasta_file: pathlib.Path,
        strain_id: Optional[str] = None,
        activity_filter: Optional[str] = "all",
        progress: Optional[rich.progress.Progress] = None,
        destination_file: Optional[pathlib.Path] = None,
    ):
        if strain_id is None:
            strain_id = str(uuid.uuid4())[:8]
        self.missing_files = []
        for f, name in [
            (systems_tsv_file, "systems_tsv"),
            (genes_tsv_file, "genes_tsv"),
            (gff_file, "gff_file"),
            (genomic_fasta_file, "fasta_file"),
            (protein_fasta_file, "faa_file"),
        ]:
            if not f.exists():
                self.missing_files.append(f"{name}: {f}")            
                # raise FileNotFoundError(f"One or more input files do not exist for strain {strain_id}")

        self.strain_id: str = strain_id
        self.systems_tsv_file: pathlib.Path = systems_tsv_file
        self.genes_tsv_file: pathlib.Path = genes_tsv_file
        self.genomic_fasta_file: pathlib.Path = genomic_fasta_file
        self.gff_file: pathlib.Path = gff_file
        self.protein_fasta_file: pathlib.Path = protein_fasta_file
        self.activity_filter: Optional[str] = activity_filter

        self.console: rich.console.Console = progress.console if progress else Console()
        self.dst: Optional[pathlib.Path] = destination_file

        self.systems_df: Optional[pd.DataFrame] = None
        self.genes_df: Optional[pd.DataFrame] = None
        self.genome_idx: Optional[Any] = None
        self.gff_database: Optional[Any] = None
        self.gff_database_path: Optional[str] = None
        return None

    def _load_and_filter_systems(
        self,
        strain_id: Optional[str],
        representatives: Optional[Iterable[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """Load and filter TSV files"""
        systems_df = pd.read_csv(self.systems_tsv_file, sep="\t")

        # filter systems by activity if applicable
        original_count = len(systems_df)
        if self.activity_filter.lower() != "all":
            if "activity" in systems_df.columns:
                systems_df = systems_df[
                    systems_df["activity"].str.lower() == self.activity_filter.lower()
                ]

        # check for duplicate systems after activity filtering
        duplicate_mask = systems_df.duplicated(keep="first")
        if duplicate_mask.any():
            duplicate_systems = systems_df[duplicate_mask]["sys_id"].tolist()
            n_duplicates = len(duplicate_systems)

            # log warning about duplicates
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] {n_duplicates} duplicate system/s in strain [bold cyan]{self.strain_id}[/]: [cyan]{', '.join(duplicate_systems[:5])}{'...' if n_duplicates > 5 else ''}[/]"
            )
            self._log_error(
                "DUPLICATE_SYSTEMS_WARNING",
                f"Found {n_duplicates} duplicate system IDs in systems TSV",
                strain_id=self.strain_id,
                exception={
                    "duplicate_sys_ids": duplicate_systems,
                    "systems_tsv_file": str(self.systems_tsv_file),
                    "total_systems": len(systems_df),
                    "unique_systems": len(systems_df["sys_id"].unique()),
                },
            )
            # keep first occurrence
            systems_df = systems_df.drop_duplicates(keep="first")
        
        # representatives filtering (if applicable)
        if representatives is not None:
            try:
                # try iterable first
                systems_df = systems_df[systems_df['sys_id'].isin(representatives)]
            except TypeError:
                # __contains__ method for non-iterable containers
                mask = systems_df['sys_id'].apply(lambda x: x in representatives)
                systems_df = systems_df[mask]

            # filtered_count = len(systems_df)
            # if self.verbose and self.progress:
            #     self.progress.console.print(
            #         f"[bold cyan]{'Filtered':>12}[/] {original_count} systems to {filtered_count} "
            #         f"representatives for {strain_id}"
            #     )

        self.systems_df = systems_df

        # log filtering details
        #### this level of logging is excessive and will be reduced in future
        if self.activity_filter.lower() != "all":
            if "activity" in systems_df.columns:
                self.console.print(
                    f"[bold green]{'Filtered':>12}[/] {original_count} systems to {len(systems_df)} "
                    f"([bold cyan]{self.activity_filter}[/] systems only)"
                )
            else:
                self.console.print(
                    f"[bold yellow]{'Warning':>12}[/] No 'activity' column found, extracting all systems"
                )
        else:
            self.console.print(
                f"[bold blue]{'Processing':>12}[/] all {original_count} systems (no activity filter)"
            )
        return None

    def _load_system_genes(self) -> Optional[pd.DataFrame]:
        """Load system genes TSV file"""
        self.genes_df = pd.read_csv(self.genes_tsv_file, sep="\t")
        return None

    def _load_genome(self):
        """Load genomic FASTA file. 
        Does not actually load sequences into memory; 
        instead using pyfaidx and writes a *.fai index file."""
        self.genome_idx = Fasta(self.genomic_fasta_file)
        return None


    def _setup_gff_database(
        self, 
        unique_id: str,
        gff_cache_dir: Optional[pathlib.Path] = None,
    ):
        """Set up GFF database with in-memory fallback"""
        db_path = ":memory:"

        try:
            if gff_cache_dir:
                os.makedirs(gff_cache_dir, exist_ok=True)
                db_path = os.path.join(str(gff_cache_dir), f"{os.path.basename(str(self.gff_file))}_{unique_id}.db")
            else:
                db_path = os.path.join(tempfile.gettempdir(), f"gff_temp_{unique_id}.db")

            db = gffutils.create_db(
                str(self.gff_file),  
                dbfn=db_path, 
                force=True, 
                merge_strategy='create_unique',
                    # widen id_spec to include more attributes for robust matching
                id_spec=['ID', 'Name', 'gbkey', 'gene', 'gene_biotype', 'locus_tag', 'old_locus_tag']
            )
            self.gff_database = db
            self.gff_database_path = db_path
            return None

        except Exception as e:
            self.console.print(
                f"[bold red]Error:[/] Failed to create GFF database: {str(e)}"
            )
            self.console.print("Using", "in-memory database")
            db_path = ":memory:"

            try:
                db = gffutils.create_db(
                    str(self.gff_file),
                    dbfn=":memory:", 
                    merge_strategy='create_unique',
                        # widen id_spec to include more attributes for robust matching
                    id_spec=['ID', 'Name', 'gbkey', 'gene', 'gene_biotype', 'locus_tag', 'old_locus_tag']
                )
                self.gff_database = db
                self.gff_database_path = db_path
                return None

            except Exception as e2:
                self._log_error(
                    "GFF_FATAL_ERROR", "Failed to create in-memory GFF database", 
                    strain_id=strain_id, files_dict=files_dict, exception=e2
                )
                self.console.print(f"[bold red]Error:[/] Failed to create GFF database: {str(e2)}")
                return None

    def process_and_extract_systems(self):
        """Process and extract defense systems"""
        results = []
        # process each system
        systems_processed = 0
        for _, system in self.systems_df.iterrows():
            single_ds_db = DefenseSystemDatabase(parent=self, system_tsv_row=system)
            single_ds_db.get_system_boundaries()
            single_ds_db.validate_system_boundaries()
            single_system_genomic_seq = single_ds_db.extract_genomic_sequence()

            if single_system_genomic_seq:
                results.append(single_system_genomic_seq)

            systems_processed += 1

            # Periodic garbage collection
            if systems_processed % 10 == 0:
                gc.collect()

        self.console.print(f"[bold green]{'Extracted':>12}[/] {len(results)} defense systems for [bold cyan]{self.strain_id}[/]")
        return results


    def _load_protein_fasta(self):
        """Load protein FASTA file using pyfaidx"""
        self.protein_fasta_idx = Fasta(self.protein_fasta_file)
        return None


    def process_and_extract_protein_sequences(self):
        """Process and extract protein sequences for defense systems."""
        protein_sizes = {}
        # process each system
        systems_processed = 0
        for _, system in self.systems_df.iterrows():
            single_ds_db = DefenseSystemDatabase(parent=self, system_tsv_row=system)
            single_ds_db.get_system_genes()
            single_system_protein_seq = single_ds_db.extract_protein_sequences()

            if single_system_protein_seq:
                protein_sizes.update(single_system_protein_seq)

            systems_processed += 1
        self.console.print(f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} defense systems for [bold cyan]{self.strain_id}[/]")
        return protein_sizes



    def _log_error(self, error_type, message, strain_id=None, system_id=None, files_dict=None, exception=None):    
        """Log error information"""
        log_file = "defense_extraction_errors.log"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {error_type}: {message}\n"

        if strain_id:
            log_message += f"  Strain: {strain_id}\n"
        if system_id:
            log_message += f"  System: {system_id}\n"
        if files_dict:
            log_message += "  Files:\n"
            for key, path in files_dict.items():
                if path:
                    log_message += f"    {key}: {path}\n"
        if exception:
            log_message += f"  Exception: {str(exception)}\n"

        log_message += "-" * 80 + "\n"

        with open(log_file, "a") as f:
            f.write(log_message)


    def _cleanup(self):
        """Cleanup temporary files and resources"""
        # if systems_df is not None:
        #     del systems_df
        # if genes_df is not None:
        #     del genes_df  
        # if genome_dict is not None:
        #     del genome_dict
        # if db is not None:
        #     del db

        # # clean up temp database
        # if db_path != ":memory:" and os.path.exists(db_path) and unique_id in str(db_path):
        #     try:
        #         os.unlink(db_path)
        #     except Exception as e:
        #         self._log_error(
        #             "CLEANUP_ERROR", "Failed to clean up temporary database", 
        #             strain_id=strain_id, files_dict={"db_path": db_path}, exception=e
        #         )

        # if not using_external_progress:
        #     progress.stop()

        # gc.collect()
        pass

class DefenseSystemDatabase:
    """
    Represents a single defense system and provides extraction methods.
    Receives reference to parent DefenseSystemDatabaseCollection to access shared resources.
    """

    def __init__(self, parent: DefenseSystemDatabaseCollection, system_tsv_row: pd.Series):
        # store reference to parent for accessing shared resources
        self.parent = parent
        
        # system-specific attributes
        self.sys_id: str = system_tsv_row['sys_id']
        self.sys_beg_hit_id: str = system_tsv_row['sys_beg']
        self.sys_end_hit_id: str = system_tsv_row['sys_end']
        # self.sys_type: str = system_tsv_row['type']
        # self.sys_subtype: str = system_tsv_row['subtype']
        self.proteins_in_system_hit_refs: list[str] = system_tsv_row['protein_in_syst'].split(",")  # locus IDs of proteins in the system
        assert len(self.proteins_in_system_hit_refs) == system_tsv_row['genes_count']
        self.sys_genes_df: Optional[pd.DataFrame] = None 
        self.sys_beg_coord: Optional[int] = None  # start coordinate of the beginning gene
        self.sys_end_coord: Optional[int] = None  # end coordinate of the ending gene
    
    # property accessors to access parent's shared resources
    @property
    def strain_id(self):
        return self.parent.strain_id
    
    @property
    def genes_df(self):
        return self.parent.genes_df
    
    @property
    def genome_idx(self):
        return self.parent.genome_idx
    
    @property
    def gff_database(self):
        return self.parent.gff_database
    
    @property
    def console(self):
        return self.parent.console
    
    @property
    def dst(self):
        return self.parent.dst
    
    @property
    def genomic_fasta_file(self):
        return self.parent.genomic_fasta_file
        
    @property
    def protein_fasta_idx(self):
        return self.parent.protein_fasta_idx

    def _log_error(self, *args, **kwargs):
        """Delegate error logging to parent"""
        return self.parent._log_error(*args, **kwargs)

    def get_system_genes(self):
        #### this is going to be for the protein sequence extraction 
        self.sys_genes_df = self.genes_df[self.genes_df['sys_id'] == self.sys_id].sort_values('hit_pos')
        self.sys_genes_hit_ids = self.sys_genes_df['hit_id'].tolist()
        # if system_genes.empty:
        #     self._log_error(
        #         "NO_SYSTEM_GENES_ERROR", f"No genes found for system {system_id}", 
        #         strain_id=strain_id, system_id=system_id, files_dict=files_dict
        #     )
        #     if progress:
        #         progress.console.print(f"[bold yellow]Warning:[/] No genes found for system {system_id}")
        #     return {}

        # if self.sys_genes_df.empty:
        #     self._handle_error(
        #         "NO_GENES_ERROR", "No genes found for system",
        #         strain_id=self.strain_id, system_id=self.sys_id, files_dict=system_files
        #     )
        #     self._print_warning(f"No genes found for system {self.sys_id}")
        #     return None

        # # beginning and ending gene
        # beg_gene_mask = self.proteins_in_system_hit_refs['hit_id'] == self.sys_beg
        # end_gene_mask = self.proteins_in_system_hit_refs['hit_id'] == self.sys_end
        # if not beg_gene_mask.any() or not end_gene_mask.any():
        #     self._log_error(
        #         "GENE_NOT_FOUND", f"Beginning or ending gene not found in system genes",
        #         strain_id=self.strain_id, system_id=self.sys_id,
        #         exception={
        #             "sys_beg": self.sys_beg,
        #             "sys_end": self.sys_end,
        #             "available_hit_ids": list(self.proteins_in_system_hit_refs['hit_id'])
        #         }
        #     )
        #     self.console.print(f"[bold red]Error:[/] Could not find beginning or ending gene for system {self.sys_id}")
        #     return None

        # except Exception as e:
        #     self._log_error(
        #         "GENE_DATA_ERROR", "Failed to extract gene information", 
        #         strain_id=strain_id, system_id=sys_id, exception=e
        #     )
        #     self.console.print(f"[bold red]Error:[/] Failed to get gene data for system {sys_id}: {str(e)}")
        #     return None
        return None

    def get_system_boundaries(self):
        """Get start and end coordinates of the defense system"""
        # get genomic coordinates from GFF database
        start_gene_start_coord, start_gene_end_coord, start_gene_seq_id = self._find_gene_coordinates(self.sys_beg_hit_id)
        self.start_gene_found = {'start_coord': start_gene_start_coord, 'end_coord': start_gene_end_coord, 'seq_id': start_gene_seq_id}
        end_gene_start_coord, end_gene_end_coord, end_gene_seq_id = self._find_gene_coordinates(self.sys_end_hit_id)
        self.end_gene_found = {'start_coord': end_gene_start_coord, 'end_coord': end_gene_end_coord, 'seq_id': end_gene_seq_id}
        return None

    def _find_gene_coordinates(self, hit_id: str) -> Tuple[int, int, str]:
        """Find gene coordinates in the GFF database"""
        feature = self.gff_database[hit_id]
        return feature.start, feature.end, feature.seqid

    def validate_system_boundaries(self):
        # validate coordinates
        if any(f is None for f in self.start_gene_found.values()) or any(
            f is None for f in self.end_gene_found.values()
        ):
            self._log_error(
                "COORD_ERROR",
                "Missing coordinates or sequence ID for system",
                strain_id=self.strain_id,
                system_id=self.sys_id,
                exception={
                    "system_start": self.start_gene_found['start_coord'],
                    "system_end": self.start_gene_found['end_coord'],
                    "start_seq_id": self.start_gene_found['seq_id'],
                    "end_seq_id": self.end_gene_found['seq_id'],
                    "beg_hit_id": self.sys_beg_hit_id,
                    "end_hit_id": self.sys_end_hit_id,
                },
            )
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] Missing coordinates for system {self.sys_id}"
            )
            return None

        if self.start_gene_found['seq_id'] != self.end_gene_found['seq_id']:
            self._log_error(
                "SEQUENCE_SPAN_ERROR",
                "System spans multiple sequences",
                strain_id=self.strain_id,
                system_id=self.sys_id,
                exception={"start_seq_id": self.start_gene_found['seq_id'], "end_seq_id": self.end_gene_found['seq_id']},
            )
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] system {self.sys_id} spans multiple sequences"
            )
            return None


        self.system_genome_start_coord = self.start_gene_found['start_coord']
        self.system_genome_end_coord = self.end_gene_found['end_coord']
        self.system_genome_seq_id = self.start_gene_found['seq_id']

        if self.system_genome_seq_id not in self.genome_idx:
            self._log_error(
                "SEQUENCE_ID_ERROR",
                f"Sequence ID {self.system_genome_seq_id} not found in genome",
                strain_id=self.strain_id,
                system_id=self.sys_id,
            )
            self.console.print(
                f"[bold red]{'Error':>12}[/] Sequence {self.system_genome_seq_id} not found in genome"
            )

        # raise warning for suspiciously large regions >1e4
        region_size = self.system_genome_end_coord - self.system_genome_start_coord + 1
        if region_size > 1e4:
            self._log_error(
                "LARGE_REGION_WARNING",
                f"System region too large: {region_size} bp",
                strain_id=self.strain_id,
                system_id=self.sys_id,
            )
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] system [cyan]{self.sys_id}[/] region too large: {region_size} bp."
            )
        # raise warning for suspiciously large regions <1e2
        if region_size < 1e2:
            self._log_error(
                "SMALL_REGION_WARNING",
                f"System region too small: {region_size} bp",
                strain_id=self.strain_id,
                system_id=self.sys_id,
            )
            self.console.print(
                f"[bold yellow]{'Warning':>12}[/] system [cyan]{self.sys_id}[/] region too small: {region_size} bp."
            )
        return None



    def extract_genomic_sequence(self):
        """Extract genomic sequence for the defense system"""
        try:
            genome_seq = self.genome_idx[self.system_genome_seq_id]
            sequence = genome_seq[self.system_genome_start_coord-1:self.system_genome_end_coord]
            sequence_length = len(sequence)

            result = [
                self.sys_id, # cluster_id
                sequence_length, # cluster_length
                str(self.genomic_fasta_file) # filename
            ]
            self.write_fasta(self.dst, self.sys_id, str(sequence))
            # data.append((self.sys_id, sequence_length, str(self.genomic_fasta_file)))
            # done.add(self.sys_id)
            # if self.verbose: 
            #     self.console.print(f"[bold blue]{'Extracted':>22}[/] genomic sequence for [cyan]{sys_id}[/] system ({sequence_length} bp)")

            del sequence, genome_seq
            return result

        except Exception as e:
            # self._log_error(
            #     "SEQUENCE_EXTRACTION_ERROR", "Failed to extract sequence", 
            #     strain_id=strain_id, system_id=sys_id, exception=e
            # )
            # self.console.print(f"[bold red]Error:[/] Failed to extract sequence for {sys_id}: {str(e)}")
            return None


    def extract_protein_sequences(self):
        """Extract protein sequence for the defense system"""
        try: 
            result = {}
            count = 0
            for seq_id in self.proteins_in_system_hit_refs:
                sequence = self.protein_fasta_idx[seq_id]
                sequence_length = len(sequence)
                protein_id = "{}__{}".format(self.sys_id, seq_id)

                result[protein_id] = sequence_length

                self.write_fasta(self.dst, protein_id, str(sequence))
                count += 1
                del sequence
            return result
        except Exception as e:
            # self._log_error(
            #     "SEQUENCE_EXTRACTION_ERROR", "Failed to extract sequence", 
            #     strain_id=strain_id, system_id=sys_id, exception=e
            # )
            # self.console.print(f"[bold red]Error:[/] Failed to extract sequence for {sys_id}: {str(e)}")
            return None


    def write_fasta(self, file: TextIO, name: str, sequence: str) -> None:
        file.write(">{}\n".format(name))
        file.write(sequence)
        file.write("\n")
        return None



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
        verbose: bool = False,
    ):
        self.progress = progress
        self.output_base_dir = output_base_dir
        self.console = progress.console if progress else Console()
        self.verbose = verbose



    def extract_systems(
        self,
        defense_system_database: DefenseSystemDatabaseCollection,
        strain_id: Optional[str] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract defense systems' sequences from genomic data
        
        Args:
            systems_tsv_file: Path to DefenseFinder systems TSV
            genes_tsv_file: Path to DefenseFinder genes TSV
            gff_file: Path to GFF annotation file
            fasta_file: Path to genome FASTA file
            gff_cache_dir: Directory to cache GFF databases
            strain_id: Optional strain identifier
            
        Returns:
            Dictionary mapping system_id to sequence data and metadata
        """
        unique_id = str(uuid.uuid4())

        using_external_progress = self.progress is not None
        progress = self.progress or rich.progress.Progress(
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
        )

        try:
            if not using_external_progress:
                progress.start()

            defense_system_database._load_and_filter_systems(strain_id=strain_id)
            defense_system_database._load_system_genes() 
            defense_system_database._load_genome()
            defense_system_database._setup_gff_database(unique_id=unique_id)
            systems = defense_system_database.process_and_extract_systems()

        except Exception as e:
            self._log_error(
                "EXTRACTION_FATAL_ERROR", "Uncaught exception in extraction process", 
                strain_id=strain_id, exception=e
            )
            self.console.print(f"[bold red]Fatal Error:[/] {str(e)}")
            return {}

        finally:
            defense_system_database._cleanup()
            # cleanup


        return systems

    def extract_protein_representatives_sequences(
        self,
        defense_system_database: DefenseSystemDatabaseCollection,
        strain_id: str,
        representatives: Optional[Iterable[str]] = None,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Extract gene sequences (protein sequence) for defense systems
        
        Args:
            systems_tsv_file: Path to DefenseFinder systems TSV
            genes_tsv_file: Path to DefenseFinder genes TSV
            faa_file: Path to protein FASTA file
            strain_id: Optional strain identifier
            activity_filter: Filter systems by activity type (default: "defense")
            representatives: Optional set of representative cluster IDs to extract (for efficiency)
            
        Returns:
            Dictionary mapping system_id to gene sequence data
        """
        using_external_progress = self.progress is not None
        progress = self.progress or rich.progress.Progress(
            rich.progress.SpinnerColumn(),
            rich.progress.TextColumn("[bold blue]{task.description}"),
            rich.progress.BarColumn(),
            rich.progress.TimeElapsedColumn(),
            rich.progress.TimeRemainingColumn(),
            rich.progress.TextColumn("[bold]{task.completed}/{task.total}")
        )
        try:
            if not using_external_progress:
                progress.start()

            defense_system_database._load_and_filter_systems(strain_id=strain_id, representatives=representatives)
            defense_system_database._load_system_genes() 
            defense_system_database._load_protein_fasta()
            protein_sizes_for_system = defense_system_database.process_and_extract_protein_sequences()


        except Exception as e:
            self._log_error(
                "GENE_EXTRACTION_ERROR", "Failed to extract gene sequences", 
                strain_id=strain_id, exception=e
            )
            self.console.print(f"[bold red]Error:[/] Failed to extract gene sequences: {str(e)}")

        finally:
            if not using_external_progress:
                progress.stop()

        total_proteins = len(protein_sizes_for_system.values())
        progress.console.print(f"[bold green]{'Extracted':>12}[/] {total_proteins} proteins from {len(protein_sizes_for_system)} systems for [bold cyan]{strain_id}[/]")

        return protein_sizes_for_system



    def _log_extraction_results(self, protein_count: int, system_id: str, progress: rich.progress.Progress):
        """Log the results of sequence extraction"""
        if protein_count > 0:
            progress.console.print(f"[bold green]{'Extracted':>22}[/] {protein_count} proteins from [cyan]{system_id}[/] system")
        else:
            progress.console.print(f"[bold yellow]{'Warning':>22}[/] No sequences extracted for [cyan]{system_id}[/] system")



    def _log_error(self, error_type, message, strain_id=None, system_id=None, files_dict=None, exception=None):
        """Log error information"""
        log_file = "defense_extraction_errors.log"

        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {error_type}: {message}\n"

        if strain_id:
            log_message += f"  Strain: {strain_id}\n"
        if system_id:
            log_message += f"  System: {system_id}\n"
        if files_dict:
            log_message += "  Files:\n"
            for key, path in files_dict.items():
                if path:
                    log_message += f"    {key}: {path}\n"
        if exception:
            log_message += f"  Exception: {str(exception)}\n"

        log_message += "-" * 80 + "\n"

        with open(log_file, "a") as f:
            f.write(log_message)
