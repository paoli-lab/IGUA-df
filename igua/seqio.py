import abc
import io
import typing
import pathlib
import tempfile
import gzip
import warnings

import Bio.Seq
import gb_io
import pandas
import rich.progress

from .mmseqs import MMSeqs
from .mmseqs import Database
from .defense_extractor import DefenseExtractor



_GZIP_MAGIC = b'\x1f\x8b'


class BaseDataset(abc.ABC):
    """Base class for dataset extraction.
    This class defines the basic structure and methods for extracting nucleotide and
    protein sequences from various file formats. It serves as a base class for specific 
    dataset classes like GenBankDataset and GFFDataset.
    """
    @abc.abstractmethod
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        ) -> pandas.DataFrame:
        pass
    
    @abc.abstractmethod
    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
        ) -> typing.Dict[str, int]:
        pass
    
    def write_fasta(self, file: typing.TextIO, name: str, sequence: str) -> None:
        file.write(">{}\n".format(name))
        file.write(sequence)
        file.write("\n")
        return None

    def translate_orf(self, sequence: typing.Union[str, bytes], translation_table: int = 11) -> str:
        return str(Bio.Seq.Seq(sequence).translate(translation_table))

    
    def create_sequence_database(
        self, 
        mmseqs: MMSeqs, 
        progress: rich.progress.Progress, 
        inputs: typing.List[pathlib.Path],
        output_db_path: pathlib.Path
    ) -> Database:
        """Default implementation creates a temporary file then a database"""
        # Create temporary FASTA file
        tmp_fasta = output_db_path.with_suffix(".fna")
        self.extract_sequences(progress, inputs, tmp_fasta)
        # Create database from temporary file
        return Database.create(mmseqs, tmp_fasta)
    
    def create_protein_database(
        self, 
        mmseqs: MMSeqs,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        representatives: typing.Container[str],
        output_db_path: pathlib.Path
    ) -> typing.Tuple[Database, typing.Dict[str, int]]:
        """Default implementation creates a temporary file then a database"""
        # Create temporary FASTA file
        tmp_fasta = output_db_path.with_suffix(".faa")
        protein_sizes = self.extract_proteins(progress, inputs, tmp_fasta, representatives)
        # Create database from temporary file
        return Database.create(mmseqs, tmp_fasta), protein_sizes


class GenBankDataset(BaseDataset):
    """GenBank dataset class.
    This class is used to extract nucleotide and protein sequences from GenBank files.
    It inherits from the BaseDataset class and implements the extract_sequences
    and extract_proteins methods.
    """
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Extracts nucleotide sequences from GenBank files.
        Args:
            progress (rich.progress.Progress): Progress bar for tracking progress.
            inputs (typing.List[pathlib.Path]): List of input GenBank files.
            output (pathlib.Path): Output file path for the extracted sequences. 
        Returns:
            pandas.DataFrame: DataFrame containing the extracted sequences.
        """
        data = []
        done = set()
        n_duplicate = 0
        with open(output, "w") as dst:
            task1 = progress.add_task(f"[bold blue]{'Working':>9}[/]")
            for input_path in progress.track(inputs, task_id=task1):
                task2 = progress.add_task(f"[bold blue]{'Reading':>9}[/]")
                with io.BufferedReader(progress.open(input_path, "rb", task_id=task2)) as reader:  # type: ignore
                    if reader.peek().startswith(_GZIP_MAGIC):
                        reader = gzip.GzipFile(mode="rb", fileobj=reader)  # type: ignore
                    for record in gb_io.iter(reader):
                        if record.name in done:
                            n_duplicate += 1
                        else:
                            self.write_fasta(dst, record.name, record.sequence.decode("ascii"))
                            data.append((record.name, len(record.sequence), input_path))
                            done.add(record.name)
                progress.remove_task(task2)
            progress.remove_task(task1)
        if n_duplicate > 0:
            progress.console.print(
                f"[bold yellow]{'Skipped':>12}[/] {n_duplicate} clusters with duplicate identifiers"
            )
        return pandas.DataFrame(
            data=data,
            columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")

    
    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str],
    ) -> typing.Dict[str, int]:
        """Extracts protein sequences from GenBank files.
        Args:
            progress (rich.progress.Progress): Progress bar for tracking progress.
            inputs (typing.List[pathlib.Path]): List of input GenBank files.
            output (pathlib.Path): Output file path for the extracted protein sequences.
            representatives (typing.Container[str]): Set of representative cluster IDs.
        Returns:
            typing.Dict[str, int]: Dictionary containing protein IDs and their sizes. 
        """
        protein_sizes = {}
        with output.open("w") as dst:
            for input_path in inputs:
                task = progress.add_task(f"[bold blue]{'Reading':>9}[/]")
                with io.BufferedReader(progress.open(input_path, "rb", task_id=task)) as reader:  # type: ignore
                    if reader.peek()[:2] == b'\x1f\x8b':
                        reader = gzip.GzipFile(mode="rb", fileobj=reader)  # type: ignore
                    for record in gb_io.iter(reader):
                        if record.name in representatives:
                            for i, feat in enumerate(
                                filter(lambda f: f.kind == "CDS", record.features)
                            ):
                                qualifier = next((qualifier for qualifier in feat.qualifiers if qualifier.key == "translation"), None)
                                if qualifier is None:
                                    rich.print(f"[bold yellow]{'Warning':>12}[/] no 'translation' qualifier found in CDS feature of {record.name!r}")
                                    translation = self.translate_orf(record.sequence[feat.location.start:feat.location.end])
                                else:
                                    translation = qualifier.value.rstrip("*")
                                protein_id = "{}_{}".format(record.name, i)
                                if protein_id not in protein_sizes:
                                    self.write_fasta(dst, protein_id, translation)
                                    protein_sizes[protein_id] = len(translation)
                progress.remove_task(task)
        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(representatives)} nucleotide representative"
        )
        return protein_sizes



# Update the DefenseFinderDataset class
class DefenseFinderDataset(BaseDataset):
    """DefenseFinder dataset class.
    This class is used to extract nucleotide and protein sequences from DefenseFinder output files.
    """
    def __init__(self):
        """Initialize the DefenseFinderDataset class."""
        self.defense_metadata = None  # pathlib.Path to TSV with paths to defense finder files
        self.defense_systems_tsv = None  # pathlib.Path to DefenseFinder systems TSV
        self.defense_genes_tsv = None  # pathlib.Path to DefenseFinder genes TSV
        self.gff_file = None  # pathlib.Path to GFF file
        self.genome_file = None  # pathlib.Path to genome FASTA file
        self.write_output = False  # Whether to write output files
        self.output_dir = None  # Output directory for files
    
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
    ) -> pandas.DataFrame:
        """Extracts nucleotide sequences from defense system FASTA files.
        
        Args:
            progress: Progress bar for tracking.
            inputs: List of input files (FASTA files or TSV summary).
            output: Output path for combined sequences.
            
        Returns:
            DataFrame with cluster_id, cluster_length, filename
        """
        # Check for direct DefenseFinder integration mode
        if self._check_defense_finder_integration_mode():
            return self._process_with_defense_extractor(progress, output)
        
        # Handle TSV summary file
        if len(inputs) == 1 and inputs[0].suffix.lower() == ".tsv":
            progress.console.print(f"[bold blue]{'Found':>12}[/] TSV summary file")
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                
                # Check if we have all required columns for advanced processing
                required_cols = ["systems_tsv", "genes_tsv", "gff_file", "fasta_file"]
                if all(col in df.columns for col in required_cols):
                    # Use defense_extractor for processing
                    return self._process_defense_files_from_tsv(progress, df, output)
                
                # First check for fasta_file column (genome sequences)
                if "fasta_file" in df.columns:
                    fa_files = [pathlib.Path(f) for f in df["fasta_file"] if f]
                    progress.console.print(f"[bold blue]{'Using':>12}[/] fasta_file column with {len(fa_files)} files")
                    return self._process_fa_files(progress, fa_files, output)
                # Fall back to genomic_file for backward compatibility
                elif "genomic_file" in df.columns:
                    fa_files = [pathlib.Path(f) for f in df["genomic_file"] if f]
                    progress.console.print(f"[bold blue]{'Using':>12}[/] genomic_file column with {len(fa_files)} files")
                    return self._process_fa_files(progress, fa_files, output)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] TSV missing required columns"
                    )
            except Exception as e:
                progress.console.print(f"[bold red]{'Error':>12}[/] reading TSV: {e}")


        # Handle directory with FASTA files
        elif len(inputs) == 1 and inputs[0].is_dir():
            fa_files = list(inputs[0].glob("**/*.fna"))
            progress.console.print(f"[bold blue]{'Found':>12}[/] {len(fa_files)} FASTA files")
            return self._process_fa_files(progress, fa_files, output)
            
        # Handle direct list of FASTA files
        elif all(f.suffix.lower() in (".fa", ".fna", ".fasta") for f in inputs):
            progress.console.print(f"[bold blue]{'Found':>12}[/] {len(inputs)} FASTA files")
            return self._process_fa_files(progress, inputs, output)
            
        # Return empty DataFrame if no valid input
        progress.console.print(f"[bold red]{'Error':>12}[/] No valid input files")
        return pandas.DataFrame(columns=["cluster_id", "cluster_length", "filename"]).set_index("cluster_id")
    
    def _check_defense_finder_integration_mode(self) -> bool:
        """Check if we're in direct DefenseFinder integration mode."""
        # Check if direct DefenseFinder mode is enabled
        return (
            (self.defense_metadata is not None) or
            (self.defense_systems_tsv is not None and 
             self.defense_genes_tsv is not None and 
             self.gff_file is not None and 
             self.genome_file is not None)
        )
    
    def _process_with_defense_extractor(
        self,
        progress: rich.progress.Progress,
        output: pathlib.Path
    ) -> pandas.DataFrame:
        """Process defense systems using the DefenseExtractor."""
        # Create temporary directory for output if needed
        temp_dir = None
        if self.write_output and not self.output_dir:
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = pathlib.Path(temp_dir.name)
        else:
            output_dir = self.output_dir
        
        try:
            # Create extractor
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=output_dir,
                write_output=self.write_output
            )
            
            # Use defense_metadata file if provided
            if self.defense_metadata:
                progress.console.print(f"[bold blue]{'Using':>12}[/] defense metadata file: {self.defense_metadata}")
                try:
                    df = pandas.read_csv(self.defense_metadata, sep="\t")
                    return self._process_defense_files_from_tsv(progress, df, output, extractor)
                except Exception as e:
                    progress.console.print(f"[bold red]{'Error':>12}[/] reading defense metadata: {e}")
                    return pandas.DataFrame(columns=["cluster_id", "cluster_length", "filename"]).set_index("cluster_id")
            
            # Use individual files
            progress.console.print(f"[bold blue]{'Using':>12}[/] direct defense finder files")
            
            # Extract systems
            systems = extractor.extract_systems(
                systems_tsv_file=self.defense_systems_tsv,
                genes_tsv_file=self.defense_genes_tsv,
                gff_file=self.gff_file,
                fasta_file=self.genome_file,
                output_dir=output_dir if self.write_output else None
            )
            
            # Write sequences to output file and create DataFrame
            data = []
            with open(output, "w") as dst:
                for sys_id, system in systems.items():
                    sequence = system["sequence"]
                    length = system["length"]
                    
                    # Write to FASTA file
                    self.write_fasta(dst, sys_id, sequence)
                    
                    # Record for DataFrame
                    file_path = system.get("file_path", str(self.genome_file))
                    data.append((sys_id, length, file_path))
            
            # Create and return DataFrame
            progress.console.print(f"[bold green]{'Extracted':>12}[/] {len(data)} systems")
            return pandas.DataFrame(
                data=data,
                columns=["cluster_id", "cluster_length", "filename"]
            ).set_index("cluster_id")
            
        finally:
            # Clean up temporary directory if created
            if temp_dir:
                temp_dir.cleanup()
    
    def _process_defense_files_from_tsv(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        extractor: typing.Optional[DefenseExtractor] = None
    ) -> pandas.DataFrame:
        """Process defense systems from a TSV file with paths to defense finder files."""
        # Create extractor if not provided
        if extractor is None:
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=self.output_dir,
                write_output=self.write_output
            )
        
        # Process each row in DataFrame
        data = []
        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] defense systems", total=len(df))
            
            for _, row in df.iterrows():
                strain_id = row.get("strain_id", None)
                progress.update(task, description=f"[bold blue]{'Processing':>9}[/] strain: {strain_id}")
                
                # Get required files
                systems_tsv = pathlib.Path(row["systems_tsv"])
                genes_tsv = pathlib.Path(row["genes_tsv"])
                gff_file = pathlib.Path(row["gff_file"])
                fasta_file = pathlib.Path(row["fasta_file"])
                
                # Check if all files exist
                missing_files = []
                for f, name in [(systems_tsv, "systems_tsv"), (genes_tsv, "genes_tsv"), 
                               (gff_file, "gff_file"), (fasta_file, "fasta_file")]:
                    if not f.exists():
                        missing_files.append(f"{name}: {f}")
                
                if missing_files:
                    progress.console.print(f"[bold yellow]{'Missing':>12}[/] files for {strain_id}: {', '.join(missing_files)}")
                    progress.update(task, advance=1)
                    continue
                
                # Extract systems
                strain_output_dir = None
                if self.write_output and self.output_dir:
                    strain_output_dir = self.output_dir / (strain_id if strain_id else "unknown")
                    strain_output_dir.mkdir(parents=True, exist_ok=True)
                
                systems = extractor.extract_systems(
                    systems_tsv_file=systems_tsv,
                    genes_tsv_file=genes_tsv,
                    gff_file=gff_file,
                    fasta_file=fasta_file,
                    output_dir=strain_output_dir,
                    strain_id=strain_id
                )
                
                # Write systems to output FASTA and record data
                for sys_id, system in systems.items():
                    sequence = system["sequence"]
                    length = system["length"]
                    
                    # Add strain ID to system ID if available
                    full_sys_id = f"{strain_id}_{sys_id}" if strain_id else sys_id
                    
                    # Write to FASTA file
                    self.write_fasta(dst, full_sys_id, sequence)
                    
                    # Record for DataFrame
                    file_path = system.get("file_path", str(fasta_file))
                    data.append((full_sys_id, length, file_path))
                
                progress.update(task, advance=1)
            
            progress.remove_task(task)
        
        # Create and return DataFrame
        progress.console.print(f"[bold green]{'Extracted':>12}[/] {len(data)} systems")
        return pandas.DataFrame(
            data=data,
            columns=["cluster_id", "cluster_length", "filename"]
        ).set_index("cluster_id")
    
    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
    ) -> typing.Dict[str, int]:
        """Extracts protein sequences from defense system protein files.
        
        Args:
            progress: Progress bar for tracking.
            inputs: List of input files (protein FASTA files or TSV summary).
            output: Output path for combined sequences.
            representatives: Set of representative cluster IDs.
            
        Returns:
            Dictionary mapping protein IDs to their lengths.
        """
        # Check for direct DefenseFinder integration mode
        if self._check_defense_finder_integration_mode():
            return self._extract_proteins_with_defense_extractor(progress, output, representatives)
        
        # Handle TSV summary file
        if len(inputs) == 1 and inputs[0].suffix.lower() == ".tsv":
            progress.console.print(f"[bold blue]{'Found':>12}[/] TSV summary file")
            try:
                df = pandas.read_csv(inputs[0], sep="\t")
                
                # Check if we have all required columns for advanced processing
                required_cols = ["systems_tsv", "genes_tsv", "faa_file", "fna_file"]
                if all(col in df.columns for col in required_cols):
                    # Use defense_extractor for processing
                    return self._extract_proteins_from_tsv(progress, df, output, representatives)
                
                # Process using existing method
                if "faa_file" in df.columns:
                    # Filter to representatives if provided
                    if representatives:
                        str_reps = set(str(r) for r in representatives)
                        progress.console.print(
                            f"[bold blue]{'Filtering':>12}[/] to {len(str_reps)} representatives"
                        )
                        sys_ids = [id for id in df["system_id"].astype(str) if id in "\t".join(str_reps)]
                        df = df[df["system_id"].astype(str).isin(sys_ids)]
                    
                    faa_files = [pathlib.Path(f) for f in df["faa_file"] if f]
                    return self._process_faa_files(progress, faa_files, output, representatives)
                else:
                    progress.console.print(
                        f"[bold yellow]{'Warning':>12}[/] TSV missing 'faa_file' column"
                    )
            except Exception as e:
                progress.console.print(f"[bold red]{'Error':>12}[/] reading TSV: {e}")
                
        # Use existing methods for other input types
        return super().extract_proteins(progress, inputs, output, representatives)
    
    def _extract_proteins_with_defense_extractor(
        self,
        progress: rich.progress.Progress,
        output: pathlib.Path,
        representatives: typing.Container[str]
    ) -> typing.Dict[str, int]:
        """Extract proteins using the DefenseExtractor."""
        # Create temporary directory for output if needed
        temp_dir = None
        if self.write_output and not self.output_dir:
            temp_dir = tempfile.TemporaryDirectory()
            output_dir = pathlib.Path(temp_dir.name)
        else:
            output_dir = self.output_dir
        
        try:
            # Create extractor
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=output_dir,
                write_output=self.write_output
            )
            
            # Use defense_metadata file if provided
            if self.defense_metadata:
                progress.console.print(f"[bold blue]{'Using':>12}[/] defense metadata file: {self.defense_metadata}")
                try:
                    df = pandas.read_csv(self.defense_metadata, sep="\t")
                    return self._extract_proteins_from_tsv(progress, df, output, representatives, extractor)
                except Exception as e:
                    progress.console.print(f"[bold red]{'Error':>12}[/] reading defense metadata: {e}")
                    return {}
            
            # Use individual files
            progress.console.print(f"[bold blue]{'Using':>12}[/] direct defense finder files")
            
            # Extract proteins
            gene_data = extractor.extract_gene_sequences(
                systems_tsv_file=self.defense_systems_tsv,
                genes_tsv_file=self.defense_genes_tsv,
                faa_file=self.defense_systems_tsv.parent / "proteins.faa",
                fna_file=self.defense_systems_tsv.parent / "genes.fna",
                output_dir=output_dir if self.write_output else None
            )
            
            # Write proteins to output file and record sizes
            protein_sizes = {}
            with open(output, "w") as dst:
                for sys_id, system in gene_data.items():
                    # Skip if not in representatives
                    if representatives and sys_id not in representatives:
                        continue
                    
                    # Write proteins
                    for prot_id, protein in system.get("proteins", {}).items():
                        seq_id = f"{sys_id}_{prot_id}"
                        sequence = protein["sequence"]
                        
                        # Write to output
                        self.write_fasta(dst, seq_id, sequence)
                        
                        # Record size
                        protein_sizes[seq_id] = len(sequence)
            
            progress.console.print(
                f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(gene_data)} systems"
            )
            return protein_sizes
            
        finally:
            # Clean up temporary directory if created
            if temp_dir:
                temp_dir.cleanup()
    
    def _extract_proteins_from_tsv(
        self,
        progress: rich.progress.Progress,
        df: pandas.DataFrame,
        output: pathlib.Path,
        representatives: typing.Container[str],
        extractor: typing.Optional[DefenseExtractor] = None
    ) -> typing.Dict[str, int]:
        """Extract proteins from defense systems specified in a TSV file."""
        # Create extractor if not provided
        if extractor is None:
            extractor = DefenseExtractor(
                progress=progress,
                output_base_dir=self.output_dir,
                write_output=self.write_output
            )
        
        # Filter to representatives if provided
        if representatives:
            rep_set = set(str(r) for r in representatives)
            # Try to match on system_id first
            if "system_id" in df.columns:
                df = df[df["system_id"].astype(str).isin(rep_set)]
            # Then try sys_id
            elif "sys_id" in df.columns:
                df = df[df["sys_id"].astype(str).isin(rep_set)]
        
        # Process each row in DataFrame
        protein_sizes = {}
        with open(output, "w") as dst:
            task = progress.add_task(f"[bold blue]{'Processing':>9}[/] protein sequences", total=len(df))
            
            for _, row in df.iterrows():
                strain_id = row.get("strain_id", None)
                progress.update(task, description=f"[bold blue]{'Processing':>9}[/] strain: {strain_id}")
                
                # Get required files
                systems_tsv = pathlib.Path(row["systems_tsv"])
                genes_tsv = pathlib.Path(row["genes_tsv"])
                faa_file = pathlib.Path(row["faa_file"])
                fna_file = pathlib.Path(row.get("fna_file", "")) # fna file is optional
                
                # Check if required files exist
                missing_files = []
                for f, name in [(systems_tsv, "systems_tsv"), (genes_tsv, "genes_tsv"), (faa_file, "faa_file")]:
                    if not f.exists():
                        missing_files.append(f"{name}: {f}")
                
                if missing_files:
                    progress.console.print(f"[bold yellow]{'Missing':>12}[/] files for {strain_id}: {', '.join(missing_files)}")
                    progress.update(task, advance=1)
                    continue
                
                # Extract gene sequences
                strain_output_dir = None
                if self.write_output and self.output_dir:
                    strain_output_dir = self.output_dir / (strain_id if strain_id else "unknown") / "proteins"
                    strain_output_dir.mkdir(parents=True, exist_ok=True)
                
                gene_data = extractor.extract_gene_sequences(
                    systems_tsv_file=systems_tsv,
                    genes_tsv_file=genes_tsv,
                    faa_file=faa_file,
                    fna_file=fna_file if fna_file.exists() else None,
                    output_dir=strain_output_dir,
                    strain_id=strain_id
                )
                
                # Write proteins to output file and record sizes
                for sys_id, system in gene_data.items():
                    # Add strain ID to system ID if available
                    full_sys_id = f"{strain_id}_{sys_id}" if strain_id else sys_id
                    
                    # Skip if not in representatives (double check)
                    if representatives and full_sys_id not in representatives and sys_id not in representatives:
                        continue
                    
                    # Write proteins
                    for prot_id, protein in system.get("proteins", {}).items():
                        seq_id = f"{full_sys_id}_{prot_id}"
                        sequence = protein["sequence"]
                        
                        # Write to output
                        self.write_fasta(dst, seq_id, sequence)
                        
                        # Record size
                        protein_sizes[seq_id] = len(sequence)
                
                progress.update(task, advance=1)
            
            progress.remove_task(task)
        
        progress.console.print(
            f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins"
        )
        return protein_sizes


class GFFDataset(BaseDataset):
    """GFF dataset class.
    This class is used to extract nucleotide and protein sequences from GFF files.
    It inherits from the BaseDataset class and implements the extract_sequences
    and extract_proteins methods.
    """
    def __init__(self, gff: typing.Optional[typing.Union[str, pathlib.Path]] = None):
        """Initializes the GFFDataset class.

        Args:
            gff (typing.Optional[typing.Union[str, pathlib.Path]], optional): pathlib.Path to the GFF file. Defaults to None.
        """
    def extract_sequences(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        ) -> pandas.DataFrame: 
        raise NotImplementedError("extract_sequences method is not implemented for GFFDataset")

    def extract_proteins(
        self,
        progress: rich.progress.Progress,
        inputs: typing.List[pathlib.Path],
        output: pathlib.Path,
        representatives: typing.Container[str]
        ) -> typing.Dict[str, int]:
        raise NotImplementedError("extract_proteins method is not implemented for GFFDataset")