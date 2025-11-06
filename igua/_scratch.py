# Yes! Here are better alternatives to gffutils that are much faster:

# 1. gffpandas + pandas (Recommended for your use case)
# Super fast in-memory lookups, similar to pyfaidx:

import gffpandas as gffpd

class GenomeResources:
    def __init__(self, ...):
        self._gff_df: Optional[pd.DataFrame] = None
        self._gff_id_index: Optional[Dict[str, int]] = None
    
    @property
    def gff_db(self) -> pd.DataFrame:
        """Load GFF as pandas DataFrame (lazy-loaded, cached)"""
        if self._gff_df is None:
            # Read GFF into DataFrame
            annotation = gffpd.read_gff3(str(self.context.gff_file))
            self._gff_df = annotation.df
            
            # Build ID index for O(1) lookup
            self._gff_id_index = {
                row['attributes'].get('ID', f"row_{i}"): i 
                for i, row in self._gff_df.iterrows()
                if row['type'] in ['gene', 'CDS', 'mRNA']
            }
        
        return self._gff_df
    
    def find_gene_feature(self, gene_id: str) -> Optional[pd.Series]:
        """Find gene feature with O(1) direct lookup, O(n) fallback"""
        df = self.gff_db  # Triggers loading
        
        # Strategy 1: Direct ID lookup (O(1))
        if gene_id in self._gff_id_index:
            return df.iloc[self._gff_id_index[gene_id]]
        
        # Strategy 2: Common variations (O(1) each)
        for variant in [
            f"gene-{gene_id}",
            f"cds-{gene_id}",
            gene_id.replace('_', '~'),
            gene_id.replace('~', '_'),
        ]:
            if variant in self._gff_id_index:
                return df.iloc[self._gff_id_index[variant]]
        
        # Strategy 3: Attribute search (O(n) but vectorized, fast)
        mask = df['type'].isin(['gene', 'CDS', 'mRNA'])
        for attr in ['locus_tag', 'Name', 'gene', 'old_locus_tag']:
            matches = df[mask].apply(
                lambda row: attr in row['attributes'] and gene_id in row['attributes'][attr],
                axis=1
            )
            if matches.any():
                return df[mask][matches].iloc[0]
        
        return None



# 2. pysam + Tabix (Fastest for coordinate-based queries)
# If you know coordinates, this is blazingly fast (works like pyfaidx):


import pysam

class GenomeResources:
    @property
    def gff_db(self) -> pysam.TabixFile:
        """Load GFF with Tabix index (lazy-loaded, cached)"""
        if self._gff_db is None:
            # Must be bgzipped and tabix-indexed first
            # pysam.tabix_index(str(self.context.gff_file), preset="gff")
            self._gff_db = pysam.TabixFile(str(self.context.gff_file))
        return self._gff_db
    
    def find_gene_feature(self, gene_id: str, chrom: str = None, start: int = None, end: int = None):
        """Fast coordinate-based lookup"""
        if chrom and start and end:
            # O(log n) coordinate lookup - super fast!
            for row in self.gff_db.fetch(chrom, start, end):
                fields = row.split('\t')
                if gene_id in fields[8]:  # attributes column
                    return fields
        
        # Fallback: scan (slower)
        for row in self.gff_db.fetch():
            if gene_id in row:
                return row.split('\t')
        
        return None



# 3. Custom Binary Search with mmap (Fastest overall)
# Roll your own indexed reader (like pyfaidx does):

import mmap
import re

class FastGFFReader:
    """Memory-mapped GFF reader with line index"""
    def __init__(self, gff_path: pathlib.Path):
        self.path = gff_path
        self._index: Dict[str, int] = {}
        self._build_index()
    
    def _build_index(self):
        """Build byte offset index for fast seeks"""
        with open(self.path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            offset = 0
            
            while True:
                line_start = offset
                line_end = mm.find(b'\n', offset)
                if line_end == -1:
                    break
                
                line = mm[line_start:line_end].decode('utf-8')
                if not line.startswith('#'):
                    fields = line.split('\t')
                    if len(fields) >= 9:
                        # Extract ID from attributes
                        attrs = dict(re.findall(r'(\w+)=([^;]+)', fields[8]))
                        if 'ID' in attrs:
                            self._index[attrs['ID']] = line_start
                
                offset = line_end + 1
            
            mm.close()
    
    def get_feature(self, gene_id: str) -> Optional[Dict]:
        """O(1) lookup by ID"""
        if gene_id not in self._index:
            return None
        
        with open(self.path, 'rb') as f:
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
            mm.seek(self._index[gene_id])
            line = mm.readline().decode('utf-8').strip()
            mm.close()
            
            fields = line.split('\t')
            return {
                'seqid': fields[0],
                'start': int(fields[3]),
                'end': int(fields[4]),
                'strand': fields[6],
                'attributes': dict(re.findall(r'(\w+)=([^;]+)', fields[8]))
            }

class GenomeResources:
    @property
    def gff_db(self) -> FastGFFReader:
        if self._gff_db is None:
            self._gff_db = FastGFFReader(self.context.gff_file)
        return self._gff_db



# 4. BCBio.GFF + In-Memory Dict (Simpler alternative)
# Lightweight, no dependencies:

from BCBio import GFF

class GenomeResources:
    @property
    def gff_db(self) -> Dict[str, Any]:
        """Load GFF into memory dictionary (lazy-loaded)"""
        if self._gff_db is None:
            self._gff_db = {}
            with open(self.context.gff_file) as handle:
                for record in GFF.parse(handle):
                    for feature in record.features:
                        # Index by multiple IDs
                        for key in [feature.id, 
                                   feature.qualifiers.get('locus_tag', [''])[0],
                                   feature.qualifiers.get('gene', [''])[0]]:
                            if key:
                                self._gff_db[key] = {
                                    'seqid': record.id,
                                    'start': feature.location.start,
                                    'end': feature.location.end,
                                    'strand': feature.location.strand,
                                }
        return self._gff_db



# Performance Comparison
# Method	Direct Lookup	Attribute Lookup	Memory Usage
# gffutils	O(log n) SQLite	O(n) or indexed	Low (disk-based)
# gffpandas	O(1) with index	O(n) vectorized	Medium (in-memory)
# pysam	O(log n) tabix	O(n) scan	Low (indexed)
# Custom mmap	O(1)	O(n) scan	Very low (mmap)
# BCBio dict	O(1)	O(1) if indexed	High (full load)

# Recommendation for Your Use Case
# Use gffpandas with custom indexing:

# Installation
# pip install gffpandas

class GenomeResources:
    def __init__(self, ...):
        self._gff_df: Optional[pd.DataFrame] = None
        self._gff_lookup: Optional[Dict[str, int]] = None
    
    @property  
    def gff_db(self) -> pd.DataFrame:
        if self._gff_df is None:
            import gffpandas as gffpd
            annotation = gffpd.read_gff3(str(self.context.gff_file))
            self._gff_df = annotation.df
            
            # Build comprehensive lookup index
            self._gff_lookup = {}
            for idx, row in self._gff_df.iterrows():
                if row['type'] not in ['gene', 'CDS', 'mRNA']:
                    continue
                
                # Index by ID
                attrs = row.get('attributes', {})
                for key in ['ID', 'locus_tag', 'Name', 'gene', 'old_locus_tag']:
                    val = attrs.get(key)
                    if val:
                        self._gff_lookup[val] = idx
        
        return self._gff_df
    
    def find_gene_feature(self, gene_id: str) -> Optional[pd.Series]:
        _ = self.gff_db  # Trigger load
        
        # O(1) lookup with variations
        for variant in [gene_id, f"gene-{gene_id}", f"cds-{gene_id}",
                       gene_id.replace('_', '~'), gene_id.replace('~', '_')]:
            if variant in self._gff_lookup:
                return self._gff_df.iloc[self._gff_lookup[variant]]
        
        return None