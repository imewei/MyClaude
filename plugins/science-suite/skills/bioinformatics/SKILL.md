---
name: bioinformatics
description: "Implement bioinformatics workflows including sequence alignment (BLAST, Smith-Waterman), genomics pipelines, phylogenetic analysis, protein structure prediction, and biological data visualization with Biopython. Use when analyzing biological sequences, building genomics pipelines, or processing bioinformatics data."
---

# Bioinformatics

Analyze biological sequences, build genomics pipelines, and process bioinformatics data.

## Expert Agent

For rigorous scientific methodology in bioinformatics research, delegate to the expert agent:

- **`research-expert`**: Research methodology specialist for experimental design, statistical rigor, and reproducibility.
  - *Location*: `plugins/science-suite/agents/research-expert.md`
  - *Capabilities*: Experimental design, literature review, statistical analysis, publication workflow.

## Biopython Essentials

```python
from Bio import SeqIO, Entrez, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Parse FASTA
def load_sequences(fasta_path: str) -> list[SeqRecord]:
    """Load all sequences from a FASTA file."""
    return list(SeqIO.parse(fasta_path, "fasta"))

# Basic sequence operations
seq = Seq("ATGCGATCGATCG")
protein = seq.translate()           # Translate to amino acids
complement = seq.complement()       # Complementary strand
rc = seq.reverse_complement()       # Reverse complement
gc_content = (seq.count("G") + seq.count("C")) / len(seq)
```

## Sequence Alignment

```python
from Bio import pairwise2
from Bio.Align import PairwiseAligner

# Modern pairwise alignment
aligner = PairwiseAligner()
aligner.mode = "local"  # Smith-Waterman
aligner.match_score = 2
aligner.mismatch_score = -1
aligner.open_gap_score = -5
aligner.extend_gap_score = -0.5

alignments = aligner.align("ACGTACGT", "ACGACGT")
best = alignments[0]
print(f"Score: {best.score}")
print(best)
```

## BLAST Searches

```python
from Bio.Blast import NCBIWWW, NCBIXML

def run_blast(sequence: str, database: str = "nr", program: str = "blastn") -> list[dict]:
    """Run remote BLAST and parse results."""
    result_handle = NCBIWWW.qblast(program, database, sequence)
    blast_records = NCBIXML.parse(result_handle)
    hits = []
    for record in blast_records:
        for alignment in record.alignments[:10]:
            for hsp in alignment.hsps:
                hits.append({
                    "title": alignment.title[:80],
                    "score": hsp.score,
                    "e_value": hsp.expect,
                    "identities": hsp.identities,
                    "align_length": hsp.align_length,
                    "query_coverage": hsp.align_length / len(sequence),
                })
    return hits

# Local BLAST (faster for large datasets)
# Install: apt-get install ncbi-blast+
# Command: blastn -query input.fasta -db nt -out results.xml -outfmt 5
```

## Genomics Pipeline

| Step | Tool | Purpose |
|------|------|---------|
| Quality control | FastQC, MultiQC | Read quality assessment |
| Trimming | Trimmomatic, fastp | Adapter and quality trimming |
| Alignment | BWA-MEM2, STAR | Read mapping to reference |
| Variant calling | GATK, DeepVariant | SNP/Indel detection |
| Annotation | SnpEff, VEP | Functional annotation |
| Visualization | IGV, pyGenomeTracks | Genome browser views |

## Phylogenetic Analysis

```python
from Bio import Phylo
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor

def build_phylogeny(alignment_path: str, method: str = "nj") -> Phylo.BaseTree.Tree:
    """Build phylogenetic tree from multiple sequence alignment."""
    alignment = AlignIO.read(alignment_path, "fasta")
    calculator = DistanceCalculator("identity")
    distance_matrix = calculator.get_distance(alignment)
    constructor = DistanceTreeConstructor()
    if method == "nj":
        tree = constructor.nj(distance_matrix)
    elif method == "upgma":
        tree = constructor.upgma(distance_matrix)
    else:
        raise ValueError(f"Unknown method: {method}")
    return tree

# Visualize
# Phylo.draw(tree)
# Phylo.draw_ascii(tree)
```

## Protein Structure

```python
from Bio.PDB import PDBParser, DSSP, NeighborSearch

def analyze_structure(pdb_path: str) -> dict:
    """Parse PDB and extract structural features."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    model = structure[0]
    residues = list(model.get_residues())
    atoms = list(model.get_atoms())
    return {
        "n_chains": len(list(model.get_chains())),
        "n_residues": len(residues),
        "n_atoms": len(atoms),
    }
```

## Database Access

```python
Entrez.email = "your.email@example.com"

def fetch_genbank(accession: str) -> SeqRecord:
    """Fetch a GenBank record by accession number."""
    handle = Entrez.efetch(db="nucleotide", id=accession, rettype="gb", retmode="text")
    record = SeqIO.read(handle, "genbank")
    handle.close()
    return record

def search_pubmed(query: str, max_results: int = 10) -> list[str]:
    """Search PubMed and return PMIDs."""
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    handle.close()
    return record["IdList"]
```

## Bioinformatics Workflow Checklist

- [ ] Validate input data quality (FastQC/MultiQC)
- [ ] Document genome build version (hg38, GRCh38, mm10)
- [ ] Record all tool versions and parameters
- [ ] Apply appropriate multiple testing correction (Bonferroni, BH-FDR)
- [ ] Validate results against known controls
- [ ] Archive raw data with checksums (MD5/SHA256)
- [ ] Use reproducible pipeline managers (Snakemake, Nextflow)
- [ ] Report BLAST E-values, not just scores
