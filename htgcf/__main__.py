import argparse
import collections
import contextlib
import csv
import os
import pathlib
import tempfile
import subprocess

import gb_io
import rich
import numpy
import pandas
import scipy.sparse
import scipy.cluster.hierarchy

from .mmseqs import run_mmseqs

try:
    import argcomplete
except ImportError as err:
    argcomplete = err

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", action="append", default=[])
parser.add_argument("-o", "--output", default="gcfs.tsv")
parser.add_argument("-t", "--tempdir")

if not isinstance(argcomplete, ImportError):
    argcomplete.autocomplete(parser)

args = parser.parse_args()


with contextlib.ExitStack() as ctx:

    progress = ctx.enter_context(rich.progress.Progress())
    tmpdir = pathlib.Path(ctx.enter_context(tempfile.TemporaryDirectory(dir=args.tempdir)))

    # extract raw sequences
    rich.print(f"[bold blue]{'Converting':>12}[/] input clusters")
    clusters_fna = tmpdir.joinpath("clusters.fna")
    with clusters_fna.open("w") as dst:
        for input_path in args.input:
            for record in gb_io.iter(input_path):
                dst.write(">{}\n".format(record.name))
                dst.write(record.sequence.decode("ascii"))
                dst.write("\n")

    # mmseqs1
    rich.print(f"[bold blue]{'Running':>12}[/] nucleotide deduplication step with [purple]mmseqs[/]")
    step1 = tmpdir.joinpath("step1")
    run_mmseqs(
        progress,
        "easy-linclust",
        clusters_fna,
        step1,
        tmpdir.joinpath("tmp"),
        createdb_mode=1,
        min_seq_id=0.85,
        c=1,
        cluster_mode=0,
        cov_mode=1,
    )

    # mmseqs2
    rich.print(f"[bold blue]{'Running':>12}[/] nucleotide clustering step with [purple]mmseqs[/]")
    step2 = tmpdir.joinpath("step2")
    run_mmseqs(
        progress,
        "easy-linclust",
        tmpdir.joinpath("step1_rep_seq.fasta"),
        step2,
        tmpdir.joinpath("tmp"),
        createdb_mode=1,
        min_seq_id=0.6,
        c=0.5,
        cluster_mode=0,
        cov_mode=0,
    )

    # load representatives
    rich.print(f"[bold blue]{'Loading':>12}[/] representative clusters")
    with tmpdir.joinpath("step2_cluster.tsv").open() as f:
        representatives = { row[0] for row in csv.reader(f, dialect="excel-tab") }
        representatives_map = {x:i for i,x in enumerate(sorted(representatives))}

    # extract proteins and record sizes
    rich.print(f"[bold blue]{'Extracting':>12}[/] protein sequences from clusters")
    cluster_proteins = collections.defaultdict(list)
    protein_sizes = {}
    proteins_faa = tmpdir.joinpath("proteins.faa")
    with proteins_faa.open("w") as dst:
        for input_path in args.input:
            for record in gb_io.iter(input_path):
                if record.name in representatives:
                    for i, feat in enumerate(filter(lambda f: f.type == "CDS", record.features)):
                        qualifiers = feat.qualifiers.to_dict()
                        translation = qualifiers["translation"][0].rstrip("*")
                        protein_id = "{}_{}".format(record.name, i)
                        dst.write(">")
                        dst.write(protein_id)
                        dst.write("\n")
                        dst.write(translation)
                        dst.write("\n")
                        protein_sizes[protein_id] = len(translation)
                        cluster_proteins[record.name].append(protein_id)

    # build db
    rich.print(f"[bold blue]{'Building':>12}[/] protein sequences database")
    proteins_db = tmpdir.joinpath("proteins.db")
    run_mmseqs(
        progress,
        "createdb",
        proteins_faa,
        proteins_db,
        createdb_mode=1,
    )

    # build linindex
    rich.print(f"[bold blue]{'Indexing':>12}[/] protein sequences database")
    run_mmseqs(
        progress,
        "createlinindex",
        proteins_db,
        tmpdir.joinpath("tmp")
    )

    # many-to-many protein search
    rich.print(f"[bold blue]{'Computing':>12}[/] pairwise similarity for proteins")
    protalis_db = tmpdir.joinpath("protalis.db")
    run_mmseqs(
        progress,
        "search",
        proteins_db,
        proteins_db,
        protalis_db,
        tmpdir.joinpath("tmp")
    )

    # convert alis
    rich.print(f"[bold blue]{'Converting':>12}[/] alignment results")
    protalis_tbl = tmpdir.joinpath("protalis.tbl")
    run_mmseqs(
        progress,
        "convertalis",
        proteins_db,
        proteins_db,
        protalis_db,
        protalis_tbl,
    )

    # make identity matrix
    identity_matrix = scipy.sparse.dok_matrix((len(representatives), len(representatives)))
    with protalis_tbl.open() as f:
        for row in csv.reader(f, dialect="excel-tab"):
            query, target, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue, bitscore = row
            if float(evalue) >= 1e-5:
                continue
            query_cluster = query.rsplit("_", 1)[0]
            target_cluster = target.rsplit("_", 1)[0]
            query_index = representatives_map[query_cluster]
            target_index = representatives_map[target_cluster]
            identity_matrix[query_index, target_index] += int(int(length)*float(pident))

    # compute total cluster sizes
    rich.print(f"[bold blue]{'Computing':>12}[/] total cluster sizes")
    cluster_sizes = numpy.zeros(len(representatives), dtype=numpy.int32)
    for protein_id, protein_size in protein_sizes.items():
        cluster_id = protein_id.rsplit("_", 1)[0]
        cluster_index = representatives_map[cluster_id]
        cluster_sizes[cluster_index] += protein_size

    # make similarity matrix
    rich.print(f"[bold blue]{'Building':>12}[/] pairwise identity matrix")
    similarity_matrix = scipy.sparse.triu(scipy.sparse.coo_matrix(identity_matrix), k=1)
    for k, (i, j) in enumerate(zip(similarity_matrix.row, similarity_matrix.col)):
        similarity_matrix.data[k] /= cluster_sizes[i] + cluster_sizes[j]

    # 
    rich.print(f"[bold blue]{'Allocating':>12}[/] condensed distance matrix")
    distance_matrix = numpy.ones( len(representatives) * (len(representatives)-1) // 2 )

    #
    rich.print(f"[bold blue]{'Filling':>12}[/] condensed distance matrix")
    for row, col, x in zip(similarity_matrix.row, similarity_matrix.col, similarity_matrix.data):
        k = (2*len(representatives) - 3 - row)*(row) >> 1 +(col) - 1
        distance_matrix[k] -= x
    numpy.clip(distance_matrix, 0, None, out=distance_matrix)

    # 
    rich.print(f"[bold blue]{'Clustering':>12}[/] using average linkage")
    Z = scipy.cluster.hierarchy.linkage(distance_matrix, method="average")

    #
    gcfs1 = pandas.read_csv(tmpdir.joinpath("step1_cluster.tsv"), sep="\t", header=None, names=["gcf1_representative", "bgc_id"])
    gcfs2 = pandas.read_csv(tmpdir.joinpath("step2_cluster.tsv"), sep="\t", header=None, names=["gcf2_representative", "gcf1_representative"])

    # 
    flat = scipy.cluster.hierarchy.fcluster(Z, criterion="distance", t=0.4)

    #
    gcfs3 = pandas.DataFrame({
        "gcf_id": [ f"GCF{i:07}" for i in flat ],
        "gcf2_representative": sorted(gcfs2["gcf2_representative"].unique()),
    })

    #
    gcfs = pandas.merge(
        pandas.merge(gcfs1, gcfs2, on="gcf1_representative"),
        gcfs3,
        on="gcf2_representative",
    )

    # extract GCF3
    gcf_members = { 
        gcf_id:";".join(sorted(rows["bgc_id"].unique())) 
        for gcf_id, rows in gcfs.groupby("gcf_id") 
    }
    gcfs3["gcf_members"] = gcfs3["gcf_id"].apply(gcf_members.__getitem__)
    gcfs3["n_members"] = gcfs3["gcf_members"].str.count(";") + 1


    # if [ ! -e "gcfs3.tsv" ]; then
    #     log Building final GCF assignment table
    #     python scripts/make_gcfs3.py --step1 step1_cluster.tsv --step2 step2_cluster.tsv -Z similarity.average_dendrogram.npz --o>
    # fi

    #
    gcfs3.to_csv(args.output, sep="\t", index=False)

