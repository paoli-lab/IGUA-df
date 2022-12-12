import argparse
import collections
import contextlib
import csv
import functools
import itertools
import os
import pathlib
import tempfile
import subprocess
import typing

import gb_io
import rich
import numpy
import pandas
import scipy.sparse
import scipy.cluster.hierarchy

try:
    from fastcluster import linkage
except ImportError:
    from scipy.cluster.hierarchy import _linkage
    linkage = functools.partial(_linkage, preserve_input=False)

try:
    import argcomplete
except ImportError as err:
    argcomplete = err

try:
    from rich_argparse import RichHelpFormatter as HelpFormatter
except ImportError:
    from argparse import HelpFormatter

from .mmseqs import MMSeqs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(formatter_class=HelpFormatter)
    parser.add_argument("-i", "--input", help="the GenBank files containing the clusters to process", action="append", default=[], required=True)
    parser.add_argument("-o", "--output", help="the name of the ouput file to generate", default="gcfs.tsv")
    parser.add_argument("-t", "--tmpdir", help="a temporary folder to use for processing", metavar="TMP")
    parser.add_argument("-j", "--jobs", help="the number of threads to use", type=int, default=1, metavar="N")
    return parser

def write_cluster(record: gb_io.Record, file: typing.TextIO) -> None:
    file.write(">{}\n".format(record.name))
    file.write(record.sequence.decode("ascii"))
    file.write("\n")

def main(argv: typing.Optional[typing.List[str]] = None) -> int:
    # build parser and get arguments
    parser = build_parser()
    if not isinstance(argcomplete, ImportError):
        argcomplete.autocomplete(parser)
    args = parser.parse_args()


    with contextlib.ExitStack() as ctx:
        # prepare progress bar
        progress = ctx.enter_context(rich.progress.Progress(
            "",
            rich.progress.SpinnerColumn(),
            *rich.progress.Progress.get_default_columns(),
        ))
        mmseqs = MMSeqs(progress=progress, threads=args.jobs)

        # open temporary folder
        if args.tmpdir is None:
            tmpdir = pathlib.Path(ctx.enter_context(tempfile.TemporaryDirectory()))
        else:
            tmpdir = pathlib.Path(args.tmpdir)

        # extract raw sequences
        progress.console.print(f"[bold blue]{'Loading':>12}[/] input clusters")
        clusters_fna = tmpdir.joinpath("clusters.fna")
        clusters_lengths = {}
        with clusters_fna.open("w") as dst:
            for input_path in args.input:
                for record in gb_io.iter(input_path):
                    if record.name in clusters_lengths:
                        raise ValueError(f"Duplicate cluster name in input: {record.name!r}")
                    write_cluster(record, dst)
                    clusters_lengths[record.name] = len(record.sequence)
        progress.console.print(f"[bold green]{'Loaded':>12}[/] {len(clusters_lengths)} clusters to process")

        # mmseqs step 1
        progress.console.print(f"[bold blue]{'Starting':>12}[/] nucleotide deduplication step with [purple]mmseqs[/]")
        mmseqs.run(
            "easy-linclust",
            clusters_fna,
            tmpdir.joinpath("step1"),
            tmpdir.joinpath("tmp"),
            createdb_mode=1,
            min_seq_id=0.85,
            c=1,
            cluster_mode=0,
            cov_mode=1,
        )
        gcfs1 = pandas.read_csv(
            tmpdir.joinpath("step1_cluster.tsv"),
            sep="\t",
            header=None,
            names=["gcf1_representative", "cluster_id"]
        )
        progress.console.print(f"[bold green]{'Reduced':>12}[/] {len(gcfs1)} clusters to {len(gcfs1.gcf1_representative.unique())} complete representatives")

        # mmseqs step 2
        progress.console.print(f"[bold blue]{'Starting':>12}[/] nucleotide clustering step with [purple]mmseqs[/]")
        mmseqs.run(
            "easy-linclust",
            tmpdir.joinpath("step1_rep_seq.fasta"),
            tmpdir.joinpath("step2"),
            tmpdir.joinpath("tmp"),
            createdb_mode=1,
            min_seq_id=0.6,
            c=0.5,
            cluster_mode=0,
            cov_mode=0,
        )
        gcfs2 = pandas.read_csv(
            tmpdir.joinpath("step2_cluster.tsv"),
            sep="\t",
            header=None,
            names=["gcf2_representative", "gcf1_representative"]
        )
        progress.console.print(f"[bold green]{'Reduced':>12}[/] {len(gcfs2)} clusters to {len(gcfs2.gcf2_representative.unique())} nucleotide representatives")

        # load representatives
        progress.console.print(f"[bold blue]{'Loading':>12}[/] representative clusters")
        with tmpdir.joinpath("step2_cluster.tsv").open() as f:
            representatives = { row[0] for row in csv.reader(f, dialect="excel-tab") }
            representatives_map = {x:i for i,x in enumerate(sorted(representatives))}
            r = len(representatives)
        progress.console.print(f"[bold green]{'Loaded':>12}[/] {r} nucleotide representative clusters")

        # extract proteins and record sizes
        progress.console.print(f"[bold blue]{'Extracting':>12}[/] protein sequences from clusters")
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
        progress.console.print(f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(cluster_proteins)} nucleotide representative")

        # build db
        progress.console.print(f"[bold blue]{'Building':>12}[/] protein sequences database")
        proteins_db = tmpdir.joinpath("proteins.db")
        mmseqs.run(
            "createdb",
            proteins_faa,
            proteins_db,
            createdb_mode=1,
        )

        # build linindex
        progress.console.print(f"[bold blue]{'Indexing':>12}[/] protein sequences database")
        mmseqs.run(
            "createlinindex",
            proteins_db,
            tmpdir.joinpath("tmp")
        )

        # many-to-many protein search
        progress.console.print(f"[bold blue]{'Computing':>12}[/] pairwise similarity for proteins")
        protalis_db = tmpdir.joinpath("protalis.db")
        mmseqs.run(
            "search",
            proteins_db,
            proteins_db,
            protalis_db,
            tmpdir.joinpath("tmp")
        )

        # # convert alis
        progress.console.print(f"[bold blue]{'Converting':>12}[/] alignment results")
        protalis_tbl = tmpdir.joinpath("protalis.tbl")
        mmseqs.run(
            "convertalis",
            proteins_db,
            proteins_db,
            protalis_db,
            protalis_tbl,
        )

        # make identity matrix
        progress.console.print(f"[bold blue]{'Creating':>12}[/] weighted identity matrix")
        identity_matrix = scipy.sparse.dok_matrix((r, r))
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
        progress.console.print(f"[bold blue]{'Computing':>12}[/] total cluster sizes")
        cluster_sizes = numpy.zeros(r, dtype=numpy.int32)
        for protein_id, protein_size in protein_sizes.items():
            cluster_id = protein_id.rsplit("_", 1)[0]
            cluster_index = representatives_map[cluster_id]
            cluster_sizes[cluster_index] += protein_size

        # size_matrix = numpy.repeat(identity_matrix.diagonal(), r).reshape(-1, r)
        distance_matrix = (identity_matrix + identity_matrix.T).toarray()

        progress.console.print(f"[bold blue]{'Computing':>12}[/] distance matrix")
        xdist = numpy.zeros((r, r))
        for i in progress.track(range(r)):
            for j in range(i+1, r):
                xdist[i, j] = xdist[j, i] = numpy.sum((distance_matrix[i] - distance_matrix[j])**2)
        xdist = numpy.sqrt(xdist) / r

        # run clustering
        progress.console.print(f"[bold blue]{'Clustering':>12}[/] gene clusters using average linkage")
        Z = linkage(xdist, method="single")
        flat = scipy.cluster.hierarchy.fcluster(Z, criterion="distance", t=0.2)

        # build GCFs based on
        gcfs3 = pandas.DataFrame({
            "gcf_id": [ f"GCF{i:07}" for i in flat ],
            "gcf2_representative": sorted(representatives),
        })
        progress.console.print(f"[bold green]{'Built':>12}[/] {len(gcfs3.gcf_id.unique())} GCFs from {len(clusters_lengths)} initial clusters")

        #
        gcfs = pandas.merge(
            pandas.merge(gcfs1, gcfs2, on="gcf1_representative"),
            gcfs3,
            on="gcf2_representative",
        )
        gcfs["length"] = gcfs["cluster_id"].apply(clusters_lengths.__getitem__)

        #
        gcfs = gcfs.sort_values("gcf_id")
        gcfs = gcfs[["cluster_id", "length", "gcf_id", "gcf1_representative", "gcf2_representative"]]
        gcfs.to_csv(args.output, sep="\t", index=False)

        # # extract GCF3 members
        # gcf_members = {
        #     gcf_id:";".join(sorted(rows["cluster_id"].unique()))
        #     for gcf_id, rows in gcfs.groupby("gcf_id")
        # }
        # gcfs3["gcf_members"] = gcfs3["gcf_id"].apply(gcf_members.__getitem__)
        # gcfs3["n_members"] = gcfs3["gcf_members"].str.count(";") + 1



        # #
        # gcfs3.to_csv(args.output, sep="\t", index=False)

