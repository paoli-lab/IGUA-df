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
import multiprocessing.pool

import gb_io
import rich
import numpy
import pandas
import scipy.sparse
import scipy.cluster.hierarchy
from sklearn.metrics.pairwise import pairwise_distances

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
    parser.add_argument("-i", "--input", help="the GenBank files containing the clusters to process", action="append", type=pathlib.Path, default=[], required=True)
    parser.add_argument("-o", "--output", help="the name of the ouput file to generate", default=pathlib.Path("gcfs.tsv"), type=pathlib.Path)
    parser.add_argument("-w", "--workdir", help="a folder to use for processing", metavar="TMP")
    parser.add_argument("-j", "--jobs", help="the number of threads to use", type=int, default=1, metavar="N")
    parser.add_argument("--prefix", help="the prefix for GCF identifiers", default="GCF")
    return parser

def write_cluster(record: gb_io.Record, file: typing.TextIO) -> None:
    file.write(">{}\n".format(record.name))
    file.write(record.sequence.decode("ascii"))
    file.write("\n")

def extract_sequences(progress: rich.progress.Progress, inputs: typing.List[pathlib.Path], output: pathlib.Path) -> typing.Dict[str, int]:
    clusters_lengths = {}
    n_duplicate = 0
    progress.console.print(f"[bold blue]{'Loading':>12}[/] input clusters")
    with open(output, "w") as dst:
        for input_path in inputs:
            task = progress.add_task(f"[bold blue]{'Reading':>9}[/]")
            with progress.open(input_path, "rb", task_id=task) as reader:
                for record in gb_io.iter(reader):
                    if record.name in clusters_lengths:
                        n_duplicate += 1
                    write_cluster(record, dst)
                    clusters_lengths[record.name] = len(record.sequence)
            progress.remove_task(task)
    progress.console.print(f"[bold green]{'Loaded':>12}[/] {len(clusters_lengths)} clusters to process")
    if n_duplicate > 0:
        progress.console.print(f"[bold yellow]{'Skipped':>12}[/] {n_duplicate} clusters with duplicate identifiers")
    return clusters_lengths

def deduplicate_sequences(mmseqs: MMSeqs, input_path: pathlib.Path, output_prefix: pathlib.Path, tmpdir: pathlib.Path):
    mmseqs.run(
        "easy-linclust",
        input_path,
        output_prefix,
        tmpdir,
        createdb_mode=1,
        min_seq_id=0.85,
        c=1,
        cluster_mode=0,
        cov_mode=1,
    )

def cluster_sequences(mmseqs: MMSeqs, input_path: pathlib.Path, output_prefix: pathlib.Path, tmpdir: pathlib.Path):
    mmseqs.run(
        "easy-linclust",
        input_path,
        output_prefix,
        tmpdir,
        createdb_mode=1,
        min_seq_id=0.6,
        c=0.5,
        cluster_mode=0,
        cov_mode=0,
    )

def extract_proteins(progress: rich.progress.Progress, inputs: typing.List[pathlib.Path], output: pathlib.Path, representatives: typing.Container[str]) -> typing.Dict[str, int]:
    progress.console.print(f"[bold blue]{'Extracting':>12}[/] protein sequences from clusters")
    protein_sizes = {}
    with output.open("w") as dst:
        for input_path in inputs:
            task = progress.add_task(f"[bold blue]{'Reading':>9}[/]")
            with progress.open(input_path, "rb", task_id=task) as reader:
                for record in gb_io.iter(reader):
                    if record.name in representatives:
                        for i, feat in enumerate(filter(lambda f: f.type == "CDS", record.features)):
                            qualifiers = feat.qualifiers.to_dict()
                            translation = qualifiers["translation"][0].rstrip("*")
                            protein_id = "{}_{}".format(record.name, i)
                            if protein_id not in protein_sizes:
                                dst.write(">")
                                dst.write(protein_id)
                                dst.write("\n")
                                dst.write(translation)
                                dst.write("\n")
                                protein_sizes[protein_id] = len(translation)
            progress.remove_task(task)
    progress.console.print(f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(representatives)} nucleotide representative")
    return protein_sizes

def cluster_proteins(mmseqs: MMSeqs, input_path: pathlib.Path, output_prefix: pathlib.Path, tmpdir: pathlib.Path):
    mmseqs.run(
        "easy-linclust",
        input_path,
        output_prefix,
        tmpdir,
        createdb_mode=1,
        min_seq_id=0.6,
        c=0.5,
        cluster_mode=0,
        cov_mode=0,
    )

def make_identity_matrix(progress: rich.progress.Progress, input_path: pathlib.Path, representatives: typing.Dict[str, int]) -> scipy.sparse.spmatrix:
    # make identity matrix
    progress.console.print(f"[bold blue]{'Creating':>12}[/] weighted identity matrix")
    r = len(representatives)
    identity_matrix = numpy.zeros((r, r))
    task = progress.add_task(description=f"[bold blue]{'Reading':>9}[/]")
    with progress.open(input_path, "r", task_id=task) as f:
        for row in csv.reader(f, dialect="excel-tab"):
            query, target, pident, length, mismatch, gapopen, qstart, qend, sstart, send, evalue, bitscore = row
            if float(evalue) >= 1e-5:
                continue
            query_cluster = query.rsplit("_", 1)[0]
            target_cluster = target.rsplit("_", 1)[0]
            query_index = representatives[query_cluster]
            target_index = representatives[target_cluster]
            identity_matrix[query_index, target_index] += int(int(length)*float(pident))
    progress.remove_task(task)
    return numpy.add(identity_matrix, identity_matrix.T, out=identity_matrix)

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
        if args.workdir is None:
            workdir = pathlib.Path(ctx.enter_context(tempfile.TemporaryDirectory()))
        else:
            workdir = pathlib.Path(args.workdir)
            workdir.mkdir(parents=True, exist_ok=True)

        # extract raw sequences
        clusters_fna = workdir.joinpath("clusters.fna")
        sequence_lengths = extract_sequences(progress, args.input, clusters_fna)

        # deduplicate fragments
        if not workdir.joinpath("step1_cluster.tsv").exists():
            progress.console.print(f"[bold blue]{'Starting':>12}[/] nucleotide deduplication step with [purple]mmseqs[/]")
            gcfs1 = deduplicate_sequences(mmseqs, clusters_fna, workdir.joinpath("step1"), workdir.joinpath("tmp"))
        gcfs1 = pandas.read_csv(
            workdir.joinpath("step1_cluster.tsv"),
            sep="\t",
            header=None,
            names=["gcf1_representative", "cluster_id"]
        )
        progress.console.print(f"[bold green]{'Reduced':>12}[/] {len(gcfs1)} clusters to {len(gcfs1.gcf1_representative.unique())} complete representatives")

        # cluster sequences
        if not workdir.joinpath("step2_cluster.tsv").exists():
            progress.console.print(f"[bold blue]{'Starting':>12}[/] nucleotide clustering step with [purple]mmseqs[/]")
            cluster_sequences(mmseqs, workdir.joinpath("step1_rep_seq.fasta"), workdir.joinpath("step2"), workdir.joinpath("tmp"))
        gcfs2 = pandas.read_csv(
            workdir.joinpath("step2_cluster.tsv"),
            sep="\t",
            header=None,
            names=["gcf2_representative", "gcf1_representative"]
        )
        progress.console.print(f"[bold green]{'Reduced':>12}[/] {len(gcfs2)} clusters to {len(gcfs2.gcf2_representative.unique())} nucleotide representatives")

        # load representatives
        progress.console.print(f"[bold blue]{'Extracting':>12}[/] representative clusters")
        representatives = { x:i for i,x in enumerate(sorted(gcfs2["gcf2_representative"].unique())) }
        r = len(representatives)
        progress.console.print(f"[bold green]{'Loaded':>12}[/] {r} nucleotide representative clusters")

        # extract proteins and record sizes
        proteins_faa = workdir.joinpath("proteins.faa")
        protein_sizes = extract_proteins(progress, args.input, proteins_faa, representatives)

        # cluster proteins
        if not workdir.joinpath("step3_cluster.tsv").exists():
            cluster_proteins(mmseqs, proteins_faa, workdir.joinpath("step3"), workdir.joinpath("tmp"))
        prot_clusters = pandas.read_csv(
            workdir.joinpath("step3_cluster.tsv"),
            sep="\t",
            header=None,
            names=["protein_representative", "protein_id"]
        )

        # extract protein representatives
        prot_clusters["cluster_id"] = prot_clusters["protein_id"].str.rsplit("_", 1).str[0]
        protein_representatives = { x: i for i, x in enumerate(sorted(prot_clusters['protein_representative'].unique())) }
        p = len(protein_representatives)
        progress.console.print(f"[bold green]{'Found':>12}[/] {p} protein representatives for {len(prot_clusters)} proteins")

        # build weighted compositional array
        progress.console.print(f"[bold blue]{'Building':>12}[/] weighted compositional array")
        compositions = scipy.sparse.dok_matrix((r, p), dtype=numpy.int64) #scipy.sparse.dok_matrix((r, p), dtype=numpy.int32)
        for row in prot_clusters.itertuples():
            cluster_index = representatives[row.cluster_id]
            prot_index = protein_representatives[row.protein_representative]
            compositions[cluster_index, prot_index] += protein_sizes[row.protein_representative]

        # compute distances
        progress.console.print(f"[bold blue]{'Computing':>12}[/] pairwise distance based on protein composition")
        distance_matrix = pairwise_distances(
            compositions.tocsr(),
            metric="cityblock",
            n_jobs=args.jobs
        )

        # run clustering
        progress.console.print(f"[bold blue]{'Clustering':>12}[/] gene clusters using average linkage")
        Z = linkage(distance_matrix, method="average")
        flat = scipy.cluster.hierarchy.fcluster(Z, criterion="distance", t=0.1)

        # build GCFs based on
        gcfs3 = pandas.DataFrame({
            "gcf_id": [ f"{args.prefix}{i:07}" for i in flat ],
            "gcf2_representative": sorted(representatives),
        })
        progress.console.print(f"[bold green]{'Built':>12}[/] {len(gcfs3.gcf_id.unique())} GCFs from {len(sequence_lengths)} initial clusters")

        # build final GCF table
        gcfs = pandas.merge(
            pandas.merge(gcfs1, gcfs2, on="gcf1_representative"),
            gcfs3,
            on="gcf2_representative",
        )

        # save results
        gcfs = gcfs.sort_values("gcf_id")
        gcfs = gcfs[["cluster_id", "gcf_id", "gcf1_representative", "gcf2_representative"]]
        gcfs.to_csv(args.output, sep="\t", index=False)



