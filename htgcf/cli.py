import argparse
import collections
import contextlib
import csv
import functools
import gzip
import itertools
import io
import os
import pathlib
import tempfile
import subprocess
import typing
import multiprocessing.pool

import anndata
import gb_io
import rich
import numpy
import pandas
import scipy.sparse
from scipy.cluster.hierarchy import fcluster

try:
    from fastcluster import linkage as _linkage
    linkage = functools.partial(_linkage, preserve_input=False)
except ImportError:
    from scipy.cluster.hierarchy import linkage

try:
    import argcomplete
except ImportError as err:
    argcomplete = err

try:
    from rich_argparse import RichHelpFormatter as HelpFormatter
except ImportError:
    from argparse import HelpFormatter

from .mmseqs import MMSeqs, Database, Clustering
from ._manhattan import sparse_manhattan


_GZIP_MAGIC = b'\x1f\x8b'


_PARAMS_NUC1 = dict(
    e_value=0.001,
    sequence_identity=0.85,
    coverage=1,
    cluster_mode=0,
    coverage_mode=1,
    spaced_kmer_mode=0,
)

_PARAMS_NUC2 = dict(
    e_value=0.001,
    sequence_identity=0.6,
    coverage=0.5,
    cluster_mode=0,
    coverage_mode=0,
    spaced_kmer_mode=0,
)

_PARAMS_PROT = dict(
    e_value=0.001,
    coverage=0.9,
    coverage_mode=1,
    sequence_identity=0.5,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="htgcf",
        formatter_class=HelpFormatter
    )
    parser.add_argument(
        "-i",
        "--input",
        help="the GenBank files containing the clusters to process",
        action="append",
        type=pathlib.Path,
        default=[],
        required=True,
    )
    parser.add_argument(
        "-o",
        "--output",
        help="the name of the ouput file to generate",
        default=pathlib.Path("gcfs.tsv"),
        type=pathlib.Path,
    )
    parser.add_argument(
        "-C",
        "--compositions",
        help="a file where to write compositional data for GCF representatives",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-F",
        "--features",
        help="a file where to write protein features in FASTA format",
        type=pathlib.Path,
    )
    parser.add_argument(
        "-w", "--workdir", help="a folder to use for processing", metavar="TMP"
    )
    parser.add_argument(
        "-j",
        "--jobs",
        help="the number of threads to use",
        type=int,
        default=os.cpu_count(),
        metavar="N",
    )
    parser.add_argument(
        "--prefix", help="the prefix for GCF identifiers", default="GCF"
    )
    parser.add_argument(
        "--clustering-method",
        help="the hierarchical method to use for protein-level clustering",
        default="average",
        choices={
            "average",
            "single",
            "complete",
            "weighted",
            "centroid",
            "median",
            "ward"
        }
    )
    parser.add_argument(
        "--clustering-distance",
        help="the distance threshold after which to stop merging clusters",
        type=float,
        default=0.5,
    )
    return parser


def write_cluster(record: gb_io.Record, file: typing.TextIO) -> None:
    file.write(">{}\n".format(record.name))
    file.write(record.sequence.decode("ascii"))
    file.write("\n")


def extract_sequences(
    progress: rich.progress.Progress,
    inputs: typing.List[pathlib.Path],
    output: pathlib.Path,
) -> pandas.DataFrame:
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
                        write_cluster(record, dst)
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


def deduplicate_sequences(
    mmseqs: MMSeqs,
    input_path: pathlib.Path,
    output_prefix: pathlib.Path,
    tmpdir: pathlib.Path,
) -> None:
    db = Database.create(mmseqs, input_path)
    result = db.cluster(
        mmseqs, 
        output_prefix.with_suffix(".db"),
        **_PARAMS_NUC1,
    )
    subdb = result.to_subdb(
        mmseqs, 
        output_prefix.with_name(f"{output_prefix.name}_rep_seq.db")
    )
    return result.to_dataframe(mmseqs)
  

def cluster_sequences(
    mmseqs: MMSeqs,
    input_db: pathlib.Path,
    output_prefix: pathlib.Path,
    tmpdir: pathlib.Path,
) -> None:
    db = Database(input_db)
    result = db.cluster(
        mmseqs, 
        output_prefix.with_suffix(".db"),
        **_PARAMS_NUC2,
    )
    return result.to_dataframe(mmseqs)


def extract_proteins(
    progress: rich.progress.Progress,
    inputs: typing.List[pathlib.Path],
    output: pathlib.Path,
    representatives: typing.Container[str],
) -> typing.Dict[str, int]:
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
                            filter(lambda f: f.type == "CDS", record.features)
                        ):
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
    progress.console.print(
        f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(representatives)} nucleotide representative"
    )
    return protein_sizes


def cluster_proteins(
    mmseqs: MMSeqs,
    input_path: pathlib.Path,
    output_prefix: pathlib.Path,
    tmpdir: pathlib.Path,
) -> None:
    db = Database.create(mmseqs, input_path, input_path.with_suffix(".db"))
    result = db.cluster(
        mmseqs, 
        output_prefix.with_suffix(".db"),
        **_PARAMS_PROT,
    )
    return result.to_dataframe(mmseqs)


    # indb_path = input_path.with_suffix(".db")
    # outdb_path = 
    # # create input database (zero-copy, as we formatted the input as two-line FASTA)
    # mmseqs.run(
    #     "createdb",
    #     input_path,
    #     indb_path,
    #     dbtype=1,
    #     shuffle=0,
    #     createdb_mode=1,
    #     write_lookup=0,
    #     id_offset=0,
    #     compressed=0,
    # ).check_returncode()
    # # run clustering
    # mmseqs.run(
    #     "linclust",
    #     indb_path,
    #     outdb_path,
    #     tmpdir,
    #     e=0.001,
    #     cov_mode=1,
    #     c=0.9,
    #     min_seq_id=0.5,
    #     remove_tmp_files=1,
    # ).check_returncode()
    # # build `clusters.tsv` file, which is needed for the final tables
    # mmseqs.run(
    #     "createtsv",
    #     indb_path,
    #     indb_path,
    #     outdb_path,
    #     output_prefix.with_name(f"{output_prefix.name}_cluster.tsv"),
    # ).check_returncode()


def get_protein_representative(
    mmseqs: MMSeqs,
    input_path: pathlib.Path,
    output_prefix: pathlib.Path,
    fasta_path: pathlib.Path
) -> None:
    db = Database(input_path.with_suffix(".db"))
    result = Clustering(output_prefix.with_suffix(".db"), db)
    subdb = result.to_subdb(mmseqs, output_prefix.with_name(f"{output_prefix.name}_rep_seq.db"))
    subdb.to_fasta(mmseqs, path)


def make_compositions(
    progress: rich.progress.Progress,
    protein_clusters: pandas.DataFrame,
    representatives: typing.Dict[str, int],
    protein_representatives: typing.Dict[str, int],
    protein_sizes: typing.Dict[str, int],
) -> anndata.AnnData: #scipy.sparse.csr_matrix:
    compositions = scipy.sparse.dok_matrix(
        (len(representatives), len(protein_representatives)), dtype=numpy.int32
    )

    task = progress.add_task(description=f"[bold blue]{'Working':>9}[/]", total=len(protein_clusters))
    for row in progress.track(protein_clusters.itertuples(), task_id=task):
        cluster_index = representatives[row.cluster_id]
        prot_index = protein_representatives[row.protein_representative]
        compositions[cluster_index, prot_index] += protein_sizes[
            row.protein_representative
        ]
    progress.remove_task(task)

    sorted_representatives = sorted(representatives, key=representatives.__getitem__)
    sorted_protein_representatives = sorted(protein_representatives, key=protein_representatives.__getitem__)
    return anndata.AnnData(
        X=compositions.tocsr(),
        dtype=numpy.int32,
        obs=pandas.DataFrame(index=pandas.Index(sorted_representatives, name="cluster_id")),
        var=pandas.DataFrame(
            index=pandas.Index(sorted_protein_representatives, name="protein_id"),
            data=dict(size=[protein_sizes[x] for x in sorted_protein_representatives]),
        )
    )


def compute_distances(
    progress: rich.progress.Progress,
    compositions: scipy.sparse.csr_matrix,
    jobs: typing.Optional[int],
) -> numpy.ndarray:
    n = 0
    r = compositions.shape[0]
    # compute the number of amino acids per cluster
    clusters_aa = numpy.zeros(r, dtype=numpy.int32)
    clusters_aa[:] = compositions.sum(axis=1).A1
    # compute manhattan distance on sparse matrix
    distance_vector = numpy.zeros(r*(r-1) // 2, dtype=numpy.double)
    sparse_manhattan(
        compositions.data,
        compositions.indices,
        compositions.indptr,
        distance_vector,
        threads=jobs,
    )
    # ponderate by sum of amino-acid distance
    for i in range(r-1):
        l = r - (i+1)
        distance_vector[n:n+l] /= (clusters_aa[i+1:] + clusters_aa[i]).clip(min=1)
        n += l
    # check distances are in [0, 1]
    return numpy.clip(distance_vector, 0.0, 1.0, out=distance_vector)


def main(argv: typing.Optional[typing.List[str]] = None) -> int:
    # build parser and get arguments
    parser = build_parser()
    if not isinstance(argcomplete, ImportError):
        argcomplete.autocomplete(parser)
    args = parser.parse_args()

    with contextlib.ExitStack() as ctx:
        # prepare progress bar
        progress = ctx.enter_context(
            rich.progress.Progress(
                "",
                rich.progress.SpinnerColumn(),
                *rich.progress.Progress.get_default_columns(),
            )
        )
        mmseqs = MMSeqs(progress=progress, threads=args.jobs)

        # open temporary folder
        if args.workdir is None:
            workdir = pathlib.Path(ctx.enter_context(tempfile.TemporaryDirectory()))
        else:
            workdir = pathlib.Path(args.workdir)
            workdir.mkdir(parents=True, exist_ok=True)

        # extract raw sequences
        clusters_fna = workdir.joinpath("clusters.fna")
        progress.console.print(f"[bold blue]{'Loading':>12}[/] input clusters")
        input_sequences = extract_sequences(progress, args.input, clusters_fna)
        progress.console.print(
            f"[bold green]{'Loaded':>12}[/] {len(input_sequences)} clusters to process"
        )

        # deduplicate fragments
        progress.console.print(
            f"[bold blue]{'Starting':>12}[/] nucleotide deduplication step with [purple]mmseqs[/]"
        )
        gcfs1 = (
            deduplicate_sequences(
                mmseqs, 
                clusters_fna, 
                workdir.joinpath("step1"), 
                workdir.joinpath("tmp")
            ).rename(columns={
                "representative": "fragment_representative",
                "id": "cluster_id",
            }).sort_values("cluster_id")
        )
        progress.console.print(
            f"[bold green]{'Reduced':>12}[/] {len(gcfs1)} clusters to {len(gcfs1.fragment_representative.unique())} complete representatives"
        )

        # cluster sequences
        progress.console.print(
            f"[bold blue]{'Starting':>12}[/] nucleotide clustering step with [purple]mmseqs[/]"
        )
        gcfs2 = cluster_sequences(
            mmseqs,
            workdir.joinpath("step1_rep_seq.db"),
            workdir.joinpath("step2"),
            workdir.joinpath("tmp"),
        ).rename(columns={
            "representative": "nucleotide_representative",
            "id": "fragment_representative",
        }).sort_values("fragment_representative")
        progress.console.print(
            f"[bold green]{'Reduced':>12}[/] {len(gcfs2)} clusters to {len(gcfs2.nucleotide_representative.unique())} nucleotide representatives"
        )

        # load representatives
        progress.console.print(
            f"[bold blue]{'Extracting':>12}[/] representative clusters"
        )
        representatives = {
            x: i
            for i, x in enumerate(sorted(gcfs2["nucleotide_representative"].unique()))
        }
        progress.console.print(
            f"[bold green]{'Loaded':>12}[/] {len(representatives)} nucleotide representative clusters"
        )

        # extract proteins and record sizes
        proteins_faa = workdir.joinpath("proteins.faa")
        progress.console.print(
            f"[bold blue]{'Extracting':>12}[/] protein sequences from clusters"
        )
        protein_sizes = extract_proteins(
            progress, args.input, proteins_faa, representatives
        )

        # cluster proteins
        prot_clusters = cluster_proteins(
            mmseqs, proteins_faa, workdir.joinpath("step3"), workdir.joinpath("tmp")
        ).rename(columns={
            "representative": "protein_representative",
            "id": "protein_id",
        }).sort_values("protein_id")

        # extract protein representatives
        prot_clusters["cluster_id"] = (
            prot_clusters["protein_id"].str.rsplit("_", 1).str[0]
        )
        protein_representatives = {
            x: i
            for i, x in enumerate(
                sorted(prot_clusters["protein_representative"].unique())
            )
        }
        progress.console.print(
            f"[bold green]{'Found':>12}[/] {len(protein_representatives)} protein representatives for {len(prot_clusters)} proteins"
        )

        # build weighted compositional array
        progress.console.print(
            f"[bold blue]{'Building':>12}[/] weighted compositional array"
        )
        compositions = make_compositions(
            progress, prot_clusters, representatives, protein_representatives, protein_sizes
        )

        # compute and ponderate distances
        progress.console.print(
            f"[bold blue]{'Computing':>12}[/] pairwise distance based on protein composition"
        )
        distance_vector = compute_distances(progress, compositions.X, args.jobs)

        # run hierarchical clustering
        progress.console.print(
            f"[bold blue]{'Clustering':>12}[/] gene clusters using average linkage"
        )
        Z = linkage(distance_vector, method=args.clustering_method)
        flat = fcluster(Z, criterion="distance", t=args.clustering_distance)

        # build GCFs based on flat clustering
        gcfs3 = pandas.DataFrame(
            {
                "gcf_id": [f"{args.prefix}{i:07}" for i in flat],
                "nucleotide_representative": sorted(representatives),
            }
        )
        progress.console.print(
            f"[bold green]{'Built':>12}[/] {len(gcfs3.gcf_id.unique())} GCFs from {len(input_sequences)} initial clusters"
        )

        # extract protein representative using the largest cluster of each GCF
        gcf3_representatives = (
            pandas.merge(
                gcfs3,
                input_sequences["cluster_length"],
                left_on="nucleotide_representative",
                right_index=True
            )
            .sort_values("cluster_length")
            .drop_duplicates("gcf_id", keep="last")
            .set_index("gcf_id")
        )
        gcfs3 = pandas.merge(
            gcfs3,
            gcf3_representatives["nucleotide_representative"].rename("gcf_representative"),
            left_on="gcf_id",
            right_index=True,
        )

        # build final GCF table
        gcfs = pandas.merge(
            pandas.merge(
                pandas.merge(gcfs1, gcfs2, on="fragment_representative"),
                gcfs3,
                on="nucleotide_representative",
            ),
            input_sequences,
            left_on="cluster_id",
            right_index=True,
        )

        # save results
        gcfs.sort_values(["gcf_id", "cluster_length"], inplace=True)
        gcfs = gcfs[
            [
                "cluster_id",
                "cluster_length",
                "gcf_id",
                "gcf_representative",
                "nucleotide_representative",
                "fragment_representative",
                "filename"
            ]
        ]
        gcfs.to_csv(args.output, sep="\t", index=False)
        progress.console.print(
            f"[bold green]{'Saved':>12}[/] final GCFs table to {str(args.output)!r}"
        )

        # save compositions
        if args.compositions is not None:
            gcf_representatives = gcfs["gcf_representative"].unique()
            representatives_compositions = anndata.AnnData(
                X=compositions[gcf_representatives].X,
                dtype=compositions.X.dtype,
                var=compositions.var,
                obs=(
                    gcfs.set_index("cluster_id")
                        .loc[gcf_representatives, ["gcf_id", "gcf_representative"]]
                        .set_index("gcf_id")
                )
            )
            representatives_compositions.write(args.compositions)
            progress.console.print(
                f"[bold green]{'Saved':>12}[/] GCF compositions to {str(args.compositions)!r}"
            )

        # save protein features
        if args.features is not None:
            get_protein_representative(
                mmseqs,
                proteins_faa,
                workdir.joinpath("step3"),
                args.features,
            )
            progress.console.print(
                f"[bold green]{'Saved':>12}[/] protein features to {str(args.features)!r}"
            )


    return 0