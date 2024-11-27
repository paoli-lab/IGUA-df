import io
import typing
import pathlib
import gzip
import warnings

import Bio.Seq
import gb_io
import pandas
import rich.progress

_GZIP_MAGIC = b'\x1f\x8b'


def write_fasta(file: typing.TextIO, name: str, sequence: str) -> None:
    file.write(">{}\n".format(name))
    file.write(sequence)
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
                        write_fasta(dst, record.name, record.sequence.decode("ascii"))
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


def translate_orf(sequence: typing.Union[str, bytes], translation_table: int = 11) -> str:
    return str(Bio.Seq.Seq(sequence).translate(translation_table))

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
                            filter(lambda f: f.kind == "CDS", record.features)
                        ):
                            qualifier = next((qualifier for qualifier in feat.qualifiers if qualifier.key == "translation"), None)
                            if qualifier is None:
                                rich.print(f"[bold yellow]{'Warning':>12}[/] no 'translation' qualifier found in CDS feature of {record.name!r}")
                                translation = translate_orf(record.sequence[feat.location.start:feat.location.end])
                            else:
                                translation = qualifier.value.rstrip("*")
                            protein_id = "{}_{}".format(record.name, i)
                            if protein_id not in protein_sizes:
                                write_fasta(dst, protein_id, translation)
                                protein_sizes[protein_id] = len(translation)
            progress.remove_task(task)
    progress.console.print(
        f"[bold green]{'Extracted':>12}[/] {len(protein_sizes)} proteins from {len(representatives)} nucleotide representative"
    )
    return protein_sizes
