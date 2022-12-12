import os
import subprocess
import typing

import rich.progress


COMMANDS = {
  "easy-search",
  "easy-linsearch",
  "easy-cluster",
  "easy-linclust",
  "easy-taxonomy",
  "easy-rbh",
  "search",
  "linsearch",
  "map",
  "rbh",
  "linclust",
  "cluster",
  "clusterupdate",
  "taxonomy",
  "databases",
  "createdb",
  "createindex",
  "createlinindex",
  "convertmsa",
  "tsv2db",
  "tar2db",
  "msa2profile",
  "compress",
  "decompress",
  "rmdb",
  "mvdb",
  "cpdb",
  "lndb",
  "unpackdb",
  "touchdb",
  "createsubdb",
  "concatdbs",
  "mergedbs",
  "subtractdbs",
  "convertalis",
  "createtsv",
  "convert2fasta",
  "result2flat",
  "createseqfiledb",
  "taxonomyreport",
  "extractorfs",
  "extractframes",
  "orftocontig",
  "reverseseq",
  "translatenucs",
  "translateaa",
  "splitsequence",
  "masksequence",
  "extractalignedregion",
  "swapresults",
  "result2rbh",
  "result2msa",
  "result2dnamsa",
  "result2stats",
  "filterresult",
  "offsetalignment",
  "proteinaln2nucl",
  "result2repseq",
  "sortresult",
  "summarizealis",
  "summarizeresult",
  "createtaxdb",
  "createbintaxonomy",
  "addtaxonomy",
  "taxonomyreport",
  "filtertaxdb",
  "filtertaxseqdb",
  "aggregatetax",
  "aggregatetaxweights",
  "lcaalign",
  "lca",
  "majoritylca",
  "multihitdb",
  "multihitsearch",
  "besthitperset",
  "combinepvalperset",
  "mergeresultsbyset",
  "prefilter",
  "ungappedprefilter",
  "kmermatcher",
  "kmersearch",
  "align",
  "alignall",
  "transitivealign",
  "rescorediagonal",
  "alignbykmer",
  "clust",
  "clusthash",
  "mergeclusters",
  "result2profile",
  "msa2result",
  "msa2profile",
  "profile2pssm",
  "profile2consensus",
  "profile2repseq",
  "convertprofiledb",
  "enrich",
  "result2pp",
  "profile2cs",
  "convertca3m",
  "expandaln",
  "expand2profile",
  "view",
  "apply",
  "filterdb",
  "swapdb",
  "prefixid",
  "suffixid",
  "renamedbkeys",
  "diffseqdbs",
  "summarizetabs",
  "gff2db",
  "maskbygff",
  "convertkb",
  "summarizeheaders",
  "nrtotaxmapping",
  "extractdomains",
  "countkmer",
}

class MMSeqs:

    def __init__(self, binary="mmseqs", progress=None, threads=None):
        self.binary = binary
        self.progress = progress
        self.threads = threads

    def _wrap_progress(
        self,
        process: subprocess.Popen,
    ) -> int:
        """Wrap the progress output from ``mmseqs`` into a `rich` progress bar.
        """
        assert self.progress is not None
        assert process.stdout is not None

        buffer = bytearray()
        command = ""
        task = self.progress.add_task(f"[bold blue]{'Running':>9}[/] [purple]{command}[/]", total=65)
        bar_column = next(c for c in self.progress.columns if isinstance(c, rich.progress.BarColumn))

        for x in iter(lambda: process.stdout.read(1), b""):
            buffer.append(ord(x))
            if buffer.startswith(b"["):
                # update progress
                self.progress.update(task_id=task, completed=buffer.count(b'='))
            if buffer.endswith(b"\n"):
                # extract current command being run
                _command = next(iter(buffer.split()), b"").decode()
                if _command in COMMANDS:
                    command = _command
                    bar_column.bar_width = 40 - len(command)
                    self.progress.reset(task_id=task, description=f"[bold blue]{'Running':>9}[/] [purple]{command}[/]")
                # clear current buffer
                buffer.clear()

        self.progress.update(task_id=task, visible=False)
        self.progress.remove_task(task)
        return process.wait()

    def run(
        self,
        command: str,
        *args: typing.Union[str, bytes, os.PathLike],
        **kwargs: object
    ) -> subprocess.CompletedProcess:
        """Run an arbitrary ``mmseqs`` command.
        """
        # build actual command line
        params = [self.binary, command, *args]
        for k, v in kwargs.items():
            dash = "-" if len(k) == 1 else "--"
            flag = "".join([dash, k.replace("_", "-")])
            params.append(flag)
            params.append(repr(v))

        # use fixed number of threads
        if self.threads is not None and (command == "easy-linclust" or command == "search"):
            params.extend(["--threads", str(self.threads)])

        # start mmseqs subprocess
        process = subprocess.Popen(params, stdout=subprocess.PIPE, bufsize=0)

        # wrap progress if a rich progress bar is available
        if self.progress:
            self._wrap_progress(process)

        # return a completed process instance for compatibility
        return subprocess.CompletedProcess(
            params,
            process.returncode
        )

