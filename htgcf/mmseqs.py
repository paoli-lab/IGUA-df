import subprocess

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

def wrap_progress(
    progress: rich.progress.Progress, 
    process: subprocess.Popen
):
    """Wrap the progress output from ``mmseqs`` into a `rich` progress bar. 
    """
    
    buffer = bytearray()
    command = ""
    task = progress.add_task(f"[bold blue]{'Running':>12}[/] [purple]{command}[/]", total=65) 
    bar_column = next(c for c in progress.columns if isinstance(c, rich.progress.BarColumn))
    
    for x in iter(lambda: process.stdout.read(1), b""):
        buffer.append(ord(x))
        if buffer.startswith(b"["):
            # update progress
            progress.update(task_id=task, completed=buffer.count(b'='))
        if buffer.endswith(b"\n"):
            # extract current command being run
            _command = next(iter(buffer.split()), b"").decode()
            if _command in COMMANDS:
                command = _command
                bar_column.bar_width = 40 - len(command)
                progress.reset(task_id=task, description=f"[bold blue]{'Running':>12}[/] [purple]{command}[/]")
            # clear current buffer
            buffer.clear()
        
    progress.update(task_id=task, visible=False)
    progress.remove_task(task)
    return process.wait()


def run_mmseqs(
    progress,
    command,
    *args,
    **kwargs,
):
    # build actual command line
    params = ["mmseqs", command, *args]
    for k, v in kwargs.items():
        dash = "-" if len(k) == 1 else "--"
        flag = "".join([dash, k.replace("_", "-")])
        params.append(flag)
        params.append(repr(v))

    # invoke mmseqs subprocess
    process = subprocess.Popen(params, stdout=subprocess.PIPE, bufsize=0)
    wrap_progress(progress, process)

    # check result
    if process.returncode != 0:
        raise RuntimeError(f"mmseqs failed with return code {process.returncode}")
        
    

