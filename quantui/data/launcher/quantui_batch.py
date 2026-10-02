#!/usr/bin/env python3
"""quantui-batch: submit and follow QuantUI calculations from a Slurm login node.

Installed by ``quantui install-launcher`` (run inside the QuantUI Apptainer
image). This file runs on the HOST, outside the image, so it uses only the
Python 3.6+ standard library and the Slurm commands. QuantUI itself is only
ever run inside the image:

  1. ``quantui submit --prepare-only`` (inside the image) turns an .xyz file or
     a request JSON into a job folder: request.json + submit.slurm, with
     cores/memory/time from QuantUI's own estimator.
  2. ``sbatch submit.slurm`` (here, on the host) queues it.

Results land in <job folder>/attempt-NN_job<id>/ and show up in the QuantUI
app's History the next time the app starts.

Commands (run ``quantui-batch help`` for the full text):

  submit FILE... [options]    prepare and queue one job per file
  estimate FILE... [options]  show the cores/memory/time a job would get
  status [-n N | --all]       list recent jobs and what each is doing
  log JOB [-f]                show (or follow) a job's live.log
  rerun JOB [sbatch options]  run a job again, e.g. with --mem=64G
  cancel JOB                  cancel a queued or running job
  path JOB                    print a job's folder
"""

import getpass
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Filled in by `quantui install-launcher`; environment variables win.
DEFAULT_IMAGE = "@QUANTUI_IMAGE@"
DEFAULT_STAGING = "@QUANTUI_STAGING_DIR@"
INSTALLED_FROM = "@QUANTUI_VERSION@"

IMAGE = os.environ.get("QUANTUI_BATCH_IMAGE") or DEFAULT_IMAGE
STAGING = Path(
    os.environ.get("QUANTUI_STAGING_DIR")
    or (
        DEFAULT_STAGING if not DEFAULT_STAGING.startswith("@") else "~/.quantui/staging"
    )
).expanduser()
JOB_LOG = ".quantui-batch-jobs"  # one "<slurm id>\t<UTC time>" line per sbatch
ATTEMPT_RE = re.compile(r"^attempt-(\d+)_job(\w+)$")
ACTIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED")

HELP = """\
quantui-batch: run QuantUI calculations as Slurm batch jobs.

  quantui-batch submit FILE... [options]
      FILE is an .xyz file or a QuantUI request .json. One job per file.
      Options for .xyz files (--calc is required):
        --calc TYPE        single_point, geometry_opt, frequency, tddft, nmr,
                           pes_scan, reorganization_energy
        --method NAME      e.g. B3LYP          --basis NAME   e.g. def2-SVP
        --charge N         default 0           --mult N       default 1
        --solvent NAME     e.g. Water          --preopt       optimize first
        --option KEY=VALUE extra option, repeatable (e.g. --option nstates=10)
      Options for any file:
        --job-name NAME    folder and job name (repeats get _2, _3, ...)
        --cores N  --memory-gb N  --walltime HH:MM:SS   override the estimate
        --email ADDR       get an email when the job ends
  quantui-batch estimate FILE... [options]
      Print the cores, memory and time a job would get. Submits nothing.
  quantui-batch status [-n N | --all]
      Recent jobs (default 10), newest first, with what each is doing.
  quantui-batch log JOB [-f]
      Last 40 lines of the job's live.log; -f keeps following it.
  quantui-batch rerun JOB [sbatch options]
      Run the job again as a new attempt (earlier attempts are kept).
      Example after running out of memory: quantui-batch rerun JOB --mem=64G
  quantui-batch cancel JOB
  quantui-batch path JOB
      JOB is the job's folder name (as shown by status), or a path to it.

Image:       {image}
Job folders: {staging}
At most {limit} of your QuantUI jobs may be queued or running at once
(set QUANTUI_MAX_CONCURRENT_JOBS to change).
"""


def die(msg, code=1):
    sys.stderr.write("quantui-batch: " + msg + "\n")
    sys.exit(code)


def max_jobs():
    try:
        return max(1, int(os.environ.get("QUANTUI_MAX_CONCURRENT_JOBS", "2")))
    except ValueError:
        return 2


def run(cmd, check=False, capture=True, env=None):
    """subprocess.run with text output (3.6-compatible spelling)."""
    try:
        return subprocess.run(
            cmd,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.PIPE if capture else None,
            universal_newlines=True,  # noqa: UP021 — `text=` needs 3.7; host may be 3.6
            check=check,
            env=env,
        )
    except FileNotFoundError:
        die("command not found: " + cmd[0])


# --------------------------------------------------------------------------
# Slurm queries
# --------------------------------------------------------------------------


def queue():
    """{job id: (state, script path)} for this user's queued/running jobs."""
    if shutil.which("squeue") is None:
        die("squeue not found: run quantui-batch on the cluster login node")
    out = run(["squeue", "-u", getpass.getuser(), "-h", "-o", "%i|%T|%o"])
    jobs = {}
    for line in (out.stdout or "").splitlines():
        parts = line.strip().split("|", 2)
        if len(parts) == 3:
            jobs[parts[0]] = (parts[1], parts[2])
    return jobs


def active_quantui_jobs(q):
    """Count queued/running jobs whose script lives under the job-folder root."""
    roots = {str(STAGING), str(STAGING.resolve()) if STAGING.exists() else str(STAGING)}
    return sum(
        1
        for state, script in q.values()
        if state in ACTIVE_STATES and any(script.startswith(r + os.sep) for r in roots)
    )


def accounting(job_id):
    """(state, elapsed, max memory) for a finished job, from sacct."""
    if shutil.which("sacct") is None:
        return ("UNKNOWN", "", "")
    out = run(["sacct", "-j", job_id, "-n", "-P", "-o", "JobID,State,Elapsed,MaxRSS"])
    state, elapsed, maxrss = "UNKNOWN", "", ""
    for line in (out.stdout or "").splitlines():
        cols = line.split("|")
        if len(cols) < 4:
            continue
        if cols[0] == job_id:
            state = cols[1].split()[0] if cols[1] else state
            elapsed = cols[2]
        if cols[3] and _rss_bytes(cols[3]) > _rss_bytes(maxrss):
            maxrss = cols[3]
    return (state, elapsed, maxrss)


def _rss_bytes(text):
    m = re.match(r"^([\d.]+)([KMGT]?)", text or "")
    if not m:
        return 0.0
    scale = {"": 1, "K": 1e3, "M": 1e6, "G": 1e9, "T": 1e12}[m.group(2)]
    return float(m.group(1)) * scale


# --------------------------------------------------------------------------
# Job folders
# --------------------------------------------------------------------------


def job_ids(job_dir):
    path = job_dir / JOB_LOG
    if not path.exists():
        return []
    return [ln.split("\t")[0] for ln in path.read_text().splitlines() if ln.strip()]


def record_job(job_dir, job_id):
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    with open(str(job_dir / JOB_LOG), "a") as fh:
        fh.write(f"{job_id}\t{stamp}\n")


def attempts(job_dir):
    """[(number, slurm id, path)] oldest first."""
    found = []
    for child in job_dir.iterdir():
        m = ATTEMPT_RE.match(child.name)
        if m and child.is_dir() and not child.is_symlink():
            found.append((int(m.group(1)), m.group(2), child))
    return sorted(found)


def all_jobs():
    if not STAGING.is_dir():
        return []
    dirs = [d for d in STAGING.iterdir() if (d / "submit.slurm").is_file()]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def resolve_job(name):
    path = Path(name).expanduser()
    if (path / "submit.slurm").is_file():
        return path
    exact = STAGING / name
    if (exact / "submit.slurm").is_file():
        return exact
    matches = [d for d in all_jobs() if d.name.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        die(f"no job folder named {name!r} under {STAGING} (see: quantui-batch status)")
    die(
        "{!r} matches several jobs: {}".format(
            name, ", ".join(d.name for d in matches[:8])
        )
    )


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def describe(job_dir, q):
    """(slurm id, state, detail) for one job folder."""
    tried = attempts(job_dir)
    # Jobs queued by the app (or a bare sbatch) have no JOB_LOG; their
    # attempt folders still name the Slurm job that made them.
    ids = job_ids(job_dir) or [jid for _n, jid, _p in tried]
    last = ids[-1] if ids else ""
    by_id = {jid: path for _n, jid, path in tried}

    if last and last in q:
        state = q[last][0]
        detail = ""
        progress = read_json(by_id[last] / "progress.json") if last in by_id else None
        if progress:
            pct = progress.get("percent")
            detail = progress.get("message") or progress.get("stage") or ""
            if pct is not None:
                detail = f"{float(pct):.0f}%  {detail}"
        elif state == "PENDING":
            detail = "waiting for a free node"
        return (last, state, detail)

    done = [p for _n, _jid, p in tried if (p / "result.json").is_file()]
    if last and (last not in by_id or not (by_id[last] / "result.json").is_file()):
        state, elapsed, maxrss = accounting(last)
        if state not in ("COMPLETED", "UNKNOWN") or not done:
            return (last, state, hint(job_dir, state, elapsed, maxrss))
    if done:
        result = read_json(done[-1] / "result.json") or {}
        detail = "results in " + done[-1].name
        if result.get("converged") is False:
            return (last, "DONE", "did not fully converge; " + detail)
        return (last, "DONE", detail)
    if not ids:
        return ("", "PREPARED", "not submitted: quantui-batch rerun " + job_dir.name)
    return (last, "UNKNOWN", "")


def hint(job_dir, state, elapsed, maxrss):
    name = job_dir.name
    if state == "OUT_OF_MEMORY":
        used = f" (used {maxrss})" if maxrss else ""
        return f"ran out of memory{used}: quantui-batch rerun {name} --mem=<more>G"
    if state == "TIMEOUT":
        return "hit its time limit after {}: quantui-batch rerun {} --time=<longer>".format(
            elapsed or "?", name
        )
    if state in ("FAILED", "NODE_FAIL"):
        return "see: quantui-batch log " + name
    if state == "CANCELLED":
        return "cancelled"
    return ""


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def image_or_die():
    if not IMAGE or IMAGE.startswith("@"):
        die(
            "no image configured: set QUANTUI_BATCH_IMAGE or reinstall with "
            "`apptainer exec IMAGE quantui install-launcher`"
        )
    if not Path(IMAGE).exists():
        die("image not found: " + IMAGE)
    return IMAGE


def in_image(args):
    """Command line that runs `quantui ARGS` inside the image."""
    image = image_or_die()
    binds = []
    home = str(Path.home())
    cwd = os.getcwd()
    if not cwd.startswith(home):
        binds += ["--bind", cwd]
    if not str(STAGING).startswith(home):
        binds += ["--bind", str(STAGING)]
    cmd = ["apptainer", "exec"] + binds + [image, "quantui"] + args
    if prepare_via_srun():
        # No apptainer on this node: do the (seconds-long) prepare step in a
        # tiny allocation instead.
        cmd = [
            "srun",
            "--quiet",
            "--partition=" + os.environ.get("QUANTUI_SLURM_PARTITION", "common"),
            "--cpus-per-task=1",
            "--mem=2G",
            "--time=00:05:00",
        ] + cmd
    return cmd


def sbatch_env():
    """The environment for sbatch, minus the caller's own job variables.

    Submitting from inside a job (an OnDemand Shell session is one) would
    otherwise hand that job's SLURM_* values to the new job: sbatch exports
    the whole environment, and Slurm only overwrites the variables the new
    job's own options set. An inherited SLURM_CPUS_PER_TASK=2 would then set
    OMP_NUM_THREADS for a 16-core calculation.
    """
    # SBATCH_* input variables (e.g. SBATCH_ACCOUNT) are the user's own
    # defaults, never set by Slurm inside a job, so they are kept.
    return {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}


def prepare_via_srun():
    return (
        os.environ.get("QUANTUI_BATCH_PREPARE", "") == "srun"
        or shutil.which("apptainer") is None
    )


def child_env():
    """Environment for the prepare step that runs QuantUI inside the image.

    One BLAS/OpenMP thread: the step only parses a file and writes two, and a
    login node's per-user thread limit can make numpy fail to import when
    OpenBLAS starts one thread per core. When the step goes through srun, the
    caller's SLURM_* variables are dropped so srun makes its own small
    allocation instead of a step inside the caller's job.
    """
    env = sbatch_env() if prepare_via_srun() else dict(os.environ)
    env["QUANTUI_STAGING_DIR"] = str(STAGING)
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[var] = "1"
    return env


PREPARE_FAILED_HINT = (
    "quantui-batch: the prepare step failed. If the message above is about "
    "threads, memory or a missing command, run from an OnDemand Shell session, "
    "or retry with QUANTUI_BATCH_PREPARE=srun (does the step on a compute node)."
)


def cmd_submit(argv):
    if not argv:
        die(
            "usage: quantui-batch submit FILE... [options]  (see: quantui-batch help)",
            2,
        )
    q = queue()
    active = active_quantui_jobs(q)
    limit = max_jobs()
    if active >= limit:
        die(
            f"you already have {active} QuantUI job(s) queued or running (limit {limit}). "
            "Wait for one to finish (quantui-batch status) or cancel one."
        )
    extra = [] if "--apptainer-image" in argv else ["--apptainer-image", IMAGE]
    cmd = in_image(["submit", "--prepare-only"] + extra + argv)
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        universal_newlines=True,  # noqa: UP021
        env=child_env(),
    )
    scripts = [Path(ln) for ln in (proc.stdout or "").splitlines() if ln.strip()]
    if proc.returncode != 0 and not scripts:
        sys.stderr.write(PREPARE_FAILED_HINT + "\n")
    for script in scripts:
        job_dir = script.parent
        if active >= limit:
            print(
                f"not submitted {job_dir.name} (limit of {limit} jobs reached); later run: "
                f"quantui-batch rerun {job_dir.name}"
            )
            continue
        out = run(["sbatch", "--parsable", str(script)], env=sbatch_env())
        if out.returncode != 0:
            sys.stderr.write(out.stderr or "")
            print(f"not submitted {job_dir.name}: sbatch failed")
            proc.returncode = proc.returncode or 1
            continue
        job_id = out.stdout.strip().split(";")[0]
        record_job(job_dir, job_id)
        active += 1
        print(f"submitted {job_dir.name}  (Slurm job {job_id})")
    if scripts:
        print("Check on it with: quantui-batch status")
    return proc.returncode


def cmd_estimate(argv):
    if not argv:
        die("usage: quantui-batch estimate FILE... [options]", 2)
    return subprocess.run(
        in_image(["submit", "--dry-run"] + argv), env=child_env()
    ).returncode


def cmd_status(argv):
    limit = 10
    if "--all" in argv or "-a" in argv:
        limit = None
    elif "-n" in argv:
        try:
            limit = int(argv[argv.index("-n") + 1])
        except (IndexError, ValueError):
            die("usage: quantui-batch status [-n N | --all]", 2)
    jobs = all_jobs()
    if not jobs:
        print(f"No QuantUI jobs under {STAGING} yet.")
        return 0
    q = queue()
    rows = [(d.name,) + describe(d, q) for d in jobs[:limit]]
    widths = [
        max(len(r[i]) for r in rows + [("JOB", "SLURM ID", "STATE", "")])
        for i in range(3)
    ]

    def line(cols):
        return "  ".join(c.ljust(w) for c, w in zip(cols[:3], widths)) + "  " + cols[3]

    print(line(("JOB", "SLURM ID", "STATE", "DETAIL")))
    for row in rows:
        print(line(row).rstrip())
    hidden = len(jobs) - len(rows)
    if hidden > 0:
        print(f"({hidden} older job(s) not shown; use --all)")
    print(
        f"{active_quantui_jobs(q)} of {max_jobs()} allowed QuantUI jobs queued or running."
    )
    return 0


def latest_log(job_dir):
    tried = attempts(job_dir)
    if not tried:
        return None
    return tried[-1][2] / "live.log"


def cmd_log(argv):
    if not argv:
        die("usage: quantui-batch log JOB [-f]", 2)
    job_dir = resolve_job([a for a in argv if a != "-f"][0])
    log = latest_log(job_dir)
    if log is None or not log.exists():
        print(f"{job_dir.name} has not started yet (no live.log).")
        return 0
    print(f"== {log}")
    tail = ["tail", "-n", "40"] + (["-f"] if "-f" in argv else []) + [str(log)]
    try:
        return subprocess.run(tail).returncode
    except KeyboardInterrupt:
        return 0


def cmd_rerun(argv):
    if not argv:
        die("usage: quantui-batch rerun JOB [sbatch options]", 2)
    job_dir = resolve_job(argv[0])
    q = queue()
    for jid in job_ids(job_dir):
        if jid in q and q[jid][0] in ACTIVE_STATES:
            die(
                f"{job_dir.name} is still {q[jid][0].lower()} (job {jid}); cancel it first or wait."
            )
    if active_quantui_jobs(q) >= max_jobs():
        die(
            f"you already have {active_quantui_jobs(q)} QuantUI job(s) queued or running (limit {max_jobs()})."
        )
    out = run(
        ["sbatch", "--parsable"] + argv[1:] + [str(job_dir / "submit.slurm")],
        env=sbatch_env(),
    )
    if out.returncode != 0:
        sys.stderr.write(out.stderr or "")
        return out.returncode
    job_id = out.stdout.strip().split(";")[0]
    record_job(job_dir, job_id)
    print(
        f"resubmitted {job_dir.name}  (Slurm job {job_id}; earlier attempts are kept, and an "
        "interrupted optimization resumes from its checkpoint)"
    )
    return 0


def cmd_cancel(argv):
    if not argv:
        die("usage: quantui-batch cancel JOB", 2)
    job_dir = resolve_job(argv[0])
    q = queue()
    live = [jid for jid in job_ids(job_dir) if jid in q]
    if not live:
        print(f"{job_dir.name} has no queued or running job.")
        return 0
    return run(["scancel"] + live, capture=False).returncode


def cmd_path(argv):
    if not argv:
        die("usage: quantui-batch path JOB", 2)
    print(resolve_job(argv[0]))
    return 0


COMMANDS = {
    "submit": cmd_submit,
    "estimate": cmd_estimate,
    "status": cmd_status,
    "log": cmd_log,
    "rerun": cmd_rerun,
    "cancel": cmd_cancel,
    "path": cmd_path,
}


def main(argv):
    if not argv or argv[0] in ("help", "-h", "--help"):
        print(HELP.format(image=IMAGE, staging=STAGING, limit=max_jobs()), end="")
        return 0
    if argv[0] in ("--version", "version"):
        print(f"quantui-batch (installed from QuantUI {INSTALLED_FROM})")
        return 0
    command = COMMANDS.get(argv[0])
    if command is None:
        die(f"unknown command {argv[0]!r} (see: quantui-batch help)", 2)
    return command(argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
