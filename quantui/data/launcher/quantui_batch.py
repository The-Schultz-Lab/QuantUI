#!/usr/bin/env python3
"""quantui-batch: submit and follow QuantUI calculations from a Slurm login node.

Installed by ``quantui install-launcher``, run inside the QuantUI Apptainer
image. This file runs on the HOST (an SSH session on the login node), so it
uses only the Python 3.6+ standard library and the Slurm commands. It never
starts the image and never imports QuantUI: on a login node, starting the
image and importing QuantUI's scientific stack is slow, counts as computing
there, and has failed outright (numpy against the per-user thread limit).

Instead it writes the same job folder QuantUI's own ``SlurmBackend.prepare()``
writes (request.json, submit.slurm, and the job record the app reads), using
constants copied from the installing QuantUI (SITE below) and a port of the
little logic involved: XYZ parsing, the charge/multiplicity check, the
resource estimate, job naming and the batch script. The calculation itself
runs inside the image on a compute node, through ``submit.slurm``.
tests/test_batch_submit.py keeps the port byte-for-byte equal to QuantUI.

Commands (``quantui-batch help`` for the full text):

  submit FILE... [options]    prepare and queue one job per file
  estimate FILE... [options]  show the cores/memory/time a job would get
  status [-n N | --all]       list recent jobs and what each is doing
  log JOB [-f]                show (or follow) a job's live.log
  rerun JOB [sbatch options]  run a job again, e.g. with --mem=64G
  cancel JOB                  cancel a queued or running job
  path JOB                    print a job's folder
"""

import argparse
import getpass
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# Filled in by `quantui install-launcher`.
DEFAULT_IMAGE = "@QUANTUI_IMAGE@"
IMAGE_PYTHON = "@QUANTUI_IMAGE_PYTHON@"
INSTALLED_FROM = "@QUANTUI_VERSION@"
SITE = json.loads("@QUANTUI_SITE@")

JOB_LOG = ".quantui-batch-jobs"  # one "<slurm id>\t<UTC time>" line per sbatch
ATTEMPT_RE = re.compile(r"^attempt-(\d+)_job(\w+)$")
ACTIVE_STATES = ("PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED")
BACKEND_ID = "cluster_slurm"

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


class InputError(Exception):
    """An input file or option that cannot become a job."""


def die(msg, code=1):
    sys.stderr.write("quantui-batch: " + msg + "\n")
    sys.exit(code)


# --------------------------------------------------------------------------
# Where things live (per user, at run time)
# --------------------------------------------------------------------------


def image():
    return os.environ.get("QUANTUI_BATCH_IMAGE") or DEFAULT_IMAGE


def partition():
    return os.environ.get("QUANTUI_SLURM_PARTITION") or SITE["default_partition"]


def staging_root():
    """QuantUI's job root: env, then the user's System Settings, then default."""
    override = os.environ.get("QUANTUI_STAGING_DIR")
    if override:
        return Path(override).expanduser()
    settings = os.environ.get("QUANTUI_SETTINGS_PATH") or str(
        Path.home() / ".quantui" / "settings.json"
    )
    try:
        with open(settings) as fh:
            configured = (json.load(fh).get("compute") or {}).get("slurm_job_root")
        if isinstance(configured, str) and configured.strip():
            return Path(configured.strip()).expanduser()
    except (OSError, ValueError, AttributeError):
        pass
    return Path.home() / ".quantui" / "staging"


def jobs_root():
    override = os.environ.get("QUANTUI_JOBS_DIR")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".quantui" / "jobs"


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


def sbatch_env():
    """The environment for sbatch, minus the caller's own job variables.

    Submitting from inside a job (an interactive srun session, say) would
    otherwise hand that job's SLURM_* values to the new job: sbatch exports
    the whole environment, and Slurm only overwrites the variables the new
    job's own options set. An inherited SLURM_CPUS_PER_TASK=2 would then set
    OMP_NUM_THREADS for a 16-core calculation. SBATCH_* input variables
    (e.g. SBATCH_ACCOUNT) are the user's own defaults and are kept.
    """
    return {k: v for k, v in os.environ.items() if not k.startswith("SLURM_")}


# --------------------------------------------------------------------------
# Building a request (port of quantui.backends.batch_input + molecule.py)
# --------------------------------------------------------------------------


def parse_xyz(text):
    """(atoms, coords, warnings) from XYZ text, with QuantUI's header/comment rules."""
    if not text or not text.strip():
        raise InputError("the file is empty")
    lines = text.strip().split("\n")
    expected = None
    body_start = 0
    for idx, raw in enumerate(lines):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#") or stripped.startswith("!"):
            continue
        try:
            expected = int(stripped)
            body_start = idx + 2  # count line + title line, whatever it holds
        except ValueError:
            pass
        break
    atoms, coords = [], []
    for offset, line in enumerate(lines[body_start:]):
        line_num = body_start + offset + 1
        line = line.strip()
        if not line or line.startswith("#") or line.startswith("!"):
            continue
        for comment_char in ("#", "!"):
            if comment_char in line:
                line = line.split(comment_char)[0].strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 4:
            raise InputError(f"line {line_num}: expected 'SYMBOL X Y Z', got {line!r}")
        symbol = parts[0]
        if symbol not in SITE["atomic_numbers"]:
            hint = ""
            if symbol.capitalize() in SITE["atomic_numbers"]:
                hint = f" (did you mean {symbol.capitalize()!r}?)"
            raise InputError(
                f"line {line_num}: unknown element symbol {symbol!r}{hint}"
            )
        try:
            xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
        except ValueError:
            raise InputError(f"line {line_num}: coordinates must be numbers: {line!r}")
        atoms.append(symbol)
        coords.append(xyz)
    if not atoms:
        raise InputError("no atoms found")
    warnings = []
    if expected is not None and expected != len(atoms):
        # QuantUI only warns here too; the atoms listed are what runs.
        warnings.append(
            f"the first line says {expected} atoms but the file lists {len(atoms)}"
        )
    return atoms, coords, warnings


def charge_mult_problem(atoms, charge, mult):
    """Why charge/multiplicity cannot fit this molecule, or None."""
    n_electrons = sum(SITE["atomic_numbers"].get(a, 0) for a in atoms) - charge
    if mult < 1:
        return f"multiplicity must be at least 1 (got {mult})"
    unpaired = mult - 1
    if unpaired > n_electrons:
        return (
            f"multiplicity {mult} needs {unpaired} unpaired electrons, but the "
            f"molecule has only {n_electrons}"
        )
    if (n_electrons - unpaired) % 2 != 0:
        return (
            f"{n_electrons} electrons cannot have multiplicity {mult} "
            f"(an {'odd' if n_electrons % 2 else 'even'} electron count needs "
            f"{'an even' if n_electrons % 2 else 'an odd'} multiplicity)"
        )
    return None


def parse_option_pairs(pairs):
    options = {}
    for pair in pairs or ():
        key, sep, raw = pair.partition("=")
        key = key.strip()
        if not sep or not key:
            raise InputError(f"--option expects KEY=VALUE, got {pair!r}")
        try:
            options[key] = json.loads(raw)
        except ValueError:
            options[key] = raw
    return options


def file_label(path):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", path.stem).strip("_.-") or "molecule"


def build_request(path, args, extra_options):
    """(request dict, warnings) for one input file."""
    warnings = []
    calc_types = SITE["calc_types"]
    if args.calc is not None and args.calc not in calc_types:
        raise InputError(
            f"unknown --calc {args.calc!r}; choose one of {', '.join(calc_types)}"
        )
    if path.suffix.lower() == ".xyz":
        if not args.calc:
            raise InputError(f"an .xyz input needs --calc ({', '.join(calc_types)})")
        try:
            text = path.read_text()
        except OSError as exc:
            raise InputError(f"could not read file: {exc}")
        atoms, coords, xyz_warnings = parse_xyz(text)
        warnings.extend(xyz_warnings)
        charge = 0 if args.charge is None else args.charge
        mult = 1 if args.mult is None else args.mult
        method = args.method or SITE["default_method"]
        basis = args.basis or SITE["default_basis"]
        problem = charge_mult_problem(atoms, charge, mult)
        if problem:
            raise InputError(
                f"charge {charge} and multiplicity {mult} do not fit this "
                f"molecule: {problem}"
            )
        if method.upper() not in {m.upper() for m in SITE["supported_methods"]}:
            warnings.append(
                f"method {method!r} is not in QuantUI's method list; PySCF may "
                "still accept it, but check the spelling."
            )
        options = dict(extra_options)
        if args.preopt:
            if args.calc in SITE["preopt_calc_types"]:
                options["preopt_before_run"] = True
            else:
                warnings.append(
                    f"--preopt has no effect for {args.calc} (only "
                    f"{', '.join(sorted(SITE['preopt_calc_types']))})."
                )
        label = file_label(path)
        request = {
            "request_id": f"{label[:40]}-{uuid.uuid4().hex[:8]}",
            "calc_type": args.calc,
            "method": method,
            "basis": basis,
            "charge": charge,
            "multiplicity": mult,
            "molecule": {
                "atoms": atoms,
                "coords": coords,
                "label": label,
                "charge": charge,
                "multiplicity": mult,
            },
            "options": options,
            "solvent": args.solvent,
            "run_context": {"source_file": path.name},
        }
        return request, warnings

    try:
        data = json.loads(path.read_text())
        request = {
            "request_id": data["request_id"],
            "calc_type": data["calc_type"],
            "method": data["method"],
            "basis": data["basis"],
            "charge": int(data["charge"]),
            "multiplicity": int(data["multiplicity"]),
            "molecule": dict(data["molecule"]),
            "options": dict(data.get("options") or {}),
            "solvent": data.get("solvent"),
            "run_context": dict(data.get("run_context") or {}),
        }
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise InputError(f"could not read/parse request: {exc}")
    if args.calc is not None:
        request["calc_type"] = args.calc
    if args.method is not None:
        request["method"] = args.method
    if args.basis is not None:
        request["basis"] = args.basis
    if args.charge is not None:
        request["charge"] = request["molecule"]["charge"] = args.charge
    if args.mult is not None:
        request["multiplicity"] = request["molecule"]["multiplicity"] = args.mult
    if args.solvent is not None:
        request["solvent"] = args.solvent
    request["options"].update(extra_options)
    if args.preopt and request["calc_type"] in SITE["preopt_calc_types"]:
        request["options"]["preopt_before_run"] = True
    return request, warnings


# --------------------------------------------------------------------------
# Resources, names, script (port of slurm_utils / cluster_* / SlurmBackend)
# --------------------------------------------------------------------------


def _next_walltime(walltime):
    options = SITE["walltime_options"]
    if walltime in options:
        idx = options.index(walltime)
        if idx + 1 < len(options):
            return options[idx + 1]
    return walltime


def estimate(request):
    mol = request["molecule"]
    atoms = mol.get("atoms") or []
    num_atoms = len(atoms)
    charge = int(mol.get("charge", 0))
    mult = int(mol.get("multiplicity", 1))
    z = SITE["atomic_numbers"]
    num_electrons = sum(z.get(str(a).title(), 0) for a in atoms) - charge

    basis_factor = SITE["basis_factors"].get(request["basis"], 2.0)
    method_upper = request["method"].upper()
    method_factor = 1.2 if method_upper == "UHF" else 1.0
    if method_upper in ("MP2", "CCSD", "CCSD(T)"):
        method_factor = max(method_factor, 2.5)
    elif method_upper not in ("RHF", "UHF"):
        method_factor = max(method_factor, 1.3)
    calc_factor = SITE["calc_factors"].get(request["calc_type"], 1.5)

    base_memory = max(
        4, int(2 * (max(num_electrons, 1) / 10) * basis_factor * method_factor)
    )
    memory_gb = min(int(base_memory * calc_factor), SITE["max_memory_gb"])

    if num_atoms < 10:
        cores = 4
    elif num_atoms < 20:
        cores = 8
    else:
        cores = 16
    cores = min(cores, SITE["max_cores"])

    if num_atoms < 5:
        walltime = "00:30:00"
    elif num_atoms < 10:
        walltime = "01:00:00"
    elif num_atoms < 20:
        walltime = "02:00:00"
    else:
        walltime = "04:00:00"
    if request["basis"] in ("cc-pVTZ",):
        walltime = {
            "00:30:00": "01:00:00",
            "01:00:00": "02:00:00",
            "02:00:00": "04:00:00",
            "04:00:00": "08:00:00",
        }.get(walltime, walltime)
    if calc_factor >= 3.0:
        walltime = _next_walltime(walltime)
    if mult > 1:
        walltime = _next_walltime(walltime)

    multiplier = 1
    if request["calc_type"] == "frequency" and SITE["freq_parallel"]:
        displacements = max(num_atoms, 1) * 3 * 2
        multiplier = max(1, min(max(1, cores // 2), displacements))
        if multiplier > 1:
            memory_gb = min(memory_gb * multiplier, SITE["max_memory_gb"])
    return {
        "cores": cores,
        "memory_gb": memory_gb,
        "walltime": walltime,
        "freq_parallel_memory_multiplier": multiplier,
    }


def resolve_resources(request, args):
    est = estimate(request)
    cores = args.cores or est["cores"]
    memory_gb = args.memory_gb or est["memory_gb"]
    walltime = args.walltime or est["walltime"]
    errors = []
    if not SITE["min_cores"] <= cores <= SITE["max_cores"]:
        errors.append(
            f"cores={cores} out of range [{SITE['min_cores']}, {SITE['max_cores']}]"
        )
    if not SITE["min_memory_gb"] <= memory_gb <= SITE["max_memory_gb"]:
        errors.append(
            f"memory_gb={memory_gb} out of range "
            f"[{SITE['min_memory_gb']}, {SITE['max_memory_gb']}]"
        )
    if walltime not in SITE["walltime_options"]:
        errors.append(
            f"walltime={walltime!r} not one of {', '.join(SITE['walltime_options'])}"
        )
    if errors:
        raise InputError("; ".join(errors))
    return {"cores": cores, "memory_gb": memory_gb, "walltime": walltime}


def sanitize_job_name(name):
    cleaned = re.sub(r"[^A-Za-z0-9_-]+", "_", name.strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_-")
    return cleaned[: SITE["job_name_max_len"]].rstrip("_-")


def default_job_name(request):
    def safe(text):
        return re.sub(r"[^\w\-]", "x", text)

    label = str(request["molecule"].get("label") or "quantui")
    tag = SITE["calc_tags"].get(request["calc_type"], request["calc_type"])
    parts = [label, tag, request["method"], request["basis"]]
    name = "_".join(safe(str(p)) for p in parts if p)
    return sanitize_job_name(name) or "quantui"


def new_job_dir(root, name):
    n = 1
    while True:
        candidate = root / (name if n == 1 else f"{name}_{n}")
        try:
            candidate.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            n += 1
            continue
        return candidate


def _is_within(path, parent):
    try:
        path.resolve().relative_to(parent.resolve())
    except (ValueError, OSError):
        return False
    return True


def worker_command(request_path, job_dir, image_path):
    inner = (
        f"{shlex.quote(IMAGE_PYTHON)} -m quantui.backends.worker --request "
        f'{shlex.quote(str(request_path))} --attempt-dir "$ATTEMPT_DIR"'
    )
    binds = '--bind "$HOME:$HOME"'
    if not _is_within(job_dir, Path.home()):
        job_arg = shlex.quote(str(job_dir))
        binds += f" --bind {job_arg}:{job_arg}"
    return (
        f'apptainer exec --nv {binds} --pwd "$ATTEMPT_DIR" '
        f"{shlex.quote(image_path)} {inner}"
    )


def slurm_script(job_dir, request_path, resources, image_path, email):
    extra = []
    if email:
        extra.append(f"#SBATCH --mail-user={email}")
        extra.append("#SBATCH --mail-type=" + ",".join(SITE["default_mail_events"]))
    return SITE["script_template"].format(
        job_name=job_dir.name[: SITE["slurm_job_name_max_len"]],
        partition=partition(),
        cores=resources["cores"],
        memory=resources["memory_gb"],
        walltime=resources["walltime"],
        output_file=str(job_dir / "slurm-%j.out"),
        error_file=str(job_dir / "slurm-%j.err"),
        optional_directives=("\n" + "\n".join(extra)) if extra else "",
        attempt_setup=f"\nJOB_DIR={shlex.quote(str(job_dir))}\n"
        + SITE["attempt_setup_body"],
        worker_command=worker_command(request_path, job_dir, image_path),
    )


def utc_now():
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def prepare_job(request, resources, job_name, email, image_path):
    """Write the job folder and its record, as SlurmBackend.prepare() does."""
    root = staging_root()
    name = sanitize_job_name(job_name or "") or default_job_name(request)
    job_dir = new_job_dir(root, name)
    request_path = job_dir / "request.json"
    request_path.write_text(json.dumps(request, indent=2))
    script = job_dir / "submit.slurm"
    script.write_text(slurm_script(job_dir, request_path, resources, image_path, email))
    now = utc_now()
    record = {
        "request_id": request["request_id"],
        "backend_id": BACKEND_ID,
        "status": "prepared",
        "calc_type": request["calc_type"],
        "request": request,
        "staging_dir": str(job_dir),
        "created_at": now,
        "updated_at": now,
        "slurm_job_id": None,
        "result_dir": None,
        "resources": dict(resources),
        "error": None,
        "job_dir": str(job_dir),
        "attempts": [],
        "ingested_attempts": [],
    }
    jobs = jobs_root()
    jobs.mkdir(parents=True, exist_ok=True)
    with open(str(jobs / (request["request_id"] + ".json")), "w") as fh:
        json.dump(record, fh, indent=2)
    return script


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
    root = staging_root()
    roots = {str(root), str(root.resolve()) if root.exists() else str(root)}
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
    root = staging_root()
    if not root.is_dir():
        return []
    dirs = [d for d in root.iterdir() if (d / "submit.slurm").is_file()]
    return sorted(dirs, key=lambda d: d.stat().st_mtime, reverse=True)


def resolve_job(name):
    path = Path(name).expanduser()
    if (path / "submit.slurm").is_file():
        return path
    exact = staging_root() / name
    if (exact / "submit.slurm").is_file():
        return exact
    matches = [d for d in all_jobs() if d.name.startswith(name)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        die(
            f"no job folder named {name!r} under {staging_root()} "
            "(see: quantui-batch status)"
        )
    die(f"{name!r} matches several jobs: {', '.join(d.name for d in matches[:8])}")


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
        return (
            f"hit its time limit after {elapsed or '?'}: "
            f"quantui-batch rerun {name} --time=<longer>"
        )
    if state in ("FAILED", "NODE_FAIL"):
        return "see: quantui-batch log " + name
    if state == "CANCELLED":
        return "cancelled"
    return ""


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def submit_parser(prog):
    p = argparse.ArgumentParser(prog=prog, add_help=False)
    p.add_argument("files", nargs="*")
    p.add_argument("--calc")
    p.add_argument("--method")
    p.add_argument("--basis")
    p.add_argument("--charge", type=int)
    p.add_argument("--mult", type=int)
    p.add_argument("--solvent")
    p.add_argument("--preopt", action="store_true")
    p.add_argument("--option", action="append")
    p.add_argument("--job-name", dest="job_name")
    p.add_argument("--cores", type=int)
    p.add_argument("--memory-gb", dest="memory_gb", type=int)
    p.add_argument("--walltime")
    p.add_argument("--email")
    return p


def parse_submit_args(argv, prog):
    args = submit_parser(prog).parse_args(argv)
    if not args.files:
        die(f"usage: {prog} FILE... [options]  (see: quantui-batch help)", 2)
    if args.email and not re.match(
        r"^[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}$", args.email
    ):
        die(f"invalid email address: {args.email!r}", 2)
    try:
        extra = parse_option_pairs(args.option)
    except InputError as exc:
        die(str(exc), 2)
    return args, extra


def image_or_die():
    path = image()
    if not path or path.startswith("@"):
        die(
            "no image configured: reinstall with "
            "`apptainer exec IMAGE quantui install-launcher`"
        )
    if not Path(path).exists():
        die("image not found: " + path)
    return path


def cmd_submit(argv):
    args, extra = parse_submit_args(argv, "quantui-batch submit")
    image_path = image_or_die()
    q = queue()
    active = active_quantui_jobs(q)
    limit = max_jobs()
    if active >= limit:
        die(
            f"you already have {active} QuantUI job(s) queued or running "
            f"(limit {limit}). Wait for one to finish (quantui-batch status) "
            "or cancel one."
        )
    code = 0
    submitted = 0
    for name in args.files:
        path = Path(name)
        try:
            request, warnings = build_request(path, args, extra)
            resources = resolve_resources(request, args)
        except InputError as exc:
            sys.stderr.write(f"{name}: {exc}\n")
            code = 1
            continue
        for warning in warnings:
            sys.stderr.write(f"{name}: warning: {warning}\n")
        if active >= limit:
            print(
                f"not submitted {name} (limit of {limit} jobs reached); "
                "submit it again when a job finishes"
            )
            code = 1
            continue
        script = prepare_job(request, resources, args.job_name, args.email, image_path)
        job_dir = script.parent
        out = run(["sbatch", "--parsable", str(script)], env=sbatch_env())
        if out.returncode != 0:
            sys.stderr.write(out.stderr or "")
            print(
                f"not submitted {job_dir.name}: sbatch failed; "
                f"retry with: quantui-batch rerun {job_dir.name}"
            )
            code = 1
            continue
        job_id = out.stdout.strip().split(";")[0]
        record_job(job_dir, job_id)
        active += 1
        submitted += 1
        print(
            f"submitted {job_dir.name}  (Slurm job {job_id}; "
            f"{resources['cores']} cores, {resources['memory_gb']} GB, "
            f"{resources['walltime']})"
        )
    if submitted:
        print("Check on it with: quantui-batch status")
    return code


def cmd_estimate(argv):
    args, extra = parse_submit_args(argv, "quantui-batch estimate")
    code = 0
    for name in args.files:
        try:
            request, warnings = build_request(Path(name), args, extra)
            res = resolve_resources(request, args)
        except InputError as exc:
            sys.stderr.write(f"{name}: {exc}\n")
            code = 1
            continue
        for warning in warnings:
            sys.stderr.write(f"{name}: warning: {warning}\n")
        print(
            f"{name}: {request['calc_type']} {request['method']}/{request['basis']}"
            f" -> {res['cores']} cores, {res['memory_gb']} GB, {res['walltime']}"
            "  (nothing submitted)"
        )
    return code


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
        print(f"No QuantUI jobs under {staging_root()} yet.")
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


def cmd_log(argv):
    if not argv:
        die("usage: quantui-batch log JOB [-f]", 2)
    job_dir = resolve_job([a for a in argv if a != "-f"][0])
    tried = attempts(job_dir)
    log = tried[-1][2] / "live.log" if tried else None
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
                f"{job_dir.name} is still {q[jid][0].lower()} (job {jid}); "
                "cancel it first or wait."
            )
    active = active_quantui_jobs(q)
    if active >= max_jobs():
        die(
            f"you already have {active} QuantUI job(s) queued or running "
            f"(limit {max_jobs()})."
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
        f"resubmitted {job_dir.name}  (Slurm job {job_id}; earlier attempts are "
        "kept, and an interrupted optimization resumes from its checkpoint)"
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
        print(
            HELP.format(image=image(), staging=staging_root(), limit=max_jobs()), end=""
        )
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
