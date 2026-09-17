"""Prepare one exact upstream revision without changing scheduler policies."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

from . import SGLANG_REVISION

MARKER = ".hbfsim-integration.json"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _verified_marker(source: Path):
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SGLANG_REVISION:
        raise ValueError(f"SGLang must be at {SGLANG_REVISION}; found {revision}")
    marker = json.loads((source / MARKER).read_text())
    if marker["revision"] != revision or any(sha(source / p) != digest for p, digest in marker["files"].items()):
        raise ValueError("prepared SGLang hooks changed; use a fresh checkout and prepare it again")
    return marker


def verify(source: Path):
    marker = _verified_marker(source)
    if marker.get("recipe_sha256") != sha(__file__):
        raise ValueError("prepared SGLang hooks are stale; run prepare to refresh the verified integration edits")
    return marker


def prepare(source: Path, reference: Path | None = None):
    source = source.resolve()
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        if reference:
            # A worktree also works with partial/shallow local repositories;
            # cloning every advertised ref can demand unavailable history.
            subprocess.run(["git", "-C", str(reference.resolve()), "worktree", "add",
                "--detach", str(source), SGLANG_REVISION], check=True)
        else:
            subprocess.run(["git", "clone", "--no-checkout", "--filter=blob:none",
                "https://github.com/sgl-project/sglang.git", str(source)], check=True)
            subprocess.run(["git", "-C", str(source), "checkout", "--detach", SGLANG_REVISION], check=True)
    if (source / MARKER).exists():
        marker = _verified_marker(source)
        if marker.get("recipe_sha256") == sha(__file__):
            return marker
        # Refresh only our digest-verified edits. Unrelated upstream changes
        # still fail the pristine-file checks below.
        for path in marker["files"]:
            pristine = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{path}"], text=True)
            (source / path).write_text(pristine)
        (source / MARKER).unlink()
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if revision != SGLANG_REVISION:
        raise ValueError(f"prepare requires exact SGLang revision {SGLANG_REVISION}")
    base = "tools/sglang-simulator/src/sglang_simulator/simulation/sglang/"
    paths = [base+"hook_bootstrap.py", base+"scheduler.py",
        "python/sglang/kernels/aot/python/sgl_kernel/__init__.py",
        "python/sglang/srt/utils/common.py", "python/sglang/srt/distributed/bootstrap.py"]
    originals = {p: (source / p).read_text() for p in paths}
    for p, text in originals.items():
        pristine = subprocess.check_output(["git", "-C", str(source), "show", f"HEAD:{p}"], text=True)
        if text != pristine:
            raise ValueError(f"refusing to replace existing edits in {p}; prepare a separate checkout")
    changed = dict(originals)

    def replace(path, old, new):
        if changed[path].count(old) != 1:
            raise ValueError(f"upstream hook anchor changed in {path}: {old[:70]}")
        changed[path] = changed[path].replace(old, new)

    changed[paths[0]] += "\n# Install the optional HBFSim predictor in parent and spawned interpreters.\nfrom hbserve.sglang.adapter import install as install_hbfsim\ninstall_hbfsim()\n"
    p = paths[1]
    replace(p, "                        time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.",
        "                        len(self.future_queue),  # Unique FIFO index; wall-clock ticks can collide.")
    replace(p, "        def wrapped_run_batch(self, *args, **kwargs):\n",
        "        def wrapped_run_batch(self, *args, **kwargs):\n"
        "            native_snapshot = None\n"
        "            if ConfigManager._get_raw_config().get('predictor', {}).get('name') == 'hbfsim':\n"
        "                from hbserve.sglang.adapter import capture_batch\n"
        "                native_batch = get_obj_from_args('sglang.srt.managers.schedule_batch.ScheduleBatch', *args, **kwargs)\n"
        "                native_snapshot = capture_batch(native_batch, self.future_map)\n")
    replace(p, "                if not simulation_batch.is_empty():\n",
        "                if not simulation_batch.is_empty():\n"
        "                    if native_snapshot is not None:\n"
        "                        simulation_batch.hbfsim_requests = native_snapshot\n")
    replace(p, "                cpu_overhead = max(", "                host_cpu_overhead = max(")
    replace(p, "                StateManager.step_global_clock(cpu_overhead)",
        "                cpu_overhead = host_cpu_overhead\n"
        "                predictor_config = ConfigManager._get_raw_config().get('predictor', {})\n"
        "                if predictor_config.get('name') == 'hbfsim':\n"
        "                    cpu_overhead = predictor_config['cpu_overhead_ns'] / 1e9\n"
        "                StateManager.step_global_clock(cpu_overhead)")
    replace(p, '                        "cpu_overhead": cpu_overhead,',
        '                        "cpu_overhead": cpu_overhead,\n                        "host_cpu_overhead": host_cpu_overhead,')
    replace(p, "                StateManager.step_global_clock(\n                    now - StateManager.get_last_real_time_ts()\n                )",
        "                if ConfigManager._get_raw_config().get('predictor', {}).get('name') != 'hbfsim':\n"
        "                    StateManager.step_global_clock(now - StateManager.get_last_real_time_ts())")
    replace(p, "                    StateManager.set_global_clock(next_created_time + 1e-6)",
        "                    slack = 0 if ConfigManager._get_raw_config().get('predictor', {}).get('name') == 'hbfsim' else 1e-6\n"
        "                    StateManager.set_global_clock(next_created_time + slack)")
    replace(p, "                now if self.mode == SimulationMode.BLOCKING else 0\n",
        "                now if self.mode == SimulationMode.BLOCKING else StateManager.get_global_clock()\n")
    replace(p, "                metrics.update(C_SchedulerHook.INFERENCE_PREDICTOR.get_metrics())",
        "                if hasattr(C_SchedulerHook.INFERENCE_PREDICTOR, 'finalize'):\n"
        "                    C_SchedulerHook.INFERENCE_PREDICTOR.finalize()\n"
        "                metrics.update(C_SchedulerHook.INFERENCE_PREDICTOR.get_metrics())")
    replace(paths[2], 'if sys.platform == "darwin" and platform.machine() == "arm64":',
        'if sys.platform == "darwin" and platform.machine() == "arm64" and os.environ.get("SGLANG_SIMULATOR_BOOTSTRAP") != "1":')
    changed[paths[2]] = "import os\n" + changed[paths[2]]
    replace(paths[3], "def get_cpu_ids_by_node():\n",
        "def get_cpu_ids_by_node():\n"
        "    if os.uname().sysname == 'Darwin' and os.environ.get('SGLANG_SIMULATOR_BOOTSTRAP') == '1':\n"
        "        return [','.join(map(str, range(os.cpu_count() or 1)))]\n")
    replace(paths[4], "    if _is_cpu_amx_available or _is_cpu_arm64:\n",
        "    if os.uname().sysname == 'Darwin' and os.environ.get('SGLANG_SIMULATOR_BOOTSTRAP') == '1':\n"
        "        if tp_size != 1:\n"
        "            raise ValueError('macOS simulator requires a single worker')\n"
        "        return\n"
        "    if _is_cpu_amx_available or _is_cpu_arm64:\n")
    for p, text in changed.items():
        (source / p).write_text(text)
    marker = {"revision": revision, "recipe_sha256": sha(__file__), "files": {p: sha(source / p) for p in paths}}
    (source / MARKER).write_text(json.dumps(marker, indent=2))
    return marker
