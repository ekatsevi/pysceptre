"""Separate observed GPU-memory run, outside the primary timing experiment."""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("root", type=Path)
ap.add_argument("out", type=Path)
args = ap.parse_args()
args.out.mkdir(parents=True, exist_ok=False)
command = [
    sys.executable,
    "tools/gpu/native/run-workflow.py",
    str(args.root),
    str(args.out / "workflow"),
    "outer",
    "--score-backend",
    "cuda_native",
    "--score-batch-size",
    "32",
    "--score-max-bytes",
    "17179869184",
    "--storage",
    "ram",
]
with (args.out / "worker.log").open("w") as log:
    cold_env = dict(os.environ, CUPY_CACHE_DIR=str(Path.cwd() / "cupy-cache-cold-memory"))
    started = time.perf_counter()
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=cold_env)
    query = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,used_gpu_memory",
            "--format=csv,noheader,nounits",
            "--loop-ms=200",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    samples = []

    def observe():
        assert query.stdout is not None
        for line in query.stdout:
            try:
                pid, used = [x.strip() for x in line.split(",")]
                if int(pid) == child.pid:
                    samples.append(dict(epoch=time.time(), pid=int(pid), mib=float(used)))
            except (ValueError, TypeError):
                pass

    observer = threading.Thread(target=observe, daemon=True)
    observer.start()
    try:
        status = child.wait()
    finally:
        query.terminate()
        try:
            query.wait(timeout=5)
        except subprocess.TimeoutExpired:
            query.kill()
            query.wait()
        observer.join(timeout=5)
    if status:
        raise subprocess.CalledProcessError(status, command)
    if not samples:
        raise RuntimeError("No process-specific VRAM observations collected")
    (args.out / "gpu-memory.json").write_text(
        json.dumps(
            dict(
                external_child_wall_seconds=time.perf_counter() - started,
                fresh_compiler_cache=True,
                peak_observed_mib=max(s["mib"] for s in samples),
                samples=samples,
                interval_ms=200,
                scope="Whole fresh GPU worker including import, tiny-fixture compilation, load, inference and export. Discrete nvidia-smi observations of this PID only, not a continuous exact maximum. Separate from primary timing pairs.",
                job=os.environ.get("SLURM_JOB_ID"),
            ),
            indent=2,
        )
        + "\n"
    )
