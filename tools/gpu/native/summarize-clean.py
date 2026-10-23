"""Report uninstrumented public-call and external-process measurements."""

import argparse, json, statistics
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("directory", type=Path)
a = p.parse_args()


def external(path):
    line = next(x for x in path.read_text().splitlines() if "Elapsed (wall clock)" in x)
    result = 0.0
    for part in line.rsplit(": ", 1)[1].split(":"):
        result = result * 60 + float(part)
    return result


modes = {}
for mode in ["cpu", "cuda_native"]:
    records = [json.loads((a.directory / f"r{i}-{mode}/receipt.json").read_text()) for i in [0, 1]]
    modes[mode] = dict(
        public_seconds=[x["total_inference_seconds"] for x in records],
        external_seconds=[external(a.directory / f"r{i}-{mode}-resource.txt") for i in [0, 1]],
        host_maxrss_kib=[x["host_maxrss_kib"] for x in records],
        load_hash_seconds=[x["input_load_hash_seconds"] for x in records],
        imports_seconds=[x["imports_seconds"] for x in records],
        stage_seconds={
            s: [next(c["seconds"] for c in x["clocks"] if c["stage"] == s) for x in records]
            for s in ["calibration", "power", "discovery"]
        },
    )
comparison = json.loads((a.directory / "comparison.json").read_text())
summary = dict(
    modes=modes,
    public_ratio=statistics.mean(modes["cpu"]["public_seconds"])
    / statistics.mean(modes["cuda_native"]["public_seconds"]),
    whole_process_ratio=statistics.mean(modes["cpu"]["external_seconds"])
    / statistics.mean(modes["cuda_native"]["external_seconds"]),
    max_p_abs=max(c["differences"]["p_value"]["max_abs"] or 0 for c in comparison["paired"]),
    max_z_abs=max(c["differences"]["z_orig"]["max_abs"] or 0 for c in comparison["paired"]),
    BH_changed=sum(c["differences"]["BH_discovery"]["changed"] for c in comparison["paired"]),
    stage_changed=sum(c["differences"]["stage"]["changed"] for c in comparison["paired"]),
    repeat_stability=comparison["repeat_stability"],
    scope="Two reversed-order fresh-process observations per mode; complete frozen 201158-cell, 2056-pair families; stock one-worker CPU vs B200/one CPU. No sampler/fit hooks, manual calibration audit or unrelated warmups. Product input validation, QC, numerical guards and transfers retained. Primary disk compiler caches populated; no variance estimate or multicore/platform/statistical qualification.",
)
mp = a.directory / "memory/gpu-memory.json"
if mp.exists():
    m = json.loads(mp.read_text())
    r = json.loads((a.directory / "memory/workflow/receipt.json").read_text())
    summary["separate_empty_compiler_cache_companion"] = dict(
        external_seconds=m["external_child_wall_seconds"],
        whole_child_seconds=r["whole_child_seconds"],
        public_seconds=r["total_inference_seconds"],
        peak_observed_vram_mib=m["peak_observed_mib"],
        host_maxrss_kib=r["host_maxrss_kib"],
        sample_count=len(m["samples"]),
    )
(a.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
