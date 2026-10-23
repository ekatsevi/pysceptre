"""Summarize measured complete-process costs without extrapolating kernel gains."""

import argparse
import json
import statistics
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("directory", type=Path)
args = ap.parse_args()


def seconds(value):
    result = 0.0
    for part in value.strip().split(":"):
        result = 60 * result + float(part)
    return result


def external_wall(path):
    for line in path.read_text().splitlines():
        if "Elapsed (wall clock)" in line:
            return seconds(line.rsplit(": ", 1)[1])
    raise ValueError("Missing external elapsed time: " + str(path))


modes = {}
for mode in ["cpu", "cuda_native"]:
    records = [
        json.loads((args.directory / f"r{r}-{mode}" / "receipt.json").read_text()) for r in [0, 1]
    ]
    wall = [external_wall(args.directory / f"r{r}-{mode}-resource.txt") for r in [0, 1]]
    modes[mode] = dict(
        inference_seconds=[d["total_inference_seconds"] for d in records],
        whole_child_seconds=[d["whole_child_seconds"] for d in records],
        external_wall_seconds=wall,
        mean_external_wall_seconds=statistics.mean(wall),
        mean_inference_seconds=statistics.mean(d["total_inference_seconds"] for d in records),
        host_maxrss_kib=[d["host_maxrss_kib"] for d in records],
        stage_seconds={
            stage: [next(c["seconds"] for c in d["clocks"] if c["stage"] == stage) for d in records]
            for stage in ["calibration", "power", "discovery"]
        },
    )

comparison = json.loads((args.directory / "comparison.json").read_text())
report = dict(
    modes=modes,
    inference_ratio=modes["cpu"]["mean_inference_seconds"]
    / modes["cuda_native"]["mean_inference_seconds"],
    whole_process_ratio=modes["cpu"]["mean_external_wall_seconds"]
    / modes["cuda_native"]["mean_external_wall_seconds"],
    maximum_p_difference=max(
        c["differences"]["p_value"]["max_abs"] or 0 for c in comparison["paired"]
    ),
    maximum_observed_statistic_difference=max(
        c["differences"]["z_orig"]["max_abs"] or 0 for c in comparison["paired"]
    ),
    maximum_theta_difference=max(
        c["fit_differences"]["max_theta_abs"] for c in comparison["paired"]
    ),
    changed_BH_decisions=sum(
        c["differences"]["BH_discovery"]["changed"] for c in comparison["paired"]
    ),
    changed_stages=sum(c["differences"]["stage"]["changed"] for c in comparison["paired"]),
    rng_or_draw_changes=sum(sum(c["draw_differences"].values()) for c in comparison["paired"]),
    repeat_stability=comparison["repeat_stability"],
    scope="Two reversed-order fresh-process observations per mode, complete frozen families, one pinned CPU and one B200; identical sampler fingerprint overhead included. No variance estimate, multicore CPU comparison, other-dataset or release qualification.",
)
memory = args.directory / "memory/gpu-memory.json"
if memory.exists():
    d = json.loads(memory.read_text())
    report["peak_observed_vram_mib"] = d["peak_observed_mib"]
    report["vram_sample_count"] = len(d["samples"])
(args.directory / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
