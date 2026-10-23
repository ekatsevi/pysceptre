# Experimental integrated resident execution

This increment connects the sufficient-statistic response fit to a real
public calibration/power/discovery workflow. It starts from Broad `pysceptre`
`7adaa9f5fda12e0cf87df972c8fe73b73ef13e7d` and the isolated GPU experiments
through `4a536eb`. It does not change the default CPU backend.

```python
result = pysceptre.run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix, target_cells, discovery_pairs,
    moi="high", control_group="complement", resampling_mechanism="crt",
    glm_solver="outer", n_jobs=1,
    score_backend="cuda_resident",
    score_batch_size=32,
    score_max_bytes=16 * 1024**3,
    # Keep the original seed, side, approximation and scientific settings.
)
```

The same execution arguments apply to `run_calibration_check` and
`run_power_check`. This experimental `score_backend` value also selects GPU
response fitting; it is a different numerical backend, not merely a device
replacement for one matrix multiply. It currently admits only unweighted
complement-control CRT with the outer design and one worker. Unsupported
combinations are refused. Existing CPU, structured, permutation, NT-cell and
weighted analyses retain their previous paths.

## What executes where

- Shared covariates and their row outer products are uploaded once per public
  analysis. GPU FP64 Poisson fits process response batches using fixed X'y,
  count sums and count-log-count sums. The initialization and deviance
  tolerance are retained; clipped means require reference replay.
- Fitted means return to the **unchanged CPU dispersion estimator**, including
  its MLE/MM/pilot behavior and [0.01, 1000] clamp. This boundary is deliberate
  and limits acceleration. No GPU dispersion implementation is claimed.
- Nonconverged/nonfinite/clipped response batches and genes whose dispersion
  hits a boundary or fallback method use the original CPU fit. Those genes
  also retain CPU scoring. Conditioning failures select CPU scoring.
- GPU gene score vectors and small factorizations remain resident across
  target chunks. A bounded host cache retains response/mean vectors needed
  for effects and avoids rereading/reconstructing them when possible.
- Assignment fits and the entire sampler remain the original CPU code. Their
  target seeds, fitted probabilities, number/order of random draws and
  exact/approximate sampler selection are unchanged. Device banks are copies
  of these original sparse banks, including repeated-entry multiplicities.
- First-stage scores batch requested genes for a target. Every reached later
  stage can also use the resident gene and target bank. The existing adaptive
  empirical/skew-normal test runs through one shared CPU helper; it is not
  duplicated in the GPU engine. Effects, tail checks and BH remain CPU.

This implements a bounded part of the larger resident design. Initialization
still densifies each response batch, moment kernels are not fused, and the
stream-preserving GPU sampler construction is not integrated here.

## Memory and failures

`score_max_bytes` bounds retained device cache entries, further limited to
65 percent of free device memory at engine creation. Gene and bank objects
share an LRU cache. A bank or gene object too large for that cache declines
to the visible reference path. The host cache is bounded separately at half
the requested value. Shared design arrays, active-batch/solver workspaces,
temporaries, allocator reservations and the driver context are additional;
the setting is not a hard whole-process memory limit.

Import/runtime/memory failures during engine setup/fitting decline the whole
analysis to CPU. Scoring failures replay a gene or batch on CPU. Replays are
reported through warnings and `result.attrs["score_execution"]`, including
the requested/effective backend, GPU fit counts, sensitive CPU fit/scoring
counts, unexpected fallback batches, transfers, evictions and cache/pool
peaks. Sensitive-gene CPU replay is separate from unexpected backend failure.
The diagnostic `resident_fits` attribute contains scalar fitted parameters
and conditioning summaries, never a device object. This implementation does
not add GPU model objects to the existing persisted CPU fit-cache format.

## Validation and timing protocol

`tests/validation/test_resident_engine.py` exercises the three public families,
later adaptive stages, the sufficient-statistic fit with integer/fractional
responses, unsupported-mode rejection and visible unavailable-device replay.
The default local validation passed 667 tests (25 skipped, 110 deselected),
excluding the previously reproduced upstream environment-sensitive cache-key
test. The new GPU tests passed on a full B200 and a 45-GiB B200 slice.

`tools/gpu/resident/run.sbatch` runs public-family GPU preflight followed by
fresh stock-CPU / resident-GPU / resident-GPU / stock-CPU children on one B200
node. Scientific work uses the complete frozen 201,158-cell, 1,024 calibration,
eight power and 1,024 discovery pairs. Both modes use one pinned CPU, identical
settings and inputs, and the same private Python/CUDA environment. The CPU
source is the untouched frozen upstream checkout.

The harness fingerprints every target's original fitted probabilities,
complete sampled bank, before/after PCG state and budgets. Fingerprinting
overhead is included equally in both modes and recorded per target. It also
captures response coefficients, dispersion and flags; compares all returned
columns and BH decisions; and checks repeat stability. Real-data receipts
remain local rather than being committed as package fixtures.

Report inference separately from external whole-process wall time. The latter
includes interpreter/import/setup, input hashing/materialization, tiny code
warmup, GPU startup/transfers, inference, correction and export. Preflight has
already populated private compiler caches; these are not cold-install timings.
A separate fresh GPU worker samples process-specific VRAM every 200 ms after
the primary timing pairs. Its observed maximum is discrete, not continuous.

## Completed bounded experiment

Job 9377846 completed successfully on dgx011 with a full B200. In two
reversed-order observations per mode, CPU inference took 187.695 / 175.926 s
and resident inference took 128.098 / 133.017 s: a ratio of means of 1.393.
External whole-process times were CPU 219.65 / 208.87 s and resident
172.77 / 169.84 s: a ratio of means of 1.251. These are single-worker
comparisons, with equal sampler fingerprint overhead, not a comparison to
an optimized multicore CPU engine. Reformulation, caching and GPU work are
combined; a matched CPU reformulation/cache control has not been measured.

All bank, probability and before/after RNG fingerprints matched. No pair
identity, adaptive-stage or BH decision changed. Maximum absolute differences
were p=1.7223e-14, observed z=3.1683e-11, and theta=1.2075e-6. The worst theta
case was approximately 35.67356 (relative difference 3.3848e-8); this is
separate from the user-approved guide-probability tolerance. Fit methods,
clamps and convergence flags matched. CPU repeats were bitwise stable; GPU
repeats were not bitwise stable. This is bounded numerical evidence, not a
statistical or cross-platform qualification.

The separate fresh-process memory worker observed a maximum of 15,466 MiB
across 735 process-specific samples at 200 ms. The primary runs reported no
bank or gene cache evictions. The B200 has roughly 180 GB of device memory:
capacity did not limit this workload. More VRAM alone cannot remove the
remaining CPU sampler, dispersion, tails or startup/materialization costs.
Source checksums matched before/after the experiment. Raw private receipts
and summary.json are retained locally under tools/gpu/evidence/integrated-resident.

The preceding isolated warm-fit gain does not predict the workflow gain.
