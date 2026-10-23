# Experimental native resident workflow

This is an integrated research backend for Broad's undashed `pysceptre`,
starting from GPL-3.0-only v0.6.0 / `7adaa9f5fda12e0cf87df972c8fe73b73ef13e7d`.
It builds on resident candidate `eb70553`. The existing CPU default and
`cuda_resident` backend retain their previous execution paths.

```python
result = pysceptre.run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix, target_cells, pairs,
    moi="high", control_group="complement", resampling_mechanism="crt",
    glm_solver="outer", n_jobs=1, seed=271828,
    score_backend="cuda_native", score_batch_size=32,
    score_max_bytes=16 * 1024**3,
)
```

The same opt-in applies to calibration and power. Existing scientific options,
QC, target seeds, fitted assignment probabilities, budgets, pair families,
adaptive rules and BH correction are retained. It is limited to unweighted
complement-control CRT with a dense outer design (1–16 covariates) and one
worker. The native index kernel requires fewer than 2^31 cells and packed
(draw,cell) keys within int64 range; larger inputs are not qualified.

## Integrated computation

Response means use the earlier GPU sufficient-statistic Poisson fit. Native
integer-count dispersion MLE runs in batches on those resident means/counts.
Reciprocal prefix tables replace digamma/trigamma differences via their exact
integer-count identities; a fused kernel reduces cell score and information.
The iteration limit and absolute step threshold remain the reference values.
Fractional counts, unsupported count ranges, failed or nonfinite estimates,
convergence-boundary ambiguity, theta near a clamp or theta >=100 cause a
complete reference response fit and reference scoring. Near-Poisson estimates
are therefore conservative replay cases. Original MM/pilot behavior and clamps
come from the reference on replay, not an alternative GPU fallback formula.

The native sampler generates the original PCG64 raw stream in parallel using
128-bit affine jumps. It classifies variable-consumption exponential attempts,
resolves which raw words were uniforms, compacts accepted attempts in stream
order and advances the host generator only by the consumed prefix. It retains
the reference per-cell padding, 4-million-gap boundaries and continuation order.
GPU segmented sums and packed-key sorting construct a resident binary draw bank.
It generates the complete original budget even if few pairs reach later stages.

NumPy 2.4.6 and 2.5.4 are admitted after checking the owning source: PCG64 header
and Ziggurat tables are byte-identical; the exponential and geometric-inversion
functions also match. The cluster uses **2.4.6**; local CPU tests use **2.5.4**.
Earlier component notes named 2.5.4 from the fetched source; their cluster runtime
was 2.4.6, as recorded by the environment receipts. Other versions, cached uint32
state, p>=1/3, approximate draws, numerical ambiguity or bank budget pressure
restore the original RNG state and replay the complete reference sampler.
Uint64 consumption preserves an unused cached uint32 value when applicable.
Rare-run lookback and speculative exponential allocation are bounded.

The device bank holds sorted int32 cell indices and int64 row offsets, with
implicit values of one. A CPU CSR view is created only for reference scoring
or audit fingerprinting. Replayed approximate banks retain their original
multiplicity algebra on CPU. They do not enter the compact binary kernel.

Each gene retains a=(y-mu)/(1+mu/theta), w=mu/(1+mu/theta), and a tiny inverse
of H=X'WX. A fused gather/reduce evaluates s=sum(a), q=sum(w), v=sum(wX), then
s/sqrt(q-v'H^-1v). It stores two gene/cell planes instead of six at p=4.
Small/invalid variance and conditioning cases replay original scoring.

The scheduler batches only requested pairs, by target and adaptive stage.
It first evaluates stage 1, compacts active requests, evaluates their next
stage together, and repeats through stage 3 when required. Scalar moment
fitting and the original SciPy skew-normal functions remain authoritative;
the 19 tail checks and accepted tail evaluations are vectorized across active
pairs. Empirical p-values, outlier rejection, tail thresholds, probability
floor, extrapolation fallback and stage decisions retain the original logic.

## Bounds and telemetry

Retained gene and native bank bytes share the requested budget; native banks
are limited to half that amount to leave response-cache room. Oversized banks
replay sampling. Shared design, active batches, temporary sorting/reduction
storage, references held by an active job, allocator pools and driver context
are additional, so this is not a hard process-wide cap. Target chunk planning
retains the original CPU estimate. Host response/mean caching is bounded
separately at half the requested budget. Device draws weakly reference their
engine and release accounting when the target chunk ends. Worker globals are
cleared on success and error.

`result.attrs["score_execution"]` includes native bank/replay counts, theta
GPU/reference counts, sensitive gene scoring, unexpected batch replays, compact
variance guards, adaptive batches, transferred bytes, cache/pool peaks and
fit/sampling/theta/score clocks. CPU copies made for the audit are counted.
No device arrays are persisted in returned fit metadata or CPU fit caches.

## Qualification protocol

The first strict-version preflight correctly refused all native banks because
of an inaccurate version assumption; it was fixed after source verification.
A subsequent preflight caught a missing C++17 compiler option in the compact
kernel. Both failed receipts are retained separately. The corrected GPU suite
passed on a 45-GB B200 slice, including all three public families, bank/state
comparisons, sampler rollback after consumption, constructed integer boundary,
near-Poisson/fractional/zero dispersion gates, degenerate variance and extreme
tails. Additional tests cover actual third-stage scheduling, direct compact
versus original null scores, and a forced Ziggurat wedge ambiguity.

Local CPU validation passed 673 tests (37 skipped, 110 deselected), excluding
the separately reproduced upstream environment-sensitive numeric cache-key test.
The default screened-test regression introduced during development was caught
and fixed before this run. Runtime numerical work was not benchmarked on the Mac.

The full experiment uses the complete frozen 201158-cell families: 1024
calibration, 8 power and 1024 discovery pairs, original seed 271828, B1=499,
B2=4999 additional fresh draws and B3=0. It runs fresh stock CPU / native GPU /
native GPU / stock CPU children, one pinned CPU each, on one full B200 node.
The audit hashes every probability vector, complete bank and before/after RNG
state and compares all returned columns, coefficients, theta flags, stages and
BH decisions. Fingerprinting is included in both modes; GPU audit downloads
are not hidden. Source files are hashed before and after the job.

Inference clocks follow tiny kernel/context warmup and input loading. External
whole-process clocks additionally include interpreter/imports, loading/hash/
materialization, context initialization, transfers, correction and export.
Primary compiler caches have been populated by preflight. A separate fresh
worker starts with an empty compiler cache and samples PID-specific VRAM every
200 ms, recording cold first-use time, host peak RSS and observed device peak.
Cold-cache evidence is separate from the paired timing observations.

## Completed clean public-workflow measurement

Job **9380968**, dgx011, completed successfully in 10:34. Its primary sequence
was stock CPU / native GPU / native GPU / stock CPU, fresh Python processes,
one pinned CPU each, one full B200. GPU batch size was 256 and its requested
cache budget 64 GiB. The CPU source was the untouched frozen Broad checkout.
No diagnostic memory observer or profiler ran during these timing pairs.

The clean harness calls the actual three public APIs. It installs **no**
sampler or fit observers, performs **no** extra calibration-QC audit, and runs
**no** unrelated tiny fixtures. Product input validation/hash, public QC,
probability fitting, complete original draw budgets, numerical guards,
reference replays, transfers, correction and export remain in place. Source
and output comparisons run after the primary timing pairs. Compiler disk
caches have previously been populated; these are cold **process** measurements,
not an uncached installation or first-ever compilation claim.

| Scope | CPU observations (s) | GPU observations (s) | Ratio of means |
|---|---|---|---|
| Public API calls, including CUDA initialization/JIT that occurs there | 162.739, 162.646 | 19.713, 19.698 | **8.256x** |
| External whole-process wall time | 197.16, 194.36 | 51.14, 50.23 | **3.862x** |

Input loading/hash/materialization alone was 23.54 s in both modes. Imports,
CUDA startup and loading are included in external wall time. Whole-process
speedup cannot be inferred from warm kernels. Two observations are not a
variance estimate, an optimized multicore CPU comparison, a second dataset,
or cross-platform/statistical/release qualification. The gains combine
mathematical reformulation, batching/caching and GPU execution; a matched
optimized CPU reformulation has not been measured.

All pair identities, finite-result patterns, stages, fit methods, clamp and
convergence flags, and BH decisions matched the reference. Maximum absolute
differences were p=1.7278e-14, observed z=3.1661e-11, coefficients=2.2574e-11,
and theta=1.6380e-7. The worst theta was approximately 35.67356 (relative
4.59e-9). The guide-probability tolerance is **not** applied to theta or p.
Both CPU and GPU returned bitwise identical table values across their two
observations in this bounded run. Numerical agreement with CPU is not an
independent statistical calibration study.

Each GPU run sampled 140 native target banks, without a sampler replay.
Calibration used 986 GPU response fits and one complete sensitive reference
fit/scoring replay; power used 8 and discovery 32 GPU fits. There were no
unexpected batch failures, compact-variance guards or cache evictions.
The 60 runtime/harness checksums matched before/after and match the local
candidate source. Fit comparisons to the separately audited reference are
performed outside the clean timing region.

## Warmed public throughput

Jobs **9380307** and **9380877** ran sequential full workflows in a persistent
Python/CUDA session with a previously validated RAM input reused on the second
iteration. There are no sampler fingerprint hooks. The batch-32 harness kept
an extra calibration audit for comparison; its separately measured prelude is
excluded from the following public-call sums for **both** modes. The batch-256
harness does not run that prelude. CPU then GPU is fixed order here: these
are secondary steady-execution measurements, not balanced cold evidence.

| Configuration | CPU public calls (s) | GPU public calls (s) | Observed ratios |
|---|---|---|---|
| Batch 32 / 16-GiB requested cache | 160.852, 161.435 | 14.588, 13.886 | **11.03, 11.63x** |
| Batch 256 / 64-GiB requested cache | same CPU observations | 15.391, 15.032 | **10.45, 10.74x** |

All twelve family/repetition/configuration comparisons preserved identities,
stages and BH decisions; maximum p/z differences match the clean experiment's
bounds above. Batch 32 was slightly faster warm. More VRAM or a larger batch
is not automatically better: larger fit batches can keep converged genes in
work until the slowest gene finishes. Both configurations remain bounded
experimental observations, not a general tuning rule.

## Separate audit and memory companions

Job **9379928**, dgx019, performed full probability/bank/before-and-after PCG
fingerprinting in the public workflow. All 280 bank/probability/state/budget
comparisons matched exactly. No method, clamp, convergence, adaptive-stage or
BH flag changed. The maximum audited theta difference was 1.6380e-7; maximum
p difference 1.7278e-14. This validates the stream-preserving native bank path;
gene batch width does not enter target fit/chunk/seed/sampler planning.

Those audit hooks force every full native bank back to the CPU. Their measured
cost in the first native repetition was 17.07 s calibration, 2.23 s power and
12.18 s discovery, versus 2.65/0.38/2.03 s for CPU-owned banks. The harness's
extra calibration audit also costs time outside the public call. Identical
instrumentation therefore does not impose identical work on the backends.
Moreover, the audited CPU compute observations were 184.62 and 451.54 s;
the unexplained second CPU slowdown makes their timing ratio unsuitable for
performance acceptance. All rows are retained. Use this experiment for its
scientific audit, and the separate clean balanced experiment for performance.
The previous resident candidate's 1.393x/1.251x audit-enabled result remains
separately archived; it is not relabeled as a clean measurement.

After the clean timing pairs, a separate fresh GPU child with an empty compiler
cache sampled PID-specific VRAM every 200 ms. Observed peak was **5756 MiB**
(5.62 GiB) across 419 samples; host peak was **15,437,308 KiB** (14.72 GiB).
The primary clean GPU host peaks were 15,289,216 / 15,348,040 KiB, versus
11,363,372 / 11,401,208 KiB on CPU. Discrete VRAM observations are not an exact
continuous maximum. The requested cache budget is not a hard process cap.

The empty-cache observed companion took 120.44 s externally (86.94 s public
calls), versus about 50.7 s externally in the clean cached-compiler timing
pairs. This companion also runs an NVML memory observer; its time is recorded
**descriptively, not accepted as an unobserved first-use benchmark**. It shows
that first-use compilation cannot be ignored. A clean empty-cache timing pair
has not been established. No observer contributes to the primary 3.862x ratio.

A separately profiled 45-GiB-slice public workflow used cProfile and batch 256,
without benchmark audits. It identified about 18.6 s in NVRTC compilation and
28.5 s in input loading in that process; these are profiling observations,
not primary performance numbers. Actual runtime data are NumPy 2.4.6,
SciPy 1.17.1, CuPy 13.6.0, Python 3.11.11 and CUDA 12.8.1.

## Review and reproduction

The checked-in harnesses live under `tools/gpu/native/` so that the earlier
`tools/gpu/resident/` protocol remains unchanged. Experiments ran from an
isolated workspace where these new harnesses used the latter directory name;
the checked-in copies change those path references only. Runtime source bytes
match the measured source and built wheel. Original harnesses/manifests,
receipts, result tables and failed preflights remain local/private under
`tools/gpu/evidence/native-integrated`, `native-clean`, `native-warmed`,
`native-wide-warmed` and `native-profile`; actual-data-derived fits/banks are
not package fixtures or tracked test data.

The private wheel SHA256 is
`7f9ee9c5dd9327e772e7a6158e0eea69a93df73158c22f27e439e5982f762fe1`.
Its runtime files, CUDA source and notices match the candidate byte for byte.
The extracted wheel passed 21 GPU public-family/stage/boundary tests; importing
the package on CPU does not import CuPy. Full-node preflight passed 97 tests,
with 3 additional stage/algebra/wedge tests on the GPU slice. Code lint passed.

The implementation is opt-in and local. It has not been merged or published.
The current data show approximately one order of magnitude in warm public
throughput, not 100x whole-process acceleration. At the present input path,
loading alone is over ten times the entire time budget for a cold 100x claim.
