# Experimental CUDA target fitting

This private branch starts at Broad Institute `pysceptre` 0.6.0,
`7adaa9f5fda12e0cf87df972c8fe73b73ef13e7d`. It adds the opt-in
`glm_solver="cuda_targets"` to the complement-control analysis path. The
existing `outer` default, structured solver, low-MOI NT-cell analysis,
assignment methods, permutations, calibration, power and discovery remain
available. The new solver targets only binary CRT treatment-membership fits and requires
`n_jobs=1`; the default CPU solvers retain their existing parallel paths.
Permutation analyses skip that fit as before. NT-cell requests for the new
solver are rejected by the existing solver-admission boundary.

```python
result = pysceptre.run_discovery_analysis(
    response_matrix, gene_ids, covariate_matrix, target_cells, discovery_pairs,
    glm_solver="cuda_targets",
    # Supply the same scientific settings used by your CPU analysis.
)
```

Install the optional GPU dependencies with `pip install '.[gpu]'` on a
CUDA 12 installation. Choose the CuPy wheel matching the actual CUDA runtime
and driver. This branch was prepared for a CUDA 12.8 B200 environment; a
working QR smoke test alone does not establish JIT compatibility for every
array kernel. Default CPU use never imports CuPy and has no GPU dependency.

The GPU and CPU share the existing IRLS loop, marginal-rate starting values,
mean floor, active-response updates and deviance stopping rule. Arithmetic is
float64. cuBLAS reductions and cuSOLVER replace their CPU counterparts for the
batched logistic fit. This is a numerical backend change, with no different
statistical model or intentional approximation. It is not a QR solver.

Gene fits, dispersion estimation, score reconstruction, observed and null
statistics, exact/approximate sampler selection, target-hashed random streams,
adaptive budgets and tail fitting remain upstream. GPU probabilities return
to the existing CPU sampler. Small probability changes can still change a
draw or a p-value; compare these separately rather than accepting every output
at the probability tolerance. The Python package's empirical tie convention
and its differences from R remain upstream behavior.

The design and its outer products stay resident for one analysis. A lock
serializes GPU fits requested by the existing prefetch threads. CUDA state is
created lazily after the CPU gene-fitting pass. Process workers are refused
for this experimental solver.
An estimated working set exceeding 2 GiB or 70 percent of free VRAM is declined
without splitting a numerical batch. Import/runtime errors, nonfinite fits
and nonconvergence also replay the original CPU fit, with a RuntimeWarning.
`CudaTargetDesign(..., fallback=False)` makes a component diagnostic strict.
`gpu_calls`, `fallback_calls`, `last_failure` and `last_stats` describe direct
component runs. The estimate does not cap whole-process memory; CUDA libraries
and allocator reservations have additional costs.

The current scope is an experimental complement-control backend. Default CPU
caches are not silently reclassified as CUDA fits: the existing solver identity
travels in cache fingerprints. No release, merge or general GPU/platform
qualification follows from these experiments.

## Sources and provenance

The implementation base is GPL-3.0-only Broad `pysceptre`, verified against
`PY-UPSTREAM.json` in the frozen SCEPTRE report comparison. The richer Broad
scope includes both resampling methods, three analyses, RAM/backed readers,
assignment and existing sparse/indexed-score optimizations. The dashed
Katsevich-Lab repository is a different package.

- [Katsevich-Lab/py-sceptre](https://github.com/Katsevich-Lab/py-sceptre/tree/0c9f39b3d6d89196f320c6a75fd95df7f32b6ac6), MIT, declares Goodarzi Lab copyright. Its batched GPU GLM design was useful inspiration. Its implemented low-MOI discovery scope is narrower than the Broad package.
- [ArcInstitute/gpudge](https://github.com/ArcInstitute/gpudge/tree/66232795f45003e649ce19b4916eae462558fb12), MIT, Arc Research Institute copyright. Useful ideas include resident reference data, memory-bounded work, transfer accounting and allocator cleanup. Its unadjusted Wilcoxon method does not replace SCEPTRE's covariate-adjusted test.

No code was copied from either MIT repository. Existing Broad and third-party
notices remain unchanged. Earlier R score and dashed-package prototypes are
preserved in separate checkouts; their component timings are not Broad workflow
measurements.

## Validation and reproduction

`tools/gpu/smoke.sbatch` runs focused numerical, fallback and workflow tests on
one normally allocated B200. `tools/gpu/run-paired.sbatch` runs fresh-process,
reversed-order CPU/GPU tests against the frozen report inputs. The CPU source
is the original frozen source, not the modified candidate. Both use one pinned
CPU and the same private Python environment on the same GPU node. Input hashes,
complete original families, cell ordering and counts remain authoritative.
Inference excludes loading, hashing, tiny-fixture compilation, correction and
export; these are reported separately. Phase clocks in direct fit diagnostics
include synchronization. Device events cover the resident computation, including
launch gaps and scalar synchronization, not one isolated CUDA kernel.

The first direct real-target diagnostic fit 16 frozen discovery groups across
201,158 cells and four covariates. CPU times were 0.317 to 0.358 seconds.
The first GPU call was 4.573 seconds, including 2.907 seconds of import and
1.311 seconds of setup/transfer. Subsequent GPU calls were 0.015 to 0.022 seconds,
including input and returned probability transfer. Maximum coefficient
difference was 1.05e-11; probability difference 5.19e-14; deviance difference
3.64e-12. Iterations and convergence flags were unchanged. Three strict GPU
calls completed with zero CPU fallbacks. This is a component observation, not
a workflow speedup. The allocator reserved about 301 MiB in this direct run;
that excludes the driver's context and other library reservations.

The two reversed-order full backed pairs completed as follows. Each child ran
the exact frozen CRT calibration/power/discovery families, with all 201,158
retained cells and 13,135 gene labels, one pinned CPU, unchanged sampler,
complete budgets and new private environment on the same B200 node.

| Pair order | CPU inference seconds | CUDA-target inference seconds |
| --- | ---: | ---: |
| CPU then GPU | 195.862 | 196.654 |
| GPU then CPU | 196.742 | 194.730 |
| Arithmetic mean | 196.302 | 195.692 |

The mean difference is 0.31 percent, with opposite signs in the two pairs.
Whole-child means were 205.10 seconds CPU and 207.31 seconds GPU. Thus this
candidate provides no established practical workflow gain. Two pairs are
bounded observations, not a general variance estimate. OS page cache was
uncontrolled. The CPU source is untouched upstream; timings are fresh, not
inherited from the older report runtime.

All 2,056 rows per pair (1,024 calibration, eight power, 1,024 discovery) were
exactly identical CPU/GPU, including observed statistics, effects, stages,
tail parameters, p-values, externally computed BH q-values and decisions.
Each mode also reproduced its own result tables exactly across repeats.
No CUDA fallback warning occurred in these full runs. Captured legacy
NumPy/Python global RNG state stayed unchanged; this does not capture every
internal target-local PCG state.

A separate 16-target diagnostic compared actual first-stage 499-draw CSR banks:
zero changed banks, maximum fitted probability difference 5.19e-14, one strict
GPU fitting call and zero fallbacks. This is scoped to those targets and that
stage, not every seed or budget. A separate full GPU worker was observed with
`nvidia-smi` at 200 ms intervals: maximum observed VRAM 996 MiB over 1,010
PID-specific samples. This includes setup and inference and is not a continuous
exact maximum. Whole-child RSS for the primary CPU children was about6.00 to
6.02 GiB and GPU children 6.57 to6.66 GiB. Memory observations are separate from
the primary timing pairs.

The compatible numerical preflight passed 101 tests. The exact final candidate
then passed six CUDA/fallback tests on the paired node. Sixty-four default CPU
fit cases matched untouched upstream exactly in one runtime. The broader Mac
suite had 659 passes, 17 skips, 109 deselections and one cache-fingerprint test
failure reproduced on untouched upstream; it is not an all-platform green
package check. The private wheel's 46 Python files and both license notices
matched source bytewise, and CPU import did not import CuPy.

The implementation source is local commit
`d3da4e6d604af20223f3ecdcfb76a417fba5f8d0`. Paired job 9373881 and separate
memory/draw job 9374241 ran on B200 nodes dgx006 and dgx001. Both reserved the
cluster-mandated 28 CPUs/224 GiB per GPU, with inference pinned to one CPU.
The prior passed numerical preflight was 9373662; failed runtime setup attempts
9373189/9373519 remain recorded as failures. All jobs used the normal account
`ekatsevi-wharton`, QoS `wharton-dgx-b200`; no held or unrelated job changed.

Raw small receipts are in `tools/gpu/evidence/paired/` and
`tools/gpu/evidence/memory/`. Scientific result pickles stay ignored and local.
Product source/archive hashes bind the measurements; no merge, push or release
occurred.

## Opportunity ranking

The completed earlier Broad CPU profile at the same upstream source provides
the next priorities. These are exclusive boundary timers from earlier CPU
runs, not new GPU-node measurements and not the earlier R percentages.

| Component | Backed CRT seconds | RAM CRT seconds | Integration implication |
| --- | ---: | ---: | --- |
| Draw generation | 52.69 | 53.82 | Largest measured term. A GPU RNG path requires explicit stream/distribution qualification; keeping the law alone does not preserve individual draws. |
| Dispersion | 33.73 | 34.27 | Large numerical opportunity. Near-Poisson Newton paths are sensitive to reduction/special-function differences and need separate theta/fallback/statistic checks. |
| Pairwise QC | 24.50 | 2.24 | Backed I/O/counting matters; establish transfer and sparse-storage costs before GPU residence. |
| Test kernels | 20.16 | 20.94 | Existing indexed/CSR code already avoids dense draw matrices. The R component prototype does not measure this implementation. |
| Poisson fitting | 18.12 | 17.68 | Shared GPU IRLS can be extended, but preserve per-gene identity and qualify coefficient/dispersion changes. |
| Score reconstruction | 4.87 | 4.89 | Residency may help only when fitting and testing remain on-device; exporting all dense arrays can erase the gain. |
| Logistic fitting | 1.67 | 1.79 | Accelerating this term alone has a small workflow ceiling. |

The immediate recommendation is to keep `cuda_targets` experimental and avoid
a default switch. The largest follow-ups are sampling and dispersion. Neither
is implemented or statistically qualified by the target-fitting experiment.

## First-stage score batching follow-up

The optional `score_backend="cpu_batch"` and `"cuda_batch"` paths batch up
to eight genes, retain a bounded target-bank cache and supply first-stage
null statistics through the existing low-level test callback. Later stages,
weighted pairs and permutation-prefix routes retain their CPU paths. The
default `score_backend="cpu"` is unchanged. This follow-up is a diagnostic
of an architecture, not a recommended acceleration mode.

Job 9375748 ran six fresh RAM children on one B200 node, with one pinned CPU,
the same private Python/CUDA environment, exact frozen cell/row ordering and
the complete 1,024 calibration, eight power and 1,024 discovery pairs. Order
was CPU / CPU batch / GPU batch, then reversed. All children used the existing
CPU response/target fits, dispersion and sampler, seed 271828 and unchanged
adaptive stages. Inference and whole-child times were:

| Mode | First inference seconds | Reversed inference seconds | Mean inference | Mean whole child |
| --- | ---: | ---: | ---: | ---: |
| Untouched CPU | 171.906 | 171.815 | 171.860 | 203.191 |
| CPU batch | 195.300 | 193.191 | 194.245 | 224.763 |
| CUDA batch | 180.230 | 180.015 | 180.123 | 214.387 |

GPU batching was about 4.8 percent slower in inference and 5.5 percent slower
over the whole child than the untouched CPU. CPU batching was slower still.
The GPU path did execute 356 sparse matmuls per child with zero fallback
batches. It uploaded about 11.16 decimal GB of score vectors and 2.58 GB of
draw-bank buffers, downloading only 8.21 MB of null statistics. Keeping the
fitting/reconstruction on the CPU and accelerating only the first stage left
large costs intact; it is not evidence against a resident GPU design.

All returned columns, including p-values, observed statistics, effects,
adaptive stages and BH outputs, were exactly identical in each comparison;
each mode also repeated exactly. This does not mean that the intermediate
floating-point null scores were bitwise identical: first-stage empirical
counts can agree despite tiny score differences, and later-stage statistics
were computed by the unchanged CPU route. Focused actual-GPU validation
passed 73 tests. The local default validation passed 635 tests, with 22 skips
and 109 deselections, plus 29 gene-cache tests; the previously reproduced
upstream environment-sensitive numeric-cache-key test remained excluded.

Raw local receipts are under `tools/gpu/evidence/score-workflow/`, with
source-before/source-after checks and the column-by-column comparison.
The score preflight has a synthetic batch sweep and focused Nsight Systems
trace. Its Nsight Compute capture selected a reduction kernel, so it cannot
establish a roofline characterization of the sparse matmul.

The next design is described in [A resident GPU design](gpu-resident-design.md):
reorganize around sufficient-statistic fits, resident dispersion/score state,
stream-preserving sparse draw construction and requested-pair scheduling.
