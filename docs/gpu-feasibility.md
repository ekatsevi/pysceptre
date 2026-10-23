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

Measured results and explicit limits are to be filled from completed receipts.
