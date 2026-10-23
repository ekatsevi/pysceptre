# Experimental CUDA rectangles

`iter_cartesian_analysis` and `iter_cartesian_crt_analysis` yield bounded
association tables for explicitly selected CUDA execution. Both support binary
union treatment, complement controls, the full score statistic, and a caller's
complete gene-by-target eligibility mask. Existing CPU entry points and defaults
remain unchanged. CUDA failures raise rather than silently selecting another
backend.

```python
from pysceptre import iter_cartesian_crt_analysis

telemetry = {}
for frame in iter_cartesian_crt_analysis(
    responses, gene_ids, design, target_cells,
    eligible_pairs=eligible,
    seed=123,
    n_jobs=8,
    precompute_backend="cuda",
    tail_backend="cuda",
    telemetry=telemetry,
):
    save_batch(frame)
```

Collect the complete intended family before correction. The helper functions in
`pysceptre.pipeline.complete_family` reject missing, duplicated and ineligible
positions and provide global Benjamini-Hochberg adjustment. Explicit NaNs remain
recorded; its default multiplier excludes NaNs, as R's default `p.adjust` does.
No correction is applied separately to iterator batches.
`global_bh_cuda` explicitly selects CUDA sorting and prefix minima after
complete-family collection. Its default returns adjusted values to CPU;
`return_device=True` keeps them on CUDA. `max_bytes` checks a conservative
work-array estimate before uploading the family, not total process memory.

## Computation and memory

Permutations share one unchanged CPU-generated ordering pool. Null distributions
are reused by gene and treated count. Hybrid layout computes a regular first
stage and compacts later-stage demand. Direct, sequential-prefix and explicitly
selected parallel-prefix scans read the same draws. The parallel scan reorders
FP64 additions. Tail summaries can run on CUDA while the original SciPy CDF and
survival function evaluate probabilities on CPU.

The CRT retains target-specific banks. Target-name seeds, reference binomial
fit cohorts, total draw budgets and original exact/approximate sampler choices
remain explicit. All stages are generated together. Lossless integer packing
retains listing order and multiplicities. Compact a,w planes replace the full D
atlas through the inverse-information variance formula. CUDA warps traverse
genes together, reusing draw listings with contiguous accesses. The first stage
counts empirical exceedances without materializing its score rectangle. Later
stages batch active pairs into bounded distribution arrays.
`query_strategy="draw"` schedules adjacent lanes over draws of the same active
pair, retaining the listing order within every score and writing contiguous
distribution rows. The default `"grouped"` strategy schedules genes together.
Both consume the identical target-specific listings.

The CRT persistent storage bound includes its compact atlas and packed banks.
It excludes CUDA runtime state, allocator reserves and temporary workspaces.
Estimated or actual capacity violations raise. This route requires enough device
memory for the supplied family; a streaming bank implementation is future work.
Permutation tiles require substantially less resident storage.
The CRT route currently permits at most 16 design columns and int32 cell
positions. Its CPU fits still materialize the dense cell-by-covariate outer
products. Large categorical designs therefore require further work even when
the compact GPU score atlas fits in device memory.

## Models, numerical checks and checkpoints

CPU width-one gene fitting is the default. `fit_backend="cpu_fork"` explicitly
uses the original CPU fitter in Linux processes and requires a fresh process
before importing a GPU runtime. The permutation iterator also exposes the
experimental `cuda_native` fitter, whose numerical qualification depends on the
workload. The CRT iterator currently retains CPU gene and guide fits. These
routes are hybrid workflows, not fully GPU implementations.

CUDA precomputation and tail summaries reorder FP64 operations. Numerical guards
request a complete original CPU pair replay for ambiguous variances, comparisons
and tail decisions. Replay bitmaps accompany output batches. A permutation
replay scans only ranks needed by replayed targets while preserving the reference
full-width decision between prefix sums and sparse multiplication. Close
numerical agreement is distinct from bitwise equality and from calibration or
power validation. Synthetic regression checks cover stages, masks, decoded draw
identity, query compaction and checkpoint output equivalence. Real-workload
qualification and measured benefit must be established separately.

`gene_indices` selects emitted genes. Supply the complete intended target family
in its original order. CRT `target_chunk_size` and `chunk_memory_gb` must match
the reference model cohorts; changing them can move fitted-probability last bits.
`skip_gene_indices` skips persisted output after refitting the full selected
cohort. The consumer owns atomic persistence, source/settings identity checks
and whole-family completeness validation. `on_fit` exposes small fitted-model
records and `on_bank` on CRT supports independent draw auditing.

Pairwise quality control remains a CPU boundary. These iterators consume its
eligibility mask; no GPU pairwise QC method is selected or introduced.
