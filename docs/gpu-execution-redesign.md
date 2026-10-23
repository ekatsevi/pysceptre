# Execution redesign for much larger GPU gains

The measured integrated path is a scaffold, not the intended final execution
architecture. Job 9377846 used 15,466 MiB observed VRAM on a roughly 180-GB
B200, with no retained-cache evictions. A larger GPU memory capacity is not
the missing resource for this frozen 201,158-cell experiment.

## Change the unit of work

The public pair table is a sparse graph: response genes on one side, assignment
targets on the other. Compile that graph into batches, fit each unique response
and assignment once, generate each target bank once, then evaluate its requested
genes together. Retain shared design, sparse counts, response parameters and
banks on the GPU. CPU orchestration should submit batches and collect compact
results rather than alternate with the GPU for individual genes and pairs.

A draw bank Z against a batch of response vectors is sparse-matrix times
matrix, not thousands of independent sparse-matrix times vector calls. Dense
BLAS/tensor-core performance is not the right hardware ceiling for sparse
FP64 gathers. Increasing batch width reuses indices and amortizes launches;
custom row reductions can exploit the narrow design dimension without padding
the pair graph into a fictitious all-gene/all-target screen.

For binary complement-control draws, let

    a_g = (y_g - mu_g)/(1 + mu_g/theta_g)
    w_g = mu_g/(1 + mu_g/theta_g)
    H_g = X' diag(w_g) X.

Each draw needs only three aggregates:

    s = sum_selected a_g
    q = sum_selected w_g
    v = sum_selected w_g X_cell
    z = s / sqrt(q - v' inverse(H_g) v).

A fused gather/reduce can construct these directly. It does not need to
materialize four separately transformed D planes for every gene/cell.
With four covariates, a/w plus a common X need about one third the storage
of a/w/four D planes, trading modest arithmetic for less memory traffic.
This identity is for binary draws; repeated-entry approximation and weighted
modes must retain their own original algebra. Tiny variance/conditioning
cases require the reference behavior.

## Fit batches, including dispersion

Poisson Newton updates can use fixed X'y and count-log-count sums. A GPU tile
reduces mean-dependent gradient/Hessian contributions over cells for many
genes, solves many tiny systems, and repeats without copying means to CPU.
The current implementation still downloads means and estimates dispersion
serially. That boundary must move.

For integer counts, digamma/trigamma differences admit finite reciprocal sums:

    digamma(theta+y)-digamma(theta) = sum(k=0..y-1) 1/(theta+k)
    trigamma(theta)-trigamma(theta+y) = sum(k=0..y-1) 1/(theta+k)^2.

The corresponding NB score/information can be evaluated from count histograms
and tiled cell reductions; batching matters more than swapping one special
function library. Preserve the existing initialization, iteration limit,
convergence criterion, MLE/MM/pilot preference and clamps. Algebraic identities
change rounding: a near-Poisson profile can be sensitive enough to select a
different numerical path. Validate final theta, flags, statistics, tails and
BH decisions; route sensitive cases through the reference. The guide-probability
tolerance is not authority to change these quantities.

## Generate the original random stream in parallel

NumPy's PCG64 transition is affine modulo 2^128. Precompute jump coefficients;
advance each GPU block to its starting state and generate its raw words in
parallel. NumPy exponential rejection sampling consumes a variable number of
words, but that does not require serial generation:

1. Classify every raw position as an exponential-attempt start.
2. A usual attempt consumes one word; an unusual attempt consumes exactly two.
3. A consumed uniform word must not itself be interpreted as an attempt.
   Validity is the parity of the immediately preceding run of unusual words.
   Independent local look-back, or a segmented scan, resolves that parity.
4. Compact valid accepted attempts in raw-stream order. The last requested
   accepted position determines the exact number of consumed PCG words.
   Discard speculative suffix words and advance the host state by that count.
5. Apply the original geometric inversion, per-cell padding, continuation and
   draw grouping on device. Return a resident CSR bank, not millions of gaps
   to be scanned/grouped on CPU.

This preserves the existing generator rather than assuming a cuRAND generator
has the same stream. Rare transcendental comparisons and integer boundaries
need explicit ambiguity detection and reference replay. The current prototype
proves bounded raw-stream/state/membership comparisons; it is not integrated
or generally qualified. Generator types, NumPy versions, uint32 cache state,
large-probability geometric search and approximate sampler paths need explicit
handling before public use.

Primary owning sources: [NumPy exponential implementation](https://github.com/numpy/numpy/blob/v2.5.4/numpy/random/src/distributions/distributions.c),
[PCG64 implementation](https://github.com/numpy/numpy/blob/v2.5.4/numpy/random/src/pcg64/pcg64.h).
Copied constants and algorithm notices are retained with the prototype.

## Complete the execution pipeline

Batch the existing skew-normal moment and tail checks across active pairs,
retaining the same checks and escalation decisions. Compact the active pair
set after each stage, then run the next stage as one batch. Keep the public
budgets and stream consumption contract: generating fewer random values is a
semantic change unless proved to preserve the existing contract.

A persistent execution context can reuse compiled kernels, uploaded design
and validated inputs across calibration/power/discovery calls. Cold-process
claims must still include initialization, loading and transfers. Current
whole-process CPU time averages 214.26 s: 100x would mean 2.14 s total, already
less than current loading alone. Thus a cold 100x claim also requires changing
input materialization/startup, and cannot be obtained by quoting warm kernels.
Inference averages 181.81 s: 10x requires all remaining work below 18.18 s;
100x requires below 1.82 s. These are acceptance targets, not measured forecasts.

A larger VRAM budget becomes useful when it prevents reloads for substantially
larger response/target families. Tile based on graph reuse and measured live
storage; use extra memory for wider batches and shared banks. Do not make a
larger-memory GPU a prerequisite while most of the available B200 is unused.
Measure each complete public family against both the existing reference and
an optimized multicore CPU baseline. Component gains do not establish an
end-to-end gain, and the current experiment does not establish 10x or 100x.
