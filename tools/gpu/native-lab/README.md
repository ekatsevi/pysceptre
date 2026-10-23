# Experimental parallel sampler and batched dispersion

These are isolated research prototypes, not public pysceptre backends. They
are deliberately separate from the validated resident workflow in eb70553.
They investigate remaining serial boundaries; none of their speedups is a
measured whole-workflow gain.

## Sampler

`rng.cu` generates NumPy PCG64 XSL-RR raw words by parallel affine jumps modulo
2^128. The exponential classifier implements NumPy 2.5.4's Ziggurat constants
and consumption logic. Valid starts have even parity of the immediately
preceding run of unusual raw words. This resolves rejection consumption
without serial generation; compaction preserves stream order. `core.py`
returns requested exponentials and advances the host state by exactly the
consumed prefix, discarding a speculative suffix. The accepted exponential
values may differ by one ULP in rare transcendental tails; geometric integer
values and RNG state, rather than arbitrary floating exponential bits, are
the relevant bounded comparison here.

`bank.py` retains the CPU reference's cell padding/block boundaries and
continuation order, performs geometric inversion and segmented sums on GPU,
then sorts packed (draw,cell) keys into resident CSR indices/offsets. It
includes CPU planning, GPU work, synchronization and final cursor transfers;
verification downloads are outside its timing, since the intended consumer
is resident GPU scoring. The CPU comparison times the unchanged complete
sampler returning its ordinary list representation. Device CSR values are
implicit ones; public adapter integration and memory accounting remain work.

The prototype **refuses**, rather than silently changes, p>=1/3, non-PCG64,
or cached-uint32 states. It detects close wedge acceptance and geometric
integer-boundary cases. A refusal is not an integrated fallback: a caller
must restore its original RNG state before reference replay. Inputs and
integer saturation need further general validation. It does not implement
the approximate sampler, weighted draws, arbitrary NumPy versions or other
bit generators. Do not install this as an automatic sampler replacement.

Tests run under NumPy 2.5.4, CuPy 13.6, CUDA 12.8.1 on B200 hardware:

- Raw words matched at 1/255/256/257/100003/1000000-word boundaries for four
  seeds. Exponential decoding preserved final RNG states for four seeds and
  four sizes; tested geometric p=.00201/.01/.03/.1/.3 values matched exactly.
- Job 9379051, 45-GB B200 slice: synthetic 201158 cells / 5498 draws,
  13,835,307 bank memberships, exact bank/state in two observations,
  GPU 0.08370/0.08314 s vs CPU 0.71958/0.71086 s (8.60/8.55x).
  The first tiny CPU observation included Numba compilation and is excluded
  from any speedup claim. CPU wins the warmed tiny case.
- Job 9379139, full B200: four actual fitted target probability vectors,
  201158 cells / 5498 draws, two observations each. All complete bank and
  final-state comparisons exact. GPU 0.0155–0.0325 s vs single pinned CPU
  0.3658–0.5466 s, observed per-case ratios 16.82–27.08x. This includes host
  probability upload and CPU planning; excludes device initialization,
  compilation, probability fitting, final verification download and the rest
  of the public workflow. The first target includes additional allocator
  warmup. This is not a statistically repeated hardware benchmark.

`test.py` contains the original 3%-plus-1024 speculative buffer experiment;
`core.py` uses 6%-plus-1024 to avoid avoidable replay when random consumption
exceeds the former buffer. Both enlarge and replay from the original state
when necessary. `realtest.py` and `bank.py` exercise the latter version.

Constants and logic were checked against NumPy tag v2.5.4:
https://github.com/numpy/numpy/blob/v2.5.4/numpy/random/src/distributions/distributions.c
https://github.com/numpy/numpy/blob/v2.5.4/numpy/random/src/distributions/ziggurat_constants.h
https://github.com/numpy/numpy/blob/v2.5.4/numpy/random/src/pcg64/pcg64.h

NumPy and PCG notices are retained in LICENSE-NumPy.txt and LICENSE-PCG.txt.
`rng.cu` is the directly compiled source; its checked-in constants suffice
without a build-time download.

## Dispersion

`theta.py` batches original-form score/information reductions and retains
small SciPy digamma/trigamma tables. Job 9379190, 32 actual response genes,
45-GB slice: max theta difference 4.2633e-14, no nonconverged MLEs. Warm times
0.211/0.189 s including uploads versus CPU 0.389 s. This shows that retaining
CPU table work and transfers is still an expensive boundary.

`thetafast.py` instead builds reciprocal prefix tables on GPU using exact
integer-count special-function identities, and fuses cell score/information
reductions. It takes already-resident means and counts. Algebraic identities
change floating rounding and require sensitivity validation. Only the MLE
stage is implemented; original MM/pilot fallbacks, clamps, fractional counts,
near-Poisson cases and full-family downstream decisions are not qualified.
Neither dispersion prototype is integrated into the public engine.

Private actual-data-derived receipts remain outside tracked fixtures.

Job 9379276, 45-GB B200 slice, the same 32 real genes: fully resident warm
MLE took 0.004289 / 0.004084 s versus CPU 0.413481 s (96.4 / 101.2x).
Maximum absolute theta difference was 1.4496e-12, with zero failed MLEs and
matching iteration counts against the preceding original-form GPU test.
This comparison excludes uploading counts/means, which are already resident
in the proposed pipeline, and excludes output transfer and downstream tests.
The first GPU invocation, including additional JIT work, took **10.1408 s**.
It does not demonstrate a cold-process or full-workflow speedup, nor validate
all calibration genes, fallback cases, or final p/BH decisions.
