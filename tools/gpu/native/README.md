# Native backend experiments

See [the integrated report](../../../docs/gpu-native-integration.md) for the
scientific scope, exact baseline and measurements. All jobs use the frozen
private input; no data are included here. Run in a fresh experiment checkout
inside the requested B200 allocation: result paths intentionally refuse to
overwrite an existing experiment.

- `clean.sh`: reversed-order clean cold-process CPU/GPU pairs, post-run table
  comparison, then a separate empty-cache memory-observed child.
- `run.sh`: dedicated full sampler/fit/RNG audit; its times include benchmark
  hooks and are not the normal-workflow performance result.
- `warm.sh` / `wide.sh`: repeated persistent-session throughput, with explicit
  input/context reuse; see the report for the extra audit prelude in warm.sh.
- `profile.sh`: diagnostic cProfile run, not a primary benchmark.
- `smoke.sh`, `stages.sh`, `wheel-smoke.sh`: public family and numerical/stream
  guard gates. The wheel script expects `prototype.whl` in the checkout root.
- `summarize-clean.py`, `compare-clean.py`: reports from completed clean runs.

The clean benchmark retains all product-required input checks, QC, cache and
numerical guards. It removes only temporary sampler/fit hooks, the unrelated
fixture warmup and the manual calibration-QC audit. Source and outputs are
checked after primary timing. Memory observation and profiling remain separate.

The measured runtime files are source-hash bound. The experiments used these
scripts under `tools/gpu/resident/`; the checked-in copies remap that name to
`tools/gpu/native/` to preserve the earlier resident protocol. Original scripts
and private receipts are retained with the local evidence.
