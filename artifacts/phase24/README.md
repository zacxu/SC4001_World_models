# P2.4 implementation and server handoff

P2.4 is implemented. Research fitting is left to the user on the GPU server.
The default is four independent parallel workers sharing logical GPU 0, fitting
relay/seed 0, relay/seed 1, cover/seed 0 and cover/seed 1. Each completed fit
exports B/D bindings to one shared initial checkpoint and frozen normalizer.

The implementation provides episode-safe recurrent windows, correct absolute
clocks, detached burn-in, effective-batch reductions across memory-saving
microbatches, AdamW/clipping, float32 parameters and optimizer states with
bfloat16 autocast, immutable SHA-receipted checkpoints, and optimizer/RNG/sampler
resume. Independent fitting-data diagnostics, real validation-geometry banks,
16/32/64/full-prefix context comparisons, and branch refresh are included.

Use the [server runbook](../../docs/phase24_training.md). It contains the exact
small overfit and full four-worker launch commands, monitoring, recovery,
validation collection, diagnostics, later refresh prerequisites, artifact paths
and stop conditions. `python -m scripts.phase24 plan` does not launch training.

Local evidence:

- P2.4 checks: **61 tests run, 60 passed and one CUDA-only test skipped**.
- Full repository suite: **447 tests run, 443 passed and four skipped**.
- [P2.4 implementation checks](phase24_acceptance.json) and [test log](tests.txt).
- [Full regression report](regression_tests.json) and [test log](regression_tests.txt).
- The tests include tiny synthetic forward/backward and optimizer fixtures,
  numerical full-batch/accumulation comparisons, and exact interrupted/resumed
  toy-model continuation. These are not pilot training results.
- The four initial stores and their frozen statistics were checked by the
  nontraining launch-plan path. No research replay was fitted or recollected.

The measured host uses Python 3.12 and PyTorch 2.15 CPU. CUDA continuation,
compiler-dependent scenario checks and Windows symlink behavior have explicit
skip records where the necessary host capability is unavailable. No GPU
training throughput, four-worker GPU memory footprint, learned predictions,
policy performance or native validation-bank collection is claimed here.

Research fitting, held-out burn-in analysis on real validation layouts, and
P2.5 predictive-quality acceptance remain user-run work. Completed fitting
schedules report `complete_fit_stage`; they do not establish P2.5 acceptance.
Keep all server run directories, checkpoint/normalization identities, validation
banks, interaction accounting and failed attempts separately from this small
versioned implementation-evidence folder.
