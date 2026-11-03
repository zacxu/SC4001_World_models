# Explicit initial-replay continuation

The installed entry point is `python -m world_models.data.recovery`. It continues
the same family, training seed, logical episode sequence, action RNG streams,
mixture schedule, model-diagnostic partition, and original native SDK. It does
not discard/recollect a valid prefix or replace an unsuccessful seed.

Continuation is allowed only after a recorded reset/startup failure that consumed
zero physics, with no active or uncommitted replay episode and no failed action
dispatch. Original payloads, episode manifests, runtime archives, and hot/cold
ledger mirrors must verify. The requested failure report must match its explicit
SHA256. Original source/runtime provenance and actual loaded game-generation
modules/configs must still match. No dependencies are upgraded by this command.

Inspect first, using paths appropriate to the current checkout:

```powershell
.\.venv\Scripts\python.exe -m world_models.data.recovery `
  --dataset data/initial_replay/pilot/relay/seed_1 `
  --recovery-id startup-continuation-001 `
  --failure-sha256 <recorded-collection-error-sha256> `
  --origin-project . `
  --inspect-only `
  --output artifacts/phase21/recovery-inspection.json
```

To continue, remove `--inspect-only`, supply a fresh empty `--journal-root` on
the fast local drive, and choose a separate output report. Native execution
requires the same local UDP and Windows message-queue access as collection.
Do not exceed the shared four-CPU/two-joint-environment allocation when scheduling
this command alongside another collection job. The command enforces four logical
CPUs and one numerical-library thread and monitors its process tree.

`--max-startup-attempts` accepts only 1 or 2 (default 2). A fresh-pair retry is
allowed only when startup failed before any physics dispatch. Every attempt and
failure stays in the durable ledger and archived runtime evidence. A failed
action, initialization that consumed physics, or active partial episode stops
without an automatic retry and needs a separate recovery/budget decision.

Before any append, `recovery/<id>/before/` retains exact original manifest,
error, progress, specification, and ledger snapshots. `lineage.json` records
their hashes, original payload hashes, source proof, and continuation code
provenance. Old payloads and original dataset metadata remain unchanged. New
runtime attempts use a separate namespace; historical failed startup archives
are indexed separately from successful replay episodes.

Progress keeps the global prefix-plus-new transition count and identifies the
recovery ID. Final elapsed fields distinguish:

- `original_prefix_elapsed_seconds`: the frozen pre-failure progress report.
- `original_failed_collection_attempt_elapsed_seconds`: original attempt time,
  including its failed startup.
- `original_failed_startup_elapsed_seconds`: the difference between those views.
- `continuation_elapsed_seconds`: this explicit continuation, including validation.
- `total_collection_attempt_elapsed_seconds`: original attempt plus continuation;
  `elapsed_seconds` is an alias of this total.

Waiting or debugging between attempts is excluded from those active-attempt
times. CPU/RAM monitoring for the continuation is reported separately from the
original campaign's resource report. Initial synchronization and every published
or failed action remain separate explicit physical-interaction counters.

For the currently running original campaign, use the artifact
`recovery_driver.py` rather than changing live D: source files. It overlays only
the staged ledger/replay/recovery modules in its own parent process; native
workers, policies, schema, and compiler resolve from the untouched original
project. It records module paths/hashes and verifies source stability. Spawned
workers do not rerun the driver. Future datasets collected under the deployed
current source can use the installed module CLI directly.
