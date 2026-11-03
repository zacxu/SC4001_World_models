# Independent array audit

`audit_replay_arrays.py` directly decodes typed NPZ payloads one episode at a
time. It recomputes state/transition indexing, cumulative event differences,
shared rewards, player/task/enemy normalization, masks, stable episode partitions,
paired portable pointers, and archived member hashes. The production reader is
used only to exercise fit/imagination access guards. Executed native actions and
ACK accounting are audited separately by the native-journal review helper.

The helper restricts its process to logical CPUs 0-3 and sets numeric-library
thread environment variables to 1 before importing NumPy. It starts no simulator
and does not mutate datasets. Use a new output JSON for each run.

```powershell
.\.venv\Scripts\python.exe artifacts\phase21\reviews\audit_replay_arrays.py --datasets data\initial_replay\pilot\relay\seed_0 data\initial_replay\pilot\relay\seed_1 data\initial_replay\pilot\cover\seed_0 data\initial_replay\pilot\cover\seed_1 --expected-transitions 50000 --output artifacts\phase21\reviews\initial_replay_arrays_audit.json
```

`native_preflight_arrays_audit_v2.json` passed all four 1,000-transition native
preflights: 4,000 transitions, 1,023,041 scalar conditions, four CPUs, and an
observed peak RSS of approximately 122 MiB. Every preflight episode happened to
hash to the fitting partition, but the entire preflight stores correctly reject
learning access through their dataset-level eligibility guard.

The first audit attempt and `audit_replay_arrays_v1.py` are preserved. That attempt
stopped on the audit's file-order comparison: ordinary string sorting differs
from Windows Path sorting for mixed-case scenario filenames. Version 2 compares
canonical case-folded filename order while retaining exact member names and
hashes. No dataset files changed and no simulator rerun was needed.


## Completed full pilot evidence

The current [array audit](initial_replay_arrays_audit.json) passed all 200,000 transitions and 51,147,366 scalar conditions; [the native audit](native_journal_audit_full.json) passed 14,461,856 assertions, including the explicit zero-physics startup lineage. Use [the finalized acceptance](../phase21_acceptance_finalized.json) and [results index](../README.md) for the full evidence chain.

The [full-store reader benchmark](production_reader_benchmark.json) separates contiguous windows from scattered states. The [telemetry review](resource_telemetry_review.md) explains the original main observer's missing coverage. The [resource fix](resource_enforcement_patch/resource_enforcement_patch_validation.json) and [applied source record](post_collection_source_promotion_applied.json) describe post-collection changes; generation-era snapshots remain frozen separately.

The native helper defaults to checking retained hot ledger mirrors. If those copies are later removed under the explicit verified cleanup plan, add `--allow-cleaned-hot-ledgers` for a new cold-only native audit and retain the earlier full mirror audit and cleanup verification. Always choose new output paths; do not overwrite historical reports.
