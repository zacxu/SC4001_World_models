# P2.1 measured finalization

`finalize_phase21.py` aggregates completed evidence without starting simulators,
changing datasets, editing source, or replacing the original failed pipeline
report. Run it only after all four pilot datasets and the explicit relay/seed-1
continuation finish. Every required input must be promoted under the project
root so new evidence references use portable relative paths.

The finalized status is `complete_p21_pilot`; `completion_method` is
`verified_explicit_continuation`. The original raw pipeline report must still
have status `failed`, and its exact bytes/receipt are retained and referenced.
The separate finalized report describes the verified end result and recovery.

Required evidence includes:

- Four sealed 50,000-transition datasets, exactly 35,000 random and 15,000 scripted
  per case; one physical store shared by B/D with full per-arm attribution.
- Fresh production `verify` output plus full independent array and native-journal
  audits, all tied to current manifest, collection-report, and ledger hashes.
- Passed unit suite, measured production-data storage benchmark, and four passed
  1,000-transition native preflights.
- The aborted slow-drive diagnostic report, preserving its 1,222 physical
  decisions/attempts, and the passed preflight's 4,024 physical decisions.
- The original 81-file source snapshot, frozen recovery overlay/driver snapshots,
  completed driver report, and immutable prefix/failure/ledger lineage evidence.

The helper verifies the preserved 32,700-transition prefix once, then counts the
completed replay's 200,000 transitions once. It does not add the prefix again.
Reset synchronization is reported separately; diagnostics never enter replay.
The failed main orchestration lacks its final initial-collection resource summary.
The finalized report discloses that limitation and references real preflight,
continuation, and optional point-in-time affinity evidence.

Example, from the project root (replace audit/verification report names with the
actual newly passed outputs):

```powershell
.\.venv\Scripts\python.exe artifacts/phase21/reviews/finalize_phase21.py finalize `
  --project-root . `
  --data-dir data/initial_replay/pilot `
  --raw-report artifacts/phase21/phase21_acceptance.json `
  --production-report artifacts/phase21/verification/verify_report.json `
  --arrays-report artifacts/phase21/reviews/initial_replay_arrays_audit.json `
  --native-report artifacts/phase21/reviews/native_journal_audit_full.json `
  --unit-report artifacts/phase21/final_validation/unit_tests_report.json `
  --storage-report artifacts/phase21/storage_benchmark/storage_benchmark.json `
  --preflight-report artifacts/phase21/native_preflight_report.json `
  --prior-diagnostic-report artifacts/phase21_slow_d_journal_preflight/aborted_preflight.json `
  --original-source-index artifacts/phase21/verified_sources/source_index.json `
  --recovery-report artifacts/phase21/recovery_driver_report.json `
  --recovery-source-index artifacts/phase21/reviews/recovery_support/frozen_source_index.json `
  --recovery-driver-source artifacts/phase21/reviews/recovery_driver.py `
  --resource-observation artifacts/phase21/reviews/ongoing_resource_observation.json `
  --cleanup-plan artifacts/phase21/hot_journal_cleanup_plan.json `
  --output artifacts/phase21/phase21_acceptance_finalized.json
```

Use new output paths if a previous report exists. Inputs are never rewritten.
The source-index verifier accepts both the actual original snapshot schema and
the recovery snapshot schema; all pinned files and every additional retained
snapshot file are hash-checked.

## Exact hot-ledger cleanup

Finalization produces a plan containing the exact named temporary roots and every
retained hot ledger's matching durable cold copy. The old main-run relay/seed-1
hot ledger matches `recovery/<id>/before/training_interactions.jsonl`, not the
new 50,000-transition final ledger. Unknown files, changed hot bytes, changed cold
bytes, links, or unsafe root locations block planning. No deletion is performed.

After the finalized acceptance exists and all owned collection/continuation
processes have stopped, the operator may remove only the exact verified P2.1
temporary roots listed in the plan. Recheck the resolved absolute targets,
contents, and hashes before deletion. Data directories and historical reports
remain untouched.

Then generate a separate cleanup verification:

```powershell
.\.venv\Scripts\python.exe artifacts/phase21/reviews/finalize_phase21.py verify-cleanup `
  --project-root . `
  --cleanup-plan artifacts/phase21/hot_journal_cleanup_plan.json `
  --acceptance-report artifacts/phase21/phase21_acceptance_finalized.json `
  --output artifacts/phase21/hot_journal_cleanup_verified.json
```

This requires every named hot root to be absent and rehashes the cold evidence
copies, including the original failed-prefix ledger. It links the acceptance
and plan by hash. The earlier acceptance's pending-cleanup state stays unchanged
as historical evidence; the new verification records the completed cleanup.

`test_finalizer_artifact.py` contains five small synthetic contract tests. They
exercise aggregation and rejection paths only; they are not replay datasets or
measured native acceptance evidence.
