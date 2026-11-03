# P2.1 completed: pilot real replay

The approved pilot scope is complete: **relay and cover, seeds 0 and 1, four stores of 50,000 joint transitions**. Each store contains exactly **35,000 random and 15,000 scripted** transitions and is physically shared by B/D. Both variants are attributed the full 200,000 initial transitions; images are not duplicated between the variants.

The primary result is [phase21_acceptance_finalized.json](phase21_acceptance_finalized.json), status `complete_p21_pilot`, completion method `verified_explicit_continuation`. Its receipt binds the completed evidence. The [original orchestration report](phase21_acceptance.json) remains failed as historical evidence. **P2.2-P2.5, world-model training, policy learning and final seeds have not started.**

| Family | Pilot seed | Joint transitions | Episodes | Fitting / diagnostic transitions | Full case GiB |
|---|---:|---:|---:|---:|---:|
| relay | 0 | 50,000 | 349 | 43,777 / 6,223 | 1.942 |
| relay | 1 | 50,000 | 350 | 46,729 / 3,271 | 2.028 |
| cover | 0 | 50,000 | 247 | 44,725 / 5,275 | 1.996 |
| cover | 1 | 50,000 | 246 | 47,604 / 2,396 | 1.940 |

Episode hashing gives an expected 90/10 fitting/diagnostic allocation. The realized counts above are retained; diagnostic episodes and every preflight store are excluded from fitting and imagination anchors. The 1,192 episodes contain 592 successes, 195 deadline endings, five deaths and 400 explicit collection cutoffs. These are collector outcomes. Both collector kinds cover all six training geometries in each family. No validation/test geometry or prior Phase 1 trajectory was imported.

## Implementation and verification

Typed, lightly compressed uint8 NPZ chunks store one observation sequence per episode, including chunk boundaries. State targets describe `o_t`; actions, rewards, termination and event deltas describe `t -> t+1`. Image, privileged-state and event validity remain independent. B/D bindings resolve to the same sealed replay and content identity. Synthetic PPO trajectories have separate provenance and cannot enter this real-replay store.

- [Current full suite](final_validation/unit_tests_report.json): **289 tests run, 288 passed and one Windows symlink-privilege-dependent test skipped**.
- [Production verification](verification/verify_report.json): all four complete stores passed using the current code.
- [Independent array audit](reviews/initial_replay_arrays_audit.json): **200,000 transitions and 51,147,366 scalar conditions** passed.
- [Independent native audit](reviews/native_journal_audit_full.json): **14,461,856 assertions**, 805,950 matching successful dispatch/ACK records and 2,386 verified runtime archives. The original failed startup control dispatch is accounted for separately.
- [Generation-end check](generation_end_verification.json): all **81 original execution/test files and the pinned runtime** matched their original identities before any post-collection source update. [The applied source record](reviews/post_collection_source_promotion_applied.json) identifies the tested future fixes; current execution/test sources number 85. All dataset hashes remained unchanged.

Actual terminal repeats are preserved: the native audit records **799,990 published simulator tics**, rather than assuming four tics for every terminal action. The direct [cover death checks](reviews/cover_seed_0_death_rows_audit.json) include final repeats of one and two tics, with valid terminal images and independent valid health/alive targets. Unavailable-terminal-image paths are also covered by the unit suite.

## Recovery, costs and resource evidence

Relay seed 1 encountered a native ViZDoom-extension access violation during startup after **32,700 committed transitions / 226 episodes**. That failed startup consumed zero physics decisions. The [explicit continuation](recovery_driver_report.json) completed the remaining **17,300** transitions with the original family, seed, episode/RNG sequence, mixture, SDK and game-generation code. Original payloads, metadata, failure records and the exact ledger byte prefix were preserved. The [frozen overlay snapshots](reviews/recovery_support/frozen_source_index.json) identify the recovery modules actually used. The native SDK defect itself was not repaired.

The original main resource observer separately encountered `psutil.AccessDenied` / Windows `OpenProcess` WinError 5 for PID 25004. Its fault time and process lifecycle were not recorded. Full original main CPU/RAM telemetry and continuous descendant-affinity coverage cannot be reconstructed. Configured four-CPU parent pinning, the point observation and separately measured windows are narrower evidence. The [historical assessment](reviews/resource_telemetry_review.md) states those limits. Original reports are retained unchanged.

Future collection/recovery now checks resource health before every real reset/step, reports errors and coverage gaps, preserves a primary simulator exception, and skips access denial only when a non-root child is proven departed or reused. [69 targeted tests and a Python-only inheritance probe](reviews/resource_enforcement_patch/resource_enforcement_patch_validation.json) passed before the full current suite.

The final ledger counts **200,000 published training transitions + 1,192 reset synchronization decisions = 201,192 physical decisions**. Separate implementation diagnostics add 4,024 passed-preflight and 1,222 aborted-preflight decisions/attempts, for **206,438 total charged physical decisions/attempts**. Their 5,216 published diagnostic transitions never enter learning replay. The recovered prefix is counted once.

Nine temporary C: ledger copies, **159.022 MiB**, are [retained by your explicit decision](hot_journal_retention_decision.json), each with a verified D: counterpart. The optional cleanup is closed with no deletion. The earlier [retention record](hot_journal_retention.json) and [cleanup plan](hot_journal_cleanup_plan.json) remain historical evidence.

## Storage, access pattern and future capacity

The [measured inventory](storage_inventory.json) is **7.014 GiB replay / 7.905 GiB full case directories** including runtime archives, ledgers and recovery evidence. [The codec benchmark](storage_benchmark/storage_benchmark.json) selected 128-transition chunks with deflate level 1 using real preflight pixels under four logical CPUs.

The [full-store reader profile](reviews/production_reader_benchmark.json) measured **751.70-807.18 joint observations/s** for contiguous 50-state windows, and **21.16-25.17/s** for shuffled single-state requests. Scattered reads decompress approximately 103-114 times the returned RGB bytes; the four-chunk LRU stays below 39.2 MB. The OS cache was prefetched, residency was not enforced, and two native jobs shared CPUs 0-3. The earlier 256-observation codec rate is not an arbitrary single-frame sampling guarantee. Batched loading, GPU transfers and actual learner consumption remain unmeasured; use the planned recurrent sequence access and profile the actual learner in later phases.

At the observed retention rate, 40 million transitions project to **1.698 TB decimal**, before checkpoints, evaluation, independent backups and working margin. D: had **791.6 GiB free at inventory time**. Provision additional archival capacity and a verified restore procedure before final-seed collection. Image entropy, behavior and episode lengths can change this extrapolation. See the [backup/capacity runbook](reviews/backup_capacity_runbook.md). No external backup destination was configured or transferred to in this task.

Raw case directories are ignored by Git. Preserve `data/initial_replay/pilot/` together with bindings, ledgers, runtime receipts and recovery lineage in a separately verified backup. Git retains code, dependency locks and selected evidence; `.gitattributes` preserves the exact bytes of P2.1 hash-receipted artifacts.

## Read or recheck the existing data

From the project root, use a new report directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase21 verify --output-dir artifacts/phase21_recheck --data-dir data/initial_replay/pilot --benchmark-report artifacts/phase21/storage_benchmark/storage_benchmark.json
```

The complete collector uses fresh output/data locations. Existing sealed data is read through its bindings; it is not recollected for later phases. For APIs, collection methodology and future recovery commands, see [the P2.1 runbook](../../docs/phase21_replay.md) and [the finalization workflow](reviews/FINALIZATION.md).
