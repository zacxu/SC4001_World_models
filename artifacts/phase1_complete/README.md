# Phase 1 accepted: P1.5-P1.7

**Status: `complete_phase1`.** The [acceptance report](phase1_acceptance.json)
records the implemented interfaces and measured gates. No model training was
started; every trajectory here is environment-validation data, outside learning
replay. The research comparison remains B versus D.

## Implemented

- **P1.5:** versioned privileged field/event registry, units and normalization,
  applicability masks, stable authored object identifiers, signed 16.16 wire
  conversion, native-tic cumulative events and decision deltas. One shared
  reward is calculated centrally; reserved engine reward remains zero.
- **P1.6:** separate true-terminal and collection-truncation paths, explicit
  per-player reconstruction masks, immutable final reward/event capture,
  geometry holdouts with six training/two validation/two test templates per
  family, versioned nuisance banks, and learning-ineligible diagnostic records.
- **P1.7:** full acceptance runner, exact fixed-action replay, native fault
  recovery and orphan cleanup, 10,000-decision/100-reset stress campaign,
  generic local-input isolation audit, mechanics videos and measured collector
  selection under a four-logical-CPU affinity budget.

References: [implementation/runbook](../../docs/phase1_complete.md),
[target registry](../../docs/phase1_targets.md),
[splits](../../docs/phase1_splits.md), and
[replay investigation](../../docs/phase1_replay.md).

## Acceptance results

| Required check | Measured result |
| --- | --- |
| Full unit suite | **159 passed** |
| Phase 0/configuration regression | [Passed](phase0_regression.json), including common B/D collector setting |
| Geometry allocation | 6/2/2 per family; 20 distinct geometries; 240 spawn/role/enemy-bank combinations statically validated |
| Native mechanics | **20/20 successes**, including all validation/test templates; relay 59-96 decisions, cover 107-193 |
| Cooperation sanity | All four each-player-inactive checks failed to achieve team success |
| Shared reward/events/targets | Native fractional fixed-point values observed; field, normalization, counter/delta, reward and validity checks passed |
| Transient event retention | Initial/final plate flags both zero; intermediate plate entry/exit and gate open/close counters each one |
| Boundary paths | Native success, death, deadline and cutoff passed; final reward/events survived reset |
| Infrastructure faults | Native-engine exit and hard Python-participant exit produced no reward-bearing transition; retry consumed zero; fresh recovery succeeded |
| Orphan cleanup | Hard participant termination exercised real cleanup of both captured old native engines, with PID/creation-time/executable guards |
| Fixed-action replay | Both families, fresh construction and public coordinated reset, three repetitions each: **state, events, rewards and all RGB hashes exact** |
| Stress | **10,000 published joint decisions, 100 coordinated resets**, 5,000/50 per family |
| Stress synchronization | 40,510 contiguous native dispatches matched 40,510 ACKs; **zero unexplained errors, stale responses or tic mismatches** |
| Actor isolation | Both players' own visual/action/memory histories held fixed while global metadata and teammate RGB changed; probe outputs/memories unchanged |
| Video export | Two genuine native mechanics videos; 336 source actor PNGs and decoded frame counts validated; 14 milestone frames visually reviewed |
| Provenance | Execution sources/runtime stable, package pins matched, scenario/registry/split/evidence hashes validated |

Missing terminal images remain `None` with a zero per-player reconstruction mask.
That conditional path passed unit/injection tests. All terminal RGB slots actually
observed in this native campaign were available; the report does not claim a
native missing-image event occurred.

## Collector benchmark and selection

Each family used an identical fixed layout, engine seed 7000 and no-op workload
for all counts/repetitions. Three alternating repetitions measured 128 steps per
environment after warmup. Physical workload traces matched across repetitions
and collector counts. Measurements include native stepping, RGB/target handling,
durable logging and diagnostic hashing; exclude startup/warmup/teardown.

| Family | Environments | Median joint transitions/s | Joint tics/s | p50 latency, ms | p95 latency, ms | Median CPU, % of four-CPU budget | Peak summed RSS, MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Relay | 1 | 21.40 | 85.59 | 48.80 | 70.67 | 19.90 | 225.49 |
| Relay | 2 | 37.09 | 148.35 | 45.47 | 68.91 | 27.04 | 401.85 |
| Cover | 1 | 19.95 | 79.80 | 56.95 | 71.72 | 19.58 | 226.81 |
| Cover | 2 | 36.58 | 146.32 | 47.57 | 66.34 | 26.66 | 403.66 |

**Selected two collectors**, consistently applied in `configs/base.yaml` for B
and D. Two-environment throughput CV was 2.63% for relay and 0.31% for cover.
Single-environment baselines also passed the predeclared stability rule. Median
fresh startup/reset duration was approximately 5.5 seconds per environment and
is reported separately. Summed RSS may count shared pages more than once; CPU
is measured over the acceptance parent and its workers/native children. Tics/s
counts joint-environment tics; summed participant tics/s is twice that value.

This is a measured Windows development-host window with four logical CPUs, no
GPU inference and no learner overlap. Rebenchmark on the actual training host
and under its eventual learner load. It does not validate the four H200s or
claim these absolute rates are stable at every time.

## Rejected measurement windows and reuse

An earlier variable-seed preflight was rejected for throughput variation. A
controlled fixed-seed preflight selected two collectors. In the first full
campaign, the relay baseline mixed a slow first trial with later faster trials
and again failed the unchanged stability rule. Those results were retained:

- [Rejected full campaign](phase1_acceptance_rejected_benchmark.json).
- [Rejected full benchmark](benchmark_rejected_report.json), with its original
  benchmark directory and all evidence bytes still present.
- Preflights and the read-only pacing investigation remain under
  `../phase1_completion/` and `../phase1_remaining/`.

Only the benchmark was repeated. The other seven completed stages were reused
after verifying their report receipts, unchanged source/runtime/parameter
fingerprints and every referenced evidence hash. [Retry provenance](retry_provenance.json)
records that process. The untouched program-generated completed aggregate is
[`../phase1_complete_retry/phase1_acceptance.json`](../phase1_complete_retry/phase1_acceptance.json).
The current canonical `benchmark_report.json` points to that accepted retry's
actual benchmark files. Previous measurement failures were not relabeled passes.

The causes of the host's performance regimes remain unassigned. No unsupported
FPS flags were changed or failed trials removed. Exact replay and matching
workload hashes establish simulation reproducibility for the tested cases;
they do not establish constant wall-clock performance.

## Accounting and independent review

The accepted evidence bundle consumed **18,810 joint-decision attempts**:
18,641 published transitions (including benchmark warmup), 167 initialization
decisions, and two deliberately failed dispatches. The 10,000 stress threshold
counts only valid published stress transitions. The rejected full benchmark
consumed a further 2,610 attempts, separately retained; development probes are
also separate. None entered training replay or implicitly charged a B/D learning
budget. Future use must obey the explicit counted-import and split rules.

- [Target/event audit](target_event_audit.json): all 14,107 persisted mechanics,
  stress and final-boundary rows; 6,485,108 checks, zero failures.
- [Replay/runtime audit](replay_runtime_audit.json): exact replay, full stress
  decision/dispatch journals, and both real crash paths.
- [Split/mechanics audit](split_mechanics_audit.json): all geometries and compiled
  scenario bytes, all four inactive checks, videos and all 336 source PNGs.
- [Root visual review](reviews/mechanics_visual_review.json) and
  [contact sheet](reviews/mechanics_contact_sheet.png).
- [Verified source snapshot](verified_source_snapshot_index.json): 79 execution
  and documentation files, preserving the campaign's code and contracts.

Locality is presently an adapter/contract test with a deterministic probe, not
a trained B/D policy result. Run the same audit on deployment actors in Phase 3.
The split tests measure unseen geometry within the two authored task topologies,
not arbitrary-map or new-enemy generalization. Scripted successes are environment
diagnostics, not a claim that learned agents already solve the tasks.

## Mechanics videos

- [Relay: native two-player gate/latch/extraction](mechanics/relay_train_00/video/relay_mechanics.mp4).
- [Cover: native two-console combat/extraction](mechanics/cover_train_00/video/cover_mechanics.mp4).

Videos are lossy H264 review exports with external captions. Exact 128x96 PNGs,
RGB hashes and trajectory records accompany them. Natural spawn-effect occlusion
is retained in initial frames rather than replaced with fabricated pixels.

## Reproduce

From the project root, use a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase1_acceptance verify --output-dir artifacts/phase1_complete_rerun
```

Native execution needs Windows message queues/local UDP permissions. See the
completion runbook for individual stages and strict `--reuse` behavior. A report
is complete only after every gate passes, including benchmark stability and
agreement with the shared configured collector count.
