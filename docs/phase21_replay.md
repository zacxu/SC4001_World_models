# P2.1: collect and store real replay

P2.1 implements real replay and its initial collection. P2.2-P2.5 world-model
implementation/training and Phase 3/4 policy learning are separate work.
The approved initialization remains 50,000 published joint transitions for each
task-family/run-seed pair, with exactly 35,000 random and 15,000 scripted decisions.
Only pilot seeds 0 and 1 in relay and cover are collected at this stage. Final
seeds 1001-1010 remain untouched until the protocol is frozen after pilots.

## Commands and outputs

Use the existing Python 3.12 environment installed from
`requirements-phase1.lock`, the editable project, and `scripts.bootstrap_acc`.
P2.1 uses those pinned NumPy/ViZDoom/Pillow/psutil versions and the standard-library
ZIP/deflate codec; no GPU or PyTorch dependency is introduced for collection.
Run commands from the project root. The completed stores use `verify`; the
`complete` example below creates a separate fresh campaign in new directories.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m scripts.phase21 complete --output-dir artifacts/phase21_new_campaign --data-dir data/initial_replay/new_pilot_campaign
```

`complete` runs the full unit suite, collects four separate 1,000-transition
preflight datasets, verifies them, measures storage using their real random and
scripted images, then collects and audits all four 50,000-transition datasets.
The preflight is implementation validation, cannot enter fitting/imagination,
and its interactions are reported separately from the four initial datasets.
The complete collection is limited to two concurrent joint environments and four
logical CPUs, with library thread counts fixed to one. Native execution requires
Windows message queues and local UDP access.

The current pilot is completed. Its primary report is
`artifacts/phase21/phase21_acceptance_finalized.json`, with a SHA256 receipt and
status `complete_p21_pilot`. The original `phase21_acceptance.json` remains failed
as historical evidence. See [the measured results](../artifacts/phase21/README.md)
for the explicit continuation, separate resource-observer failure, and accounting.
This does not mark all of Phase 2 complete. Reports and the storage benchmark
summary are versioned; raw datasets, journals, generated scenarios and preflight
stores are ignored by Git. Back them up separately.

For a smaller standalone preflight, use a new data/report directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase21 preflight --output-dir artifacts/phase21_preflight --data-dir data/initial_replay/preflight
```

To verify the full pilot stores without collecting them again:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase21 verify --output-dir artifacts/phase21_recheck --data-dir data/initial_replay/pilot
```

Output directories are never silently overwritten. `--reuse` permits reusing a
verified complete dataset only when its origin code, runtime, storage choice and
journal mode match the current frozen collection procedure. This cache guard
does not prevent later model-training code from reading an immutable B/D binding.
An interrupted/failed attempt retains its partial
data, ledger, error record, and engine journals. Such attempts are not silently
resumed or erased. Any replacement campaign must preserve and report their
consumed interactions separately. The explicit verified continuation utility can
resume a fully committed prefix after a zero-physics startup failure. It preserves
original evidence, action/seed sequencing and accounting, refuses an active partial
episode, and never automatically retries a physics failure. Current pilot stores
are sealed; use `verify` to inspect them.

## Collection method and attribution

The initial RNG uses the existing paired `initial_collection` seed namespace,
which is identical for B/D and independent across task families and run seeds.
Layout nuisance, engine, and policy seeds are derived independently for each
stable episode ID. The training-only builder cycles through the six approved
training geometries; validation/test geometries are rejected. Spawn, texture,
enemy, and role randomization comes from the versioned training banks.

Collection alternates quotas of 700 random and 300 scripted transitions until
the exact 70/30 total is reached. Each actor's random action is uniformly sampled
from the twelve declared actions and held for 2-8 decisions, with independent
actor streams. Scripted collection uses the existing privileged waypoint
controller. Cover cycles standard, no-cover-fire, same-operator, and dwell-pause
behaviors independently of the six-layout cycle, providing both normal mechanics
and varied combat/activation outcomes. Script actions are recorded as executed
environment actions, never as actor-imitation targets.

Episodes end on success, death, failure, deadline, or an explicit external
cutoff at a mixture quota/collection-block boundary. Cutoffs preserve valid
final observations and value bootstrap; they are not task failures. Collection
never joins two reset episodes or fabricates an infrastructure-failure reward.

Each family/seed output contains one `replay/` dataset and two tiny
`bindings/B.json` and `bindings/D.json` files. Both bindings reference the same
sealed content hash using portable relative paths. Each arm is attributed the
full 50,000 initial real joint transitions; there is only one physical collection
and one image store. Branch-specific replay and synthetic PPO trajectories
belong in separate stores in later phases.

The durable `training_interactions.jsonl` ledger debits every published random
or scripted decision before replay writing. Failed non-initialization dispatches
remain charged even if no transition can be returned. Reset synchronization
dispatches are separately conserved as physical simulator overhead. The report
distinguishes research decisions, replay-committed decisions, initialization
dispatches, failed attempts, native tics, and physical totals. A storage failure
cannot make an already executed action disappear from accounting. Prior Phase 1
validation trajectories are not imported into learning replay.

## Bounded durable journals and drive performance

The native coordinator fsyncs each dispatch/acknowledgement and the training
ledger fsyncs each charged decision. Keep that durability. On the relocated
development machine, the first D:-local preflight exposed high small-write
latency; the stopped diagnostic run and all consumed interactions are retained
under `artifacts/phase21_slow_d_journal_preflight/`.

The CLI therefore creates an exclusive hot-journal spool in the OS temporary
directory, normally C: on Windows. `--journal-root` may select another parent.
RGB replay chunks always go directly to the dataset directory on D:. Closed
native runtime/scenario trees are copied to D:, fsynced, and checked against
SHA256 inventories before their hot originals are removed. The final training
ledger is archived and verified on D: on success or failure. Original journal
identity paths are preserved as historical provenance; runtime archive mappings
resolve them to the portable D: archive directories.

The default capacity guard is 512 MiB per dataset, with separate initial free-space
provisioning. It bounds journal working space, not image replay. Archive failure
retains the hot source and partial destination for inspection; no temporary-folder
context silently deletes unarchived evidence. The I/O probe in a complete campaign
records actual small durable-write samples at both journal and replay locations.
Pass earlier diagnostic summaries with `--prior-diagnostic-report` to retain their
resource accounting in a replacement campaign. This changes file placement and
archive handling, not game rules, action timing, data budgets, or fsync frequency.

## Storage and alignment

Each episode contains typed NPZ chunks and a JSON manifest. The complete dataset
manifest indexes episode manifests, chunk hashes, the typed schema, scenario
identities, and collector provenance. NPZ members are numeric or fixed-width
Unicode arrays with `allow_pickle=False`; object arrays are rejected.

RGB is `uint8 [observations,2,3,96,128]`. There are exactly L+1 stored joint
observation slots for L transitions, including across chunk boundaries. A chunk
may reference a source observation in its preceding chunk; the image is stored
once. Player/task/enemy states and cumulative counters at index t describe o_t.
Actions, reward, terminal flags, and event deltas at index t describe t->t+1.
The writer and independent audit verify engine tics and event differences.

Raw and normalized player/task/enemy targets have independent validity masks.
Missing terminal image slots have false image masks; `EpisodeView.observation`
returns None. Fixed-shape invalid slots contain storage padding only, never a
valid observation. Valid reward/events/state targets survive independently of
image availability. No nonterminal missing observation is accepted.

Sealing records manifest receipts and a content identity; sealed writers cannot
append/reopen. Readers verify manifests, chunks, typed shapes/dtypes, finite
values, episode boundaries, target alignment, and declared real-data provenance.
The default reader retains four chunks in an LRU cache; full-dataset verification
streams episodes. Materializing `.arrays()` loads one selected episode, not all
replay. Do not preload the full final campaign into RAM.

## Fitting and model-diagnostic separation

Episode IDs determine a stable partition: SHA256 of the schema version and
episode ID, modulo ten, assigns one bucket to model diagnostics. The expected
episode allocation is 90% fitting / 10% diagnostic; realized episode and transition
fractions are reported and can differ from precisely 90/10. An episode never
straddles both partitions.

`ReplayDataset.load_episode(..., purpose='world_model_fit')` and
`purpose='imagination'` reject diagnostic episodes. Their iterators expose only
fitting episodes, and valid imagination-anchor indices exclude terminal states
and missing image slots. Preflight datasets are ineligible for either learning
purpose. Diagnostic reads remain available for integrity checks and codec
benchmarks. Evaluation geometry, synthetic trajectories, and legacy validation
provenance cannot be silently relabeled as ordinary real training replay.

## Storage benchmark and acceptance scope

The benchmark compares chunk sizes 64/128 and stored/deflate-1/deflate-3 on the
same hashed native pixel sample. It measures durable write and shuffled
64-observation read throughput, latency, CPU/RAM and exact payload equality under
the four-CPU budget. Complete uses fresh mixed preflight replay, so a fresh clone
does not require the old Phase 1 PNG/video bundle.

The selected default is chunk size 128. A compressed setting must save at least
10% versus stored and sustain at least twice the provisional 40-joint-observation/s
collector target in both write and read tests. The lowest qualifying deflate
level is preferred; otherwise stored is used. Other chunk sizes remain reported
comparators. Timing is for the production image codec, not an eventual GPU
learner; the full collection separately measures total collection resource use.
Read measurements keep the OS page cache warm and do not claim direct disk I/O.

The final independent audit verifies every stored transition's source/next
alignment, event deltas, shared reward rule, masks, mixture, episode partition,
training geometry, interaction-ledger conservation, and physical B/D sharing.
Full data must cover all six training layouts with both collector kinds and
contain fitting/diagnostic episodes plus the relevant success/failure/milestone
events. These are data-integrity/coverage checks. Learned autonomous rollouts and
policy usefulness remain P2.2-P2.5 acceptance work.


## Completed pilot and current operating notes

All four approved stores contain 50,000 joint transitions with the exact 35,000/15,000 random/scripted mixture. Their immutable B/D bindings share one physical store per case. The current full suite ran 289 tests (288 passed, one Windows symlink-privilege-dependent skip); full production, array and native-journal audits passed. Source generation and dependency identities were preserved before post-run fixes. P2.2-P2.5, policy training and confirmatory seeds remain separate work.

The original main observer stopped on an AccessDenied process-affinity query; its historical fault time/lifecycle and continuous coverage are unknown. Keep the original failed report. The separate SDK startup failure and original 32,700-transition prefix are also preserved. See the [telemetry assessment](../artifacts/phase21/reviews/resource_telemetry_review.md) and [completed recovery](../artifacts/phase21/recovery_driver_report.json). Later fixes check resource health before every reset/step, retain failure reports, preserve the original exception, and treat only proven departed/reused non-root identities as benign.

For a future eligible failed initial prefix, use `python -m world_models.data.recovery` with an explicit original-failure SHA256, matching original source/runtime, a new recovery ID and fresh journal root. Inspect before appending. Follow [the recovery workflow](../artifacts/phase21/reviews/recovery_support/RECOVERY.md); its example of the current relay/seed-1 prefix describes the historical failed state, and current sealed stores reject reopening.

The full-store production-reader profile differs from the small codec microbenchmark: contiguous 50-state windows measured 751.70-807.18 joint observations/s, while shuffled individual states measured 21.16-25.17/s because each miss decompresses a larger chunk. These prefetched-OS-cache measurements shared CPUs 0-3 with native collection; residency was not enforced. GPU transfer and actual learner throughput have not been tested. Use contiguous recurrent samples and profile real batching in P2.4-P2.5 before changing storage or claiming GPU efficiency.

The completed case directories occupy 7.905 GiB, including 7.014 GiB replay. The observed 40-million-transition projection is 1.698 TB decimal before checkpoints/backups and other overhead; D: had 791.6 GiB free at measurement. [Provision archival capacity and test restoration](../artifacts/phase21/reviews/backup_capacity_runbook.md) before starting final seeds. No external backup has been configured by this task.

Nine verified temporary C: ledger copies (159.022 MiB) are retained by your explicit decision. Their D: archives remain verified; no deletion was performed and cleanup approval is no longer pending. The [retention decision](../artifacts/phase21/hot_journal_retention_decision.json) records the nine unchanged copies. This decision does not change completed replay acceptance.
