# Phase 1 completion: P1.5-P1.7

The remaining environment work implements the target/event registry, formal
geometry splits, terminal and image-loss semantics, and the measured acceptance
campaign. The main output is
[`artifacts/phase1_complete/phase1_acceptance.json`](../artifacts/phase1_complete/phase1_acceptance.json).
Its stage reports preserve parameters, source/runtime fingerprints, hashes of
the evidence files, exact native traces, and process coordination journals.
The readable measured summary is
[`artifacts/phase1_complete/README.md`](../artifacts/phase1_complete/README.md).

The earlier P1.1-P1.4 evidence remains historical and unchanged. Matching sources
were copied to `artifacts/phase1_completion/before_p15_p17/` before this work.
P1.5-P1.7 do not begin model training or add learned comparison variants; the
research comparison remains B versus D.

## P1.5: authoritative supervision

See [the registry documentation](phase1_targets.md) and
[`configs/privileged_registry_v1.json`](../configs/privileged_registry_v1.json).
Fields declare source, units, encoding, normalization and validity. The registry
is versioned and hashed in scenario and trajectory manifests. Per-player
targets include position, sine/cosine heading, planar velocity, health, armor,
ammunition and alive state. Task and stable authored object targets cover the
plate, physical gates, latch, consoles, extraction and enemies. Missing or
inapplicable targets carry false validity masks and `None`, not invented zeros.

ACS monitors cumulative events on native tics. Python derives transition deltas
from episode-relative cumulative counters, so a sparse rendered endpoint cannot
erase an intermediate event. Native combat counters are baselined on reset.
The reserved ACS reward global 0 remains zero; the wrapper computes one shared
reward from authoritative milestone deltas and the published boundary, with
component checks and equal rewards for both players. USER positions use declared
signed 16.16 conversion; the native campaign checks fractional coordinates.

All supervision remains under `training_only_info`. Local policy calls still
contain only own RGB, executed previous action and own recurrent memory.

## P1.6: boundaries, images and geometry holdouts

Success, death, task failure and the declared deadline are genuine terminals
with zero value bootstrap. Death/failure takes precedence over success; success
at the deadline takes precedence over deadline failure. Collection cutoff is
an external truncation: valid final observations, the ordinary step cost,
bootstrap one, then a fresh episode and reset actor memories at the next block.

`StepResult.image_reconstruction_mask` and the corresponding validity metadata
expose a separate per-player image-target mask. A missing terminal image stays
`None` with mask zero; no repeated or black placeholder becomes a true target.
Final rewards, events and observations are captured before reset. Infrastructure
errors produce durable records and no reward-bearing transition.

See [the split documentation](phase1_splits.md),
[`configs/environment_splits.yaml`](../configs/environment_splits.yaml), and the
generated [split manifest](../artifacts/phase1_complete/split_manifest.json).
Each family has six training, two validation and two held-out test geometries.
Holdouts use corridor/arena lengths, gate positions and pillar configurations
absent from training. This is a holdout within the two authored task topologies,
not arbitrary-map, new-weapon or new-enemy generalization. Versioned spawn,
texture and enemy banks are shared across splits to isolate geometry scope.
Static checks cover every bank combination and fresh-spawn state; native
mechanics checks exercise every template at a declared seed.

`build_split_scenario(..., purpose='training')` refuses validation/test geometry.
Diagnostic provenance is always learning-ineligible. The acceptance campaign
uses `validation_interactions.jsonl`, outside training replay. Any future
diagnostic import must explicitly enter its original real interactions in the
training budget ledger and respect the split restrictions. Benchmark and replay
reports also retain their own consumed-count deltas; no diagnostic automatically
becomes an initial Phase 2 dataset.

## P1.7: replay and coordination fixes

The investigation and source references are documented in
[the replay notes](phase1_replay.md). Fixed-action replay isolated residual state
in multiplayer map resets and prediction-dependent pose differences. The
production reset now closes both participants and restarts a fresh pair in new
episode working directories while retaining cumulative accounting. Before each
native tic, both workers call `set_action`, rendezvous, and only then call
`advance_action(1)`. Both clients disable local prediction. Generated native INI
settings are retained for readback checks.

The coordinator captures each owned native engine's PID, creation time and exact
executable while its participant is healthy. Recovery also cleans those engines
when a Python participant has already crashed, rechecking process identity before
signalling so a reused PID is not targeted. Native engine-exit and hard worker-exit
fixtures verify cleanup and fresh recovery separately.

Replay compares exact player/task/target/event/reward traces and records RGB byte
hashes independently. Any RGB difference saves the actual first differing images
and pixel-difference measurements. Passing the pinned platform does not imply
cross-platform or all-seed determinism. The legacy native-map reset path remains
available only for explicit investigation; acceptance uses fresh-pair reset.

The complete acceptance gate runs:

1. The full unit suite and versioned schema/split consistency checks.
2. Native scripted success on all 20 templates, each-player-inactive checks for
   both families, and full mechanics videos from real local frames.
3. A native plate/gate event round trip with equal inactive endpoint flags but
   nonzero cumulative entry/exit/open/close differences.
4. Distinct native success, death, deadline, collection-cutoff and participant
   failure paths, including recovery without fabricated rewards or extra retries.
5. Fixed joint-action replay for both families, three repetitions each under
   fresh environment construction and coordinated reset.
6. At least 10,000 published joint decisions and 100 coordinated resets,
   checking native tics, response identities, events, shared rewards, image
   masks, cross-episode state and the durable dispatch/ACK journal.
7. Holding each own visual/action/memory history fixed while replacing global
   diagnostic metadata and the teammate image. The Phase 1 probe responds to its
   own pixels, action and memory; repeat this generic audit with actual trained
   B/D policies in Phase 3. This is adapter isolation, not a trained-policy result.
8. One- and two-environment benchmarks for both families, three alternating
   repetitions, p50/p95 latency, transition/tic rates and process-tree CPU/RAM.

The job and child processes are restricted to four logical CPUs; original parent
affinity is restored on exit. Benchmarks retain production RGB, target extraction
and durable logging. Startup, warmup and teardown are excluded from decision
throughput and reported separately. The scheduler submits all environments'
actions before waiting and waits for the batch before its next submission.
RSS is the sampled sum over the parent, participant workers and native engines;
shared pages may be counted in several processes. Windows private memory and
CPU core-seconds are also reported. No GPU inference or learner overlap is
included, so rebenchmark on the training host and under the eventual learner load.

The validation stress runner collects two independent environments concurrently,
one per family, with 5,000 published decisions per family in the full campaign.
Its block ends are explicit collection cutoffs, not task failures. Reset warmup
decisions and intentional failed dispatches are counted separately from the
10,000 valid-transition threshold. The global diagnostic ledger and the separate
benchmark/replay deltas reconcile into one whole-campaign accounting record.

All benchmark repetitions use the same layout seed and engine seed 7000, with
physical workload trace hashes retained across counts/repetitions. A first
variable-seed preflight showed large rate variation and was rejected; it remains
under `artifacts/phase1_completion/benchmark_preflight_v1/`. The later controlled
preflight selected two collectors, and `resources.initial_collectors` was updated
consistently for both variants before the final campaign. The final benchmark
must agree with that configured choice. The cause of the earlier rate change
is unassigned; achieved stability describes the measured window, not a guarantee
about the host at all times. The source investigation deliberately avoided
unsupported FPS-flag changes.

The predeclared collector rule uses two only when both families gain at least
10% median throughput, two-environment throughput CV is at most 25%, and p95
latency is at most three times single-environment latency. The single-environment
baseline must itself have CV at most 25%; otherwise no stable selection is
published. The measured selection is common to B and D.

## Commands

Run from the project root. Native ViZDoom needs Windows message queues and local
UDP access; in a restricted execution environment use the authorized unrestricted
run. On Linux substitute `.venv/bin/python` and provide the compatible ACC tool.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m scripts.phase1_acceptance verify --output-dir artifacts/phase1_complete
```

Individual measured stages can be run in a new directory with `prepare`,
`mechanics`, `events`, `boundaries`, `replay`, `stress` or `benchmark`. Replay
requires a current mechanics report in the same directory. `--template-limit`,
reduced stress counts and reduced benchmark lengths support development probes;
they cannot produce a complete Phase 1 acceptance result. `verify` rejects
reduced template/count requirements. `--reuse` accepts a completed stage only
when its source/runtime/parameter fingerprint and every referenced evidence
file hash still match. Use a new output directory after implementation changes
and preserve completed reports.

The `psutil` and `imageio-ffmpeg` additions are pinned. MP4 exports are lossy review
artifacts with side-by-side genuine local frames and an external caption; exact
128x96 PNGs and their RGB hashes remain authoritative. Encoder version and binary
hash are recorded in the runtime manifest. The simulator lock still does not
validate CUDA/PyTorch or the promised four-H200 training allocation.
