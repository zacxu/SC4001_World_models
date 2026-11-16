# Shared imagination for decentralized cooperation

**Completed reduced study (14 November 2026):** [final paper](paper/research_paper_draft.md),
[claim audit](paper/claim_evidence.md), and [local P6.3 verification](docs/phase63_paper.md).
Four matched Relay actors completed evaluation; all had 0/20 successes. Final
study/report acceptance does not establish successful teacher transfer.

The project compares direct local MAPPO learning in a shared Transformer world
model (**B**) with teacher learning, action distillation, and local refinement in
the same framework (**D**). Both deploy two actors using only their own visual
observations, previous actions, and separate recurrent memories.

Start with the [approved proposal](research_implementation_proposal.md),
[detailed plan](detailed_research_implementation_plan.md),
[research contract](docs/research_contract.md), and [runbook](docs/runbook.md).

**Active four-day plan (2026-11-11):** Relay, seeds 0/1, B/D (four fresh runs),
frozen 50k initial data and 4,500-update models, 20 local PPO cycles per arm.
D adds 25 teacher cycles, 128 distillation trajectories and 100 supervised
updates. Final evaluation is 20 test episodes per actor. No online training
collection or model refresh is planned. P4.1's teacher, P4.2's distillation and
P4.3's unguided local refinement and P4.4's integrated D controller are implemented;
P5.2 supplies the matched reduced B controller and fresh D adapter. P5.3-P5.5
now supply accounting, final testing, paired analysis and a separate prospective
cohort freeze. Start with the [Phase 5 guide](docs/phase5_study.md); research
training and final test execution remain operator-controlled.
The old `relay_week_v1` configuration retains its historical online schedule.

Both latest Relay P2.5 screens are ready for exploratory Phase 3; the complete
four-model campaign still needs review because of Cover. Preserve and finish
the existing [B development check](docs/phase36_acceptance.md) in its original
checkout before synchronizing new source. The [P4.1 guide](docs/phase41_teacher.md)
provides verification and operator-controlled teacher training. Freeze all
cohort sources before preparing comparative runs. See the
[section-by-section reduction and four-day allocation](detailed_research_implementation_plan.md).

Phase 0 implements configuration, contracts, seed derivation, inventory, and
verification; its record is [artifacts/phase0_acceptance.json](artifacts/phase0_acceptance.json).
Phase 1's complete environment interface and acceptance campaign are documented
in the [P1.5-P1.7 runbook](docs/phase1_complete.md), with a versioned
[target registry](configs/privileged_registry_v1.json),
[geometry splits](configs/environment_splits.yaml), and measured
[phase1_acceptance.json](artifacts/phase1_complete/phase1_acceptance.json).
The [readable results](artifacts/phase1_complete/README.md) include replay,
10,000-decision/100-reset checks, native mechanics videos and collector selection.

Phase 1's first step implements P1.1–P1.4: the live two-player coordinator, local
visual/action adapters, relay, and cover-and-activate mechanics. See the
[implementation and commands](docs/phase1_steps1_4.md) and its generated
[verification report](artifacts/phase1_requested_fixes/acceptance.json).
The revised mechanics include visible extraction cues, unrestricted console
roles, moving persistent enemies, paired fire/no-fire diagnostics, durable
wrapper recovery, and explicit IWAD/port handling. Earlier evidence is archived
under `artifacts/phase1_steps1_4/before_requested_fixes/`.

## Set up on Windows PowerShell

From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase0.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m scripts.phase0 check --output artifacts/phase0_acceptance.json
```

## Set up on Linux

For the current GPU server at `/home/zixuan/projects/WM1`, follow the
[WinSCP transfer and server setup guide](linux_gpu_server_setup.md), then
[P2.5 validation](docs/phase25_validation.md). Choose available GPUs with
`CUDA_VISIBLE_DEVICES` and use tmux for persistent execution. The
[ASPIRE2A guide](linux_gpu_server_setup_through_p23.md) is retained for the previous
cluster. The commands below provide the generic Phase 0 setup.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase0.lock
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python -m scripts.phase0 check --output artifacts/phase0_acceptance.json
```

For the implemented Phase 1 environment, use the tested Python 3.12 stack.
After creating the virtual environment, install its complete dependency lock and
scenario compiler:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m scripts.bootstrap_acc
```

See [the artifact policy and fresh-clone checks](artifacts/README.md) for which
reports are versioned and how to regenerate the excluded raw outputs.

**P2.1 is completed for the approved pilot scope:** relay and cover, seeds 0 and 1,
with four shared 50,000-transition stores. See the [measured results and acceptance](artifacts/phase21/README.md)
and [real replay runbook](docs/phase21_replay.md). World-model and policy training
remain later steps. Raw data stays outside Git; preserve a separate verified backup.

**P2.2 is completed:** the shared stochastic Transformer architecture, visual
encoder/decoder, supervised heads, and prior-only imagination interface are
implemented and verified. See the [architecture and API](docs/phase22_architecture.md)
and [verification report](artifacts/phase22/phase22_acceptance.json).
These checks used untrained models. Trained predictive quality is assessed
after the server's P2.4 fit.

**P2.3 is completed:** the reduced model objective, fitting-only normalization,
and loss/gradient/event diagnostics are implemented and verified. See the
[objectives and usage](docs/phase23_objectives.md) and
[verification report](artifacts/phase23/phase23_acceptance.json).
The four frozen pilot normalization files are shared by B/D within each
task/seed pair.

**P2.4 implementation is completed; initial fitting is shortened for the pilot.** Recurrent
sampling/burn-in, exact microbatch accumulation, AdamW fitting, checkpoint/resume,
held-out diagnostics, validation banks and branch refresh are implemented.
On 2026-11-08 the user approved using the latest committed checkpoints instead
of waiting for the original 5,000-update target. Freeze all four selections with
`scripts.phase24_freeze`; preserve actual counts and use identical weights for
B/D within each task/seed. See the [selection runbook](docs/phase24_training.md)
and [original implementation evidence](artifacts/phase24/README.md). The earlier
two-GPU training script now refuses to start by default.

**P2.5 uses a compact pilot engineering gate:** screen the four shared initial
models from the frozen P2.4 selection once each, with up to 64 windows per
model. Prior-only rollouts provide 1/5/10/15-step diagnostics and simple baseline
comparisons. See the [compact validation guide](docs/phase25_validation.md).
The campaign's `pilot_model_selection.json` records four explicit checkpoint
identities, actual update counts and provisional readiness, and binds the same
P2.4 selection used by Phase 3. Counts may differ across cases in this shortened
pilot; within-case B/D pairing is unchanged.
Readiness for a small Phase 3 pilot is separate from formal P2.5 acceptance;
training-geometry diagnostic episodes do not establish held-out geometry or
final-test performance. No trained-model acceptance result is claimed here.

**P3.1 is implemented:** the shared local CNN/GRU actor and centralized team critic,
paired B/D initialization, raw Huber value loss and strict local deployment API.
See [the policy interfaces and CPU checks](docs/phase31_policy.md) and
[implementation evidence](artifacts/phase31/implementation_v1/implementation_checks.json).
Research policy training and trained-policy acceptance remain pending.

**P3.2 is implemented:** fitting-replay imagination anchors, full-prefix world
posterior and local actor memory reconstruction, and automatic cache invalidation
after parameter changes. See [the anchor API and CPU checks](docs/phase32_anchors.md)
and [implementation evidence](artifacts/phase32/implementation_v1/implementation_checks.json).

**P3.3 is implemented:** frozen on-policy imagined trajectories with independent
action streams, exact stopping/bootstrap masks and separate receipt-bound
synthetic buffers. See [the rollout API and local checks](docs/phase33_rollouts.md)
and [implementation evidence](artifacts/phase33/implementation_v1/implementation_checks.json).

**P3.4 is implemented:** recurrent local MAPPO with full current-policy prefix
replay, raw GAE/critic returns, individual player PPO ratios and complete fixed
optimizer schedules. See [the update API and CPU checks](docs/phase34_mappo.md)
and [implementation evidence](artifacts/phase34/implementation_v1/implementation_checks.json).

**P3.5 is implemented:** variant B's real collection, model refresh and imagined
PPO controller, with exact budgets, completed-block checkpoints, actor exports
and safe stop/resume. See [the GPU server training guide](docs/phase35_training.md)
and [CPU implementation evidence](artifacts/phase35/implementation_v1/implementation_checks.json).
Development training has run; P3.6 acceptance and the complete D treatment remain unresolved.

**P3.6 is implemented:** invariant verification, a small synthetic PPO learning
check, local-actor evaluation in native ViZDoom and an evidence-based acceptance
report. See [the acceptance guide](docs/phase36_acceptance.md). CPU engineering
evidence is recorded locally; actual policy acceptance requires paired untrained/
trained actor evaluations from the server. The synchronized initial development
check is `needs_review`; B-versus-D efficacy has not been measured.

**P4.1 is implemented:** the globally informed teacher, full-prefix teacher
imagination and recurrent PPO, with strict local memory and frozen world-model
weights. Its standalone runner checkpoints every cycle and supports safe
stop/resume. [All 249 CPU implementation checks passed](artifacts/phase41/implementation_v1/implementation_checks.json).
See [teacher verification and Linux training commands](docs/phase41_teacher.md).
Research teacher training is left to the operator.

**P4.2 is implemented:** a receipt-verified completed teacher supplies 128 fixed
trajectories for 100 student-only forward-KL updates. The paired student critic,
teacher and world model stay frozen. The separate runner checkpoints every
trajectory/update and records before/after training-buffer diagnostics. See
[distillation verification and Linux commands](docs/phase42_distillation.md).
The [CPU verification report](artifacts/phase42/implementation_v1/implementation_checks.json)
records 292 passing tests with zero failures, errors or skips at its original
source snapshot. The [device-handoff correction](artifacts/phase42/device_handoff_fix_v1/implementation_checks.json)
adds CUDA-origin identity and checkpoint-based initial-weight checks; the
[saved-teacher audit](artifacts/phase42/device_handoff_fix_v1/saved_teacher_audit.json)
verifies the completed Relay seed-0 teacher's original hashes. All 32 focused
regression checks passed; GPU execution itself remains operator-controlled.
Keep the active P4.1 server checkout unchanged until its final handoff is saved.
P4.5 now supplies the separate engineering gate; no research distillation was
run by the CPU implementation checks.

**P4.3 is implemented:** the completed P4.2 actor and original paired critic
start 20 ordinary local PPO cycles with fresh optimizers, producing 320 actor
and 320 critic optimizer steps. Every rollout uses the current local student;
teacher queries and teacher KL stay zero, with data/model/normalization frozen.
Each cycle commits optimizer, sampling and counter state for safe stop/resume.
Completion publishes a checkpoint-bound student handoff and actor-only export.
See [verification, training and resume commands](docs/phase43_refinement.md) and
the [CPU implementation checks](artifacts/phase43/implementation_v1/implementation_checks.json).
All **225 tests passed**, with zero failures, errors or skips.
Research refinement and native evaluation remain operator-controlled. This
standalone stage does not complete the integrated D controller or acceptance gate.

**P4.4 is implemented:** the [fixed-data registration](configs/studies/relay_four_day_v1.yaml)
and separate `world_models_campaign` controller connect all four D stages,
preserving the existing P4.1-P4.3 source hashes. Fresh runs save paired local
initial values before teacher training. Resume reconciles committed child work
before training again; completion publishes accounting and a local-only actor.
Your current P4.3 run can finish unchanged and then be registered with `adopt`
without repeating optimization, explicitly marked as development history.
See [Linux commands and completed-run registration](docs/phase44_integration.md)
and the [CPU implementation checks](artifacts/phase44/implementation_v1/implementation_checks.json).
All **87 tests passed**, with zero failures, errors or skips.
Training is operator-controlled; trained-run acceptance remains a separate audit.
P5.2 supplies reduced B execution tools and P5.3-P5.5 provide the endpoint pipeline.

**P4.5 is implemented:** [CPU verification and completed-run acceptance](docs/phase45_acceptance.md)
combine eight named invariant groups, a tiny complete D smoke fixture, a read-only
checkpoint/optimizer/RNG/input audit and isolated actor-only deployment. The
new acceptance package preserves the running P4.4 source identity. Reports
include training-buffer KL/entropy, teacher/local progression and accounting.
See the [implementation checks](artifacts/phase45/implementation_v1/implementation_checks.json).
All **82 tests passed**, with zero failures, errors or skips; all eight named
requirements passed and verification sources remained unchanged.
Run `verify` while training continues, then `accept` after P4.4 completes.
Engineering acceptance does not establish transfer benefit or cohort eligibility;
the separate Phase 5 freeze and real endpoint results remain required.

**P5.1 is implemented:** [protocol registration guide](docs/phase51_protocol.md)
provides static CPU `plan`, `verify`, `register` and `inspect` commands. The new
[`relay_endpoint_protocol_v1`](configs/studies/relay_endpoint_protocol_v1.yaml)
binds the unchanged P4.4 amendment, paired seeds, fixed inputs, exact stage/compute
budgets, 20 final test episodes, descriptive analysis and retry policies.
Receipt-bound registration freezes these choices with `candidate_only: true`;
`execution_ready` and `cohort_frozen` retain their original false values. The
additive Phase 5 freeze creates a distinct executable cohort. Existing D runs
retain their historical provenance. No training or test episodes are launched.
See the [static implementation checks](artifacts/phase51/implementation_v1/implementation_checks.json).
All **44 CPU checks passed**, with zero failures, errors or skips and unchanged
verification sources. Training and research evaluation remain operator-controlled.

**P5.2 is implemented:** [paired training guide](docs/phase52_training.md)
provides matched fresh B initialization, 20 local PPO cycles, a fresh-only D
adapter, four receipt-bound candidate slots, durable progress and safe stop/resume.
The separate package preserves historical source hashes and registered profiles.
Use the new Phase 5 verification and prospective freeze before comparative
training. Explicit development candidates retain their own chronology and
eligibility limits. Training and real test evaluation remain operator-controlled.
All **52 CPU checks passed**, with zero failures/errors/skips and unchanged
sources; see the [implementation evidence](artifacts/phase52/implementation_v1/implementation_checks.json).

**P5.3-P5.5 are implemented:** the [complete operator guide](docs/phase5_study.md)
connects source-bound verification, a fresh four-run cohort, safe preparation
and continuation, cumulative physical/logical ledgers, one 20-episode test bank
per final actor, and receipt-bound descriptive paired tables. The evaluator
loads only the local actor and refuses intermediate or incomplete cohorts;
analysis rejects missing, duplicated, mismatched or fixture evidence. Existing
P5.1/P5.2 registrations remain unchanged parents. Implementation checks use CPU
fixtures; they establish neither research completion nor transfer benefit.
See the [combined evidence](artifacts/phase535455/implementation_v1/implementation_checks.json).
All **186 CPU checks passed**, with zero failures/errors/skips, including the
complete four-arm synthetic pipeline; all 279 original P5.2 source files are
unchanged. No research training or native test evaluation was executed.

**P5.6-P5.8 are implemented:** the [server completion guide](docs/phase5_completion.md)
summarizes saved secondary diagnostics and actual resource accounting, renders
two endpoint/compute figures and authenticates final execution completeness.
The root-level `study_reporting` package preserves the existing cohort's exact
296-file source freeze and adds its own independent source-bound checks.
Run its completion commands on the original Linux server CPU after the final
episodes; no further training or native evaluation is started. Missing resource
measurements remain explicit unknowns, and zero/negative outcomes are valid.
All **121 synthetic CPU checks passed**, with zero failures/errors/skips;
see the [reporting implementation evidence](artifacts/phase565758/implementation_v1/implementation_checks.json).
Actual scientific reporting and figure rendering remain server-side steps.

**P6.1-P6.2 are implemented:** the [server diagnosis/release guide](docs/phase6_release.md)
reviews a fixed sample of saved native traces and packages sources, local actors,
records, ledgers, tables and figures. Large binary archives remain indexed on the
original server. A relocatable CPU command recreates all six tables and two
figures from the sealed release, with no simulator, Torch weight loading or
training. The independent `study_release` package preserves both earlier source
freezes. Scientific diagnosis/packaging remain operator-run; P6.3's manuscript
and full Phase 6 acceptance remain outstanding.
All **93 synthetic CPU checks passed**, with zero failures/errors/skips;
see the [implementation proof](artifacts/phase6162/implementation_v1/implementation_checks.json).
The earlier 296 scientific and 17 reporting source files remain unchanged.

The shortened-checkpoint workflow has [historical CPU verification and a saved-checkpoint
audit](artifacts/shortened_pilot_checks/README.md). The latest server P2.5 screen
is synchronized; native learned-policy acceptance remains pending. Changes to
the one-week policy profile require current implementation checks before execution.
The amended runtime has [current CPU verification and read-only checkpoint/budget
evidence](artifacts/relay_week_checks/README.md).

Preserve `configs/base.yaml` and the original task/variant/experiment files:
saved P2.4 checkpoints bind their exact contents. The prior online development budget
is the separate [study amendment](configs/studies/relay_week_v1.yaml), selected
with `--profile pilot --study-profile relay_week_v1 --initial-selection ...`.
It validates the original checkpoint configuration, records amended budgets,
and rejects changed identities on resume. Original pilot/final schedules remain
available for historical runs; do not convert an existing run to the new schedule.
P4.1 uses its own fixed teacher-stage registration. The four-day profile now has
an integrated D launcher and P5.2 adds the matched B launcher. The additive
Phase 5 entry points provide the final endpoint pipeline and prospective freeze.

The importable code uses the standard `src/world_models/` package layout; the
submodules follow the detailed plan's environment/data/model/policy/training/
evaluation/infrastructure separation. The tests use Python's built-in unittest.
