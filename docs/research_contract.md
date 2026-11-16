# Research contract

**Current scope (2026-11-11):** the approved four-day **Relay fixed-data B/D
transfer pilot** supersedes the historical one-week schedule below. Use the
[`relay_four_day_v1` registration](../configs/studies/relay_four_day_v1.yaml)
and [detailed implementation plan](../detailed_research_implementation_plan.md).
Seeds 0/1 share each seed's frozen 50,000-transition replay, normalization and
selected 4,500-update model. B and D each receive 20 local PPO cycles; D adds
25 teacher cycles, 128 teacher trajectories and 100 distillation updates, then
fresh PPO optimizers and zero teacher KL. There is no online collection/model
refresh. The endpoint protocol prescribes 20 test episodes per final actor,
80 total, bank seed 5600, with descriptive paired success/return/latch outcomes;
the current fixed-data study makes no AUC or significance claim.

P4.4 implements D's integrated schedule and explicit completed-history adoption
without changing running P4.1-P4.3 sources. Adoption preserves development
chronology and does not establish a fresh comparative cohort. P5.2 implements
the reduced B controller and fresh-only D adapter. P5.3-P5.5 implement the P5
ledger, final evaluator, paired reader and a separate prospective cohort freeze.
Actual comparative training and testing remain operator-controlled. The P4.5 engineering
gate is implemented; its decision requires the operator's completed-run audit.
See the [integration guide](phase44_integration.md). No implementation check
establishes real cooperation, transfer benefit or measured GPU throughput.

P5.1 supplies a separate
[`relay_endpoint_protocol_v1` contract](../configs/studies/relay_endpoint_protocol_v1.yaml)
and [candidate registration workflow](phase51_protocol.md). It freezes the
scientific choices, exact test-bank definitions and existing input hashes on
CPU, with `execution_ready: false` and `cohort_frozen: false` until the
separate executable-cohort registration is published. The original candidate
receipt is preserved even after that new registration.
Historical D runs keep their original chronology; no registration can silently
promote adopted development work or certify prior nonuse of the test bank.

P5.2's [paired training tools](phase52_training.md) preserve the four prescribed
seed/variant slots, initial actor/critic pairing, frozen inputs, exact stage
budgets and checkpointed optimizer/RNG continuation. `training_tools_ready: true`
does not imply a frozen executable cohort. Explicit development candidates are
labelled ineligible. The [complete Phase 5 workflow](phase5_study.md) reruns all
static/training/endpoint CPU checks, verifies unchanged original source/input
bytes and binds the additive implementation in a new cohort. All four fresh
arms must be prepared before training; all four must finish before final test
execution. Test-bank nonuse is an explicit operator attestation, not an inferred
historical fact. Final evaluation uses the same Torch version and CPU/CUDA action
RNG backend for all four actors. Analysis requires all 80 authenticated native
records and makes no superiority claim. Fixtures cannot establish study results.

P5.6-P5.8's [server reporting extension](phase5_completion.md) preserves the
existing 296-file cohort source set. It adds saved final-cycle PPO and
distillation diagnostics, recorded resource work, two endpoint/compute figures
and a final execution/evidence gate. Unrecorded historical resource measurements
remain unknown; no fresh training or native episodes are used. Poor outcomes
remain eligible for execution completion, while successful transfer and
statistical superiority are not implied by a passed gate. All scientific
post-processing for this extension is run on the original Linux server CPU.

## Historical one-week amendment

The following 2026-11-09 registration is retained as historical context. It is
the earlier online schedule, not the current fixed-data training instructions.

**Historical amendment (2026-11-09):** use the separate `relay_week_v1` study
profile for a bounded Relay B-versus-D comparison. The original `base`, `pilot`
and `final` configurations remain historical registrations and preserve the
provenance of existing world-model checkpoints. The
[detailed implementation plan](../detailed_research_implementation_plan.md)
defines the remaining implementation and execution work.

The question is whether a **single privileged-teacher action-distillation
warm-start, followed by local MAPPO**, improves short-budget real-environment
learning compared with direct local MAPPO in the same Transformer world-model
framework. The distinctive investigation is transfer from a globally informed
imagined teacher into strictly local recurrent actors under paired real-data and
local-PPO budgets. Existing components are credited to prior work; the experiment
does not isolate their individual causal contributions or establish component
synergy.

| Item | Frozen scope for the reduced study |
| --- | --- |
| Task and arms | Relay only; B and D; seeds 0 and 1; four runs |
| Initial data and model | Existing 50,000 joint transitions and selected 4,500-update model per seed, shared by B/D |
| Total real budget | 100,000 per run: 50,000 initial plus two 25,000-transition local-policy blocks |
| Model refresh | 250 optimizer updates after each online block, identically for B and D |
| Local MAPPO | 40 initial cycles plus 20 per online block; 80 cycles per arm |
| PPO batch and rollout | Horizon 15, 256 starts, four epochs and four recurrent minibatches |
| D preparation | Teacher trained once for 50 cycles; 128 teacher trajectories; 100 distillation updates with 32 sequences per minibatch |
| D refinement | All 80 local PPO cycles use student actions and no teacher KL; no online teacher refresh |
| Primary curve | Success AUC on real-count grid 0, 50,000, 75,000, 100,000 |
| Evaluation | 20 fixed stochastic test episodes/checkpoint, ten per test layout; bank seed 5600 shared across runs and checkpoints |

Both arms have the same deployed actor capacity, local inputs, critic interface,
model architecture and refresh schedule, real budget, and local MAPPO schedule.
Pair the initial real replay, world-model weights and optimizer state,
normalization, and local actor/critic initialization within each seed. D's
warm-start changes only its local actor; its local critic starts from the paired
initialization. Distillation uses its own Adam optimizer (learning rate 3e-4,
epsilon 1e-5, betas 0.9/0.999, weight decay zero, gradient clip 0.5), then local
PPO starts with fresh optimizer state. Later each local policy collects its own experience and updates
its own world model. D adds teacher and supervised computation, which must be
reported separately from interaction efficiency.

The receipt-bound P2.4 selection remains
`artifacts/phase24_selection/shortened_v1/selection.json`. The
[2026-11-09 P2.5 screen](../artifacts/phase25_runs/shortened_20261109_100750/campaign/report.json)
records both Relay models as ready for an exploratory Phase 3 check, while Cover
still needs review. This supports choosing Relay for the reduced study; it is a
validation-informed scope change and must be disclosed. Retain Cover's existing
data, checkpoints and diagnostic failures as development evidence. Full P2.5
quality acceptance and native learned-policy acceptance have not been established.

Count initial scripted/random exploration and all later training collection.
Replay reuse counts once; a synchronized two-player joint transition counts once.
The four runs attribute 400,000 training interactions in total. Physical Relay
collection is 100,000 shared initial plus 200,000 new online interactions. Report
the earlier Cover collection and all other development work separately, as well
as simulator tics, teacher/distillation computation and evaluation interactions.
Evaluation data never enter replay or normalization updates.

The primary hypothesis remains positive paired D-minus-B normalized success AUC,
with denominator 100,000 and trapezoidal integration over the four declared
checkpoints. Report both seed-level differences, their mean and observed range.
With two training seeds, use descriptive estimates and report every seed; do not
report seed-bootstrap confidence intervals, significance or equivalence claims.
Twenty evaluation episodes provide a coarse estimate on a fixed two-layout bank,
not additional independent training replications. Final success, return, latch
progress, termination outcomes and per-layout results are secondary descriptions.
There is no additional 200-episode final evaluation.

Freeze one configuration, source identity, checkpoint selection, paired RNG
scheme and test-bank definition before the comparative cohort. Development uses
validation layouts; reserve test results for the frozen analysis. The eight
paired validation episodes per untrained/trained actor in P3.6 remain an
engineering check. The first reduced-profile export follows 40 local cycles
(`--max-units 41` only for a fresh run), without requiring online collection.
Seeds 0 and 1 and Relay were already used during development; this is an
exploratory study, not an untouched confirmatory seed set.

Phase 4's reduced D implementation and Phase 5's fixed test-bank evaluator and
analysis remain pending. If D is not verified by the end of day 3 of the one-week
work plan, or bounded native feasibility remains unresolved, prioritize an
implementation-and-feasibility report with the available B/model evidence. Do
not present that fallback as a B-versus-D result. Negative, diverged or missing
runs remain visible; do not choose seeds, checkpoints or budgets after observing
test performance. An incomplete four-run curve remains incomplete.

The 2026-11-08 amendment shortened the initial fit while retaining its original
5,000-update registration. This amendment additionally changes the later training
and evaluation budgets through a separately bound study profile. Existing
configuration, checkpoint and report bytes retain their original meaning; fresh
run manifests must record the amended profile, actual initial update count and
derived schedule. Numerical and interface invariants remain required throughout.
