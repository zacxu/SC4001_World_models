# Interface, timing, and deployment contract

`world_models.contracts` implements dependency-free boundary types. A local call
contains exactly `local_rgb`, `previous_action`, and `local_hidden`. RGB is an
immutable uint8-equivalent byte buffer in `[3,96,128]` order. Previous action is
one of 12 categorical primitives, or the distinct START symbol at reset. Hidden
memory contains finite numbers and is validated against the configured size.
Each agent owns its own memory even when actor weights are shared.

The strict mapping constructor rejects extra fields such as global state,
teammate features, reward, role identity, and privileged variables. This enforces
payload shape, not a proof of pixel provenance or neural information locality.
The later simulator/policy implementations must also pass the plan's locality
and gradient-isolation tests.

Teacher recurrent memory is local; current shared tokens can enter its action
head. Local actors/students never receive those tokens. Privileged engine fields
are auxiliary targets only. The world posterior can use both synchronized views;
the autonomous prior uses current shared state/actions and a known episode clock,
without future observations or true future engine state.

At t, both actions lead to reward/termination and observations at t+1. State
targets align with observations; event deltas align with transitions. Metadata
records source/next decision index, start/end tics, actual advancement, and valid
next-image masks. A missing terminal image must not become a fabricated target.

`EpisodeClock` advances once per joint decision. Deadline, success, and death are
task terminals; death takes precedence, then success, then deadline. A collection
cutoff is a valid truncation. Infrastructure failures are separate errors.
`EpisodeBoundary` exposes the bootstrap/recursion distinction: true terminals
bootstrap zero; nonterminal cutoffs use a next value but stop recursion at the
stored sequence boundary. `RewardRule` handles step cost, bounded first-time
milestones, terminal reward, and milestone-target decomposition.

RNG namespaces separate task families, branches after initialization, components,
and the two agents. Only declared initial components share B/D seeds. Evaluation
uses its own frozen paired bank. Stable hash derivation avoids Python's randomized
`hash()`; manifest checks detect accidental uint32 seed collisions.

P3.1 implements this local boundary with `world_models.policies.LocalActor.step`.
Its tensor forward accepts only uint8 local RGB, the executed previous action,
and external local memory. START zeros that row's memory before reading its first
image. B and D students use the same architecture and paired initial weights;
within each arm, the two players share actor weights and keep separate memories.
The centralized critic accepts shared world tokens, both detached
actor memories and normalized remaining time; its gradients cannot reach either
actor or world model. See [the P3.1 API](phase31_policy.md) for tensor shapes,
sequence masks, image quantization and paired initialization.

P3.2's `AnchorPrefix` exposes only real RGB through observation `t`, executed
actions strictly before `t`, recorded decision clocks and replay identities.
`AnchorBuilder` reconstructs each local memory from reset without gradients;
its output logits and memory already include the anchor observation. The first
imagined action uses those logits directly. Cache identities include the exact
world-model and actor versions. See [the anchor API](phase32_anchors.md).

P3.3 preserves this boundary in `world_models.rollouts.RolloutGenerator`.
The first action uses the anchor's existing logits. Later actors consume only
their own decoded uint8-equivalent image and executed action. Shared dynamics
advance once per joint decision; buffers preserve T+1 states and T transitions,
with separate validity, true terminal and horizon-cutoff masks. True terminals
bootstrap zero; alive horizon cutoffs keep a final team value. Synthetic stores
cannot be loaded as real replay. See [the rollout API](phase33_rollouts.md).

P3.4's `world_models.mappo.RecurrentMAPPO` rebuilds current actor memory from
the complete own-player real prefix before each whole-sequence minibatch. Prefix
reconstruction ends before the anchor observation; gradients cover the anchor and
imagined source sequence. Behavior likelihoods, old values and stored critic
memories stay fixed for all epochs. One team GAE supplies two individual PPO
ratios; raw returns train the critic once per joint sample. Only actor and critic
weights change. Local actor computation uses IEEE FP32 for real prefixes,
imagination and PPO replay, including inside a BF16 world-model/critic context.
The pre-update actor log-probability check retains a `2e-5` absolute tolerance.
See [the MAPPO API](phase34_mappo.md).

P3.5's `world_models.cycle` controller keeps real online data separate from
imagined PPO buffers. The local actor collects temperature-one stochastic actions
with independent player streams; each real block ends at its exact joint-transition
budget and starts fresh at reset. Online replay carries `training_variant` as its
authoritative branch owner (the frozen P2 storage codec retains legacy paired
fields). Initial fitting-only normalization remains frozen; model refresh extends
the ordered real fitting cohort. Receipt-bound checkpoints retain all model,
policy, optimizer, RNG and sampler state. Partial real collection is counted and
cannot be silently recollected. See [the P3.5 training guide](phase35_training.md).

The shortened initial-fit pilot uses one immutable P2.4 selection manifest for
P2.5 and subsequent B/D initialization. Each case records its actual saved update
count and original target; selecting a checkpoint never rewrites its training
metadata. P3.5 accepts an unfinished initial fit only through this explicit
selection, binds its receipt into the run identity, and counts later model
updates from that selected count. Bare checkpoint preparation retains the
original completed-schedule requirement.

P3.6's evaluator loads only receipt-verified local actor exports. Each real
validation episode resets both local memories and previous-action START symbols;
independent action generators use a fixed bank shared by baseline and trained
actors. Episode results, traces and physical interaction debits remain outside
learning replay. The acceptance report verifies source/config identities,
completed training counters and paired native behavioral evidence; synthetic
mechanics and unit tests alone cannot grant real-policy acceptance. See
[the P3.6 acceptance guide](phase36_acceptance.md).

The historical `relay_week_v1` amendment binds a separate study profile into fresh run
identities while preserving the original P2.4 configuration and checkpoint
identities. Its Relay schedule is 50,000 shared initial interactions, two online
blocks of 25,000, 250 model updates per online block and 40/20/20 local PPO
cycles. Resume must validate the same profile and derived schedule; the original
pilot/final manifests retain their original budgets. Actors keep the same local
observation, action-history, memory and deployment interfaces.

The active four-day plan uses a one-time teacher warm-start. **P4.1 is implemented**
as a standalone 25-cycle teacher stage on each seed's frozen selected initial
model. `GlobalTeacher.memory_step` accepts only own RGB, previous action and
local hidden state. `action_from_memory` reads that memory, current shared tokens
and player ID through four-head, width-256 attention. Shared tokens are detached
and never enter recurrent memory. The teacher has no local deployment `step`.
Teacher anchors, rollouts and optimizer checkpoints have distinct schemas;
local PPO rejects teacher behavior buffers. Teacher and local forwards use IEEE
FP32 even under outer world-model autocast. Teacher collection drives its own
executed joint actions; the critic consumes teacher memories.

`scripts.phase41` provides explicit prepare/run/status/stop and CPU verification.
Preparation never optimizes; the runner commits a receipt-bound checkpoint after
every completed cycle and freezes the finished teacher for later consumption.
No student or world-model optimizer is created by this stage. See the
[P4.1 guide](phase41_teacher.md).

**P4.2 and P4.3 are implemented as separate stages.** The completed teacher's
128 trajectories supply targets for 100 supervised local-actor updates with 32
full sequences. Distillation reconstructs student histories with executed
teacher actions; teacher, model and student critic remain frozen. No teacher
memory or shared tokens enter the student. See the
[P4.2 guide](phase42_distillation.md).

P4.3 verifies the completed student handoff and saved initial/final state, loads
the distilled actor and original paired critic, then creates fresh local PPO
optimizers. All 20 local cycles use current student-generated rollouts and the
ordinary B MAPPO objective, with zero teacher queries and zero teacher KL.
World model, replay and normalization remain frozen. Every cycle commits PPO,
sampling and controller state; the final actor-only export retains the existing
strict loading interface. See the [P4.3 guide](phase43_refinement.md).
P4.4 now connects these stages through the separate `world_models_campaign`
package. It preserves historical source identities, saves paired initial values
before fresh teacher work, reconciles child commits after interruption and
publishes one checkpoint-bound local export and accounting record. Completed
standalone histories may be explicitly adopted without optimization, with their
original chronology and development status intact. See the
[P4.4 guide](phase44_integration.md). The [P4.5 acceptance gate](phase45_acceptance.md)
now binds explicit invariant checks to a read-only completed-run audit and an
actor-only deployment subprocess. It preserves all upstream source/receipt bytes
and publishes a separate engineering decision. A pass is not real-policy,
transfer or cohort acceptance. Trained-run acceptance awaits the operator's audit;
no online training collection/model refresh occurs in this D schedule.

P3.6 remains a validation-only paired B integration gate, with eight episodes
per actor; the prior development profile first exports after 40 local PPO cycles.
Its native behavioral criterion is unchanged. Phase 5 uses its separately
implemented endpoint evaluator and analysis for the fixed 20-episode bank (seed 5600),
balanced over the two test templates and used once per completed local actor.
That evaluator retains actor-only loading, independent player RNGs,
immutable exports, separate evaluation accounting and test-data exclusion from
learning. P3.6's validation evaluator does not itself provide this test protocol.

P5.1's separate `world_models_study` package defines that final bank statically:
20 records on the existing two Relay test templates, seed 5600, with scenario
seeds shared by all four actors and action seeds including master seed, episode
and player while excluding B/D branch. It binds source/config/task/input hashes
without opening environments or performing learning. Receipt-bound candidate
registration is distinct from prospective executable cohort membership. The
native endpoint evaluator and paired reader are provided separately below; see the
[P5.1 guide](phase51_protocol.md). P5.2's separate `world_models_paired` package
adds the fixed-data B runner, fresh D adapter and four immutable candidate slots.
Initial pairing covers actor, critic, selected model, replay and normalization
hashes; each branch owns its optimizer and sampling RNG state. Its final actor
uses the existing actor-only interface. Candidate training tools do not establish
comparative eligibility or endpoint evidence; see the [P5.2 guide](phase52_training.md).

`world_models_endpoint` adds a separate source-bound prospective registration,
durable metadata audits and cumulative accounting (`scripts.phase5/phase53`).
All four prepared slots are paired before training. Before native evaluation,
`final_actor` verifies all four completed audits, original immutable inputs and
actor-only final export receipts. It performs no model/controller import.
`scripts.phase54` loads only the local actor in FP32, executes the explicit test
geometry with independent player streams, and seals episode/trace/accounting
receipts. One actor has one claimed final-test output; resume retains its first
valid episode outcomes. `scripts.phase55` authenticates exactly four native
reports, the shared bank and sampler/backend/Torch version, then publishes
descriptive tables. Incomplete evidence has no imputed metrics. Historical D
export metadata remains unchanged; the new evaluation manifest owns the test
split declaration. See the [operator guide](phase5_study.md).

The root-level `study_reporting` extension supplies P5.6-P5.8 without adding
files to that cohort's frozen `src/scripts/tests` source set. It authenticates
the old evidence through unchanged readers and binds its own Python sources,
reporting dependency pin and complete synthetic test log separately. `complete`
and `inspect` require the original Linux server and saved paths. They create no
learner, optimizer or native environment. Checkpoint diagnostics are read on
CPU, all figures bind their saved numeric data, and missing measurements remain
null. Acceptance checks four completed arms and 80 native outcomes plus retry,
input, source and publication integrity; it does not require D to outperform B.
See the [completion guide](phase5_completion.md).

`study_release` supplies P6.1-P6.2 outside both frozen source namespaces. Its
server command consumes the unchanged P5 acceptance reader and native trace
bytes, not a model or optimizer. Trace claims retain line/hash references and
leave unsupported causal classifications undetermined. A strict file inventory
and portable input mapping preserve actor exports, all final outcomes, tables
and plots in a release directory. Saved-record reproduction imports only pure
aggregation/plotting helpers from the sealed sources; no original absolute paths,
Torch weight deserialization or simulator execution are required. Use `python -B`
to avoid modifying the sealed inventory. Partial stage completion never accepts
P6.3 or establishes successful transfer. See the [release guide](phase6_release.md).
