# Detailed research implementation plan

**Project:** Shared Imagination and Policy Distillation for Decentralized Visual Cooperation  
**Basis:** Approved [research implementation proposal](research_implementation_proposal.md) and [conversation transcript](conversation_transcript.md).  
**Scope:** Implement and evaluate variant B and variant D only, following the proposal's six phases.

This is the implementation specification for the approved research direction. Numerical architecture, training, and experiment settings below are concrete **pilot defaults**, not measured results or claims of optimality. Resolve feasibility issues during pilots, record any changes, and freeze the final configuration before confirmatory experiments. Information-access rules, accounting rules, and the B-versus-D comparison apply throughout.

## 0. Research contract and project-wide conventions

### 0.1 Research question, hypothesis, and claims

**Question:** Does globally informed teacher training followed by action-distribution distillation and local MAPPO refinement improve the real-environment learning efficiency of two decentralized visual agents, compared with direct local MAPPO learning in the same shared-world-model framework?

**Primary hypothesis:** At a matched total real-interaction budget, D has higher normalized area under the real-environment team success-rate learning curve than B.

The treatment is the complete teacher-guided training procedure. The study can establish whether that procedure helps within this world-model framework and task distribution. It cannot isolate the effects of the Transformer, shared tokens, imagination, or the distillation loss individually. The intended contribution is the integrated system and the controlled empirical result.

| Property | B: direct local learning | D: teacher-guided learning |
|---|---|---|
| Shared Transformer world-model architecture | Same | Same |
| Initial real dataset and model checkpoint | Paired by seed | Paired by seed |
| Local recurrent actor architecture and initialization | Paired by seed | Paired by seed, before distillation |
| Local imagined MAPPO opportunities | Matched | Matched |
| Global teacher | Absent | Imagined MAPPO teacher |
| Action distillation | Absent | Warm-start and decaying guidance |
| Real collection after initialization | B's local actors | D's local students |
| Deployment | Local actor pair | Local student pair |

Use separate training runs for the two task families. Within each family, train on its declared mixture of training layouts. A single multi-task policy across both families is outside the initial experiment.

### 0.2 Information-access contract

| Component | Permitted inputs |
|---|---|
| Local B actor / D student | Its RGB observation, its own previous action, and its own recurrent memory. |
| Teacher's recurrent memory | Its local observation/action history. |
| Teacher action head | Its local memory plus current shared world tokens through cross-attention. |
| World-model posterior | Both synchronized visual histories, executed joint actions, and the episode clock. |
| Autonomous world-model prior | Current shared latent state, current joint action, and the deterministically advanced episode clock. |
| Centralized critic | Current shared latent state, both relevant actor memories, and normalized remaining task time. |
| Auxiliary supervision / diagnostics | Declared engine variables and task events from real replay. |

Start with privileged variables as **auxiliary targets only**. Do not feed engine variables into the world-model encoder in the initial B/D design. Both variants receive the same permitted target fields and loss specification. This makes the initial proposal choice concrete without introducing another experimental factor.

Local actors receive neither teammate features nor team reward as an input. Reward remains a training signal. Use parameter sharing between the two local actors, with separate hidden states; distinguish roles through locally visible spawn/scene cues. Agent identity is supplied to world-model action/view components. Adding identity to local actors would require a declared, consistently applied input change during pilots.

The deployed package contains the local visual encoder, recurrent actor, action mapping, and observation preprocessing. It does not require the shared world model, centralized critic, or teacher.

### 0.3 Timing and notation

Use this indexing convention in replay, model training, policy learning, and tests:

$$
(o_t^1,o_t^2),\ (a_t^1,a_t^2)
\longrightarrow (r_t,d_t,o_{t+1}^1,o_{t+1}^2).
$$

- One **joint transition** is one simultaneous two-player action decision and the resulting transition. Do not count it twice because there are two players.
- Start with four simulator tics per action decision. Record actual advancement, including shortened final steps.
- `terminated` means task success, task failure, death, or the task deadline as defined below.
- `truncated` means a valid externally cut episode, such as a collection block ending. It does not imply task failure.
- An imagination horizon is a rollout cutoff with a bootstrap value, not a predicted terminal event.
- A worker crash or missing synchronized observation is an infrastructure event, never a fabricated task failure.

Carry an integer episode clock `j_t` from replay, where zero denotes the first observation and each joint decision increments it once. The task limit `L` is part of the declared scenario. Initialize the clock at an imagination anchor, advance it deterministically, and terminate by `j_(t+1) >= L`. Supply normalized remaining time to the model/critic; this is known elapsed history, not a future simulator observation. Local actors still receive their declared local interface. Display the task countdown in their local rendered HUD so the deadline is also locally observable.

### 0.4 Implementation organization and dependencies

Proposed repository organization:

```text
configs/
  base.yaml                 # common settings; the single home for tunable defaults
  tasks/{relay,cover}.yaml
  variants/{B,D}.yaml       # D adds teacher/distillation; common settings inherited
  experiments/{pilot,final}.yaml
src/
  env/                     # coordinator, workers, observation/action adapters
  data/                    # real replay, episode manifests, sequence sampling
  models/                  # visual encoder, shared dynamics, decoders, heads
  policies/                # local recurrent actor, teacher, centralized critic
  training/                # model fitting, imagination, MAPPO, distillation
  evaluation/              # isolated evaluator, metrics, paired analysis
  infrastructure/          # configuration, checkpoints, counters, resources
scenarios/{relay,cover}/   # editable map/ACS sources and generated assets
scripts/                   # CLI entry points for each phase
tests/                     # interface, learning-math, and integration checks
artifacts/                 # ignored generated runs, manifests, metrics, checkpoints
docs/                      # protocol, environment contract, runbook, findings
```

Use a tested Python/PyTorch/ViZDoom stack, a pinned environment lock, and versioned scenario tooling. Prefer Linux for the substantial training runs if available; retain portable Python interfaces for the current Windows workspace. Select exact dependency versions after the two-player smoke test on the actual GPU host, and record CUDA/driver versions and GPU/CPU/RAM/storage inventory. Do not infer simulator throughput from the GPU count.

Begin with one learner per GPU rather than mandatory distributed training. Use mixed precision where numerically appropriate, with probability calculations, losses, and optimizer states in float32. Start data-loader workers at zero and limit CPU thread pools; increase only after profiling. Four CPU cores must also serve the two game engines and coordinator.

Phase dependency: **1 → 2 → 3 → 4 → 5 → 6**. Evaluation and checkpoint infrastructure starts in Phase 1 and is exercised throughout. Proceed through the gates below; successful research does not require D to win.

## Phase 1 — Establish cooperative tasks and the evaluation interface

### P1.1 Pin the simulator and validate two-player synchronization

1. Install and record the engine, Python bindings, base game assets, scenario compiler, and renderer configuration. Pin asset hashes and retain licensing/provenance information for distributable maps and base assets.
2. Implement `TwoPlayerCoopEnv` with one coordinator and two participant workers, each owning one `DoomGame`. One hosts and the other joins locally, with unique ports and working directories for each joint environment.
   Configure cooperative scenarios explicitly rather than inheriting deathmatch flags from a synchronization example. Disable automatic respawn; fix friendly fire and autoaim to disabled for the initial tasks and use identical settings in B/D.
3. Use synchronous `PLAYER` mode and off-screen rendering. Initialize both participants concurrently.
4. Collect both actions before advancing either participant. Release both workers, then wait for both results. Sequential blocking advancement can prevent the second participant from entering the synchronization barrier.
5. At every normal decision boundary, assert equal participant tics and the expected advancement. Attach episode and decision IDs to every request/response; reject stale replies.
6. Coordinate reset and termination across participants. Test the host/guest reset sequence explicitly rather than assuming a single-player reset is sufficient.
7. Add a watchdog and an error record containing worker status, last acknowledged tic, action dispatch ID, and episode ID. Recover at a fresh episode boundary; preserve consumed-interaction accounting.

Current official documentation and the official multiple-instance example support this starting approach, but determinism and reset behavior must be tested on the pinned setup. [ViZDoom modes](https://vizdoom.farama.org/api/python/enums/), [multiple-instance example](https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/main/examples/python/multiple_instances.py), [DoomGame API](https://vizdoom.farama.org/api/python/doom_game/).

### P1.2 Define the observation and action interfaces

Render RGB at 160×120 and resize consistently to **128×96**; tensors use `[channels, height, width] = [3,96,128]`. Store images as uint8 and convert to float32/bfloat16 `[0,1]` only at the model boundary. Validate the resizing implementation with saved frames. Keep readable local health/ammo indicators and locally visible task indicators. Exclude automaps, global scoreboards, debug overlays, object metadata, depth, and segmentation labels from actor inputs.

Start with this 12-action categorical space, using the same mapping in both variants:

| ID | Buttons |
|---:|---|
| 0 | No-op |
| 1–3 | Forward; backward; strafe left |
| 4–6 | Strafe right; turn left; turn right |
| 7–8 | Attack; use |
| 9–11 | Forward+attack; strafe left+attack; strafe right+attack |

All actions remain available to the actor. Do not expose state-derived action masks that reveal hidden world information. Tune turn speed and action repetition in the mechanics pilot; if four-tic repetition makes required aiming/interactions unreliable, test two tics and freeze the selected common setting.

Define local API signatures independently from the privileged environment wrapper:

```text
actor_step(local_rgb, previous_action, local_hidden)
    -> action_distribution, next_local_hidden
env_step(action_player_1, action_player_2)
    -> local_observations, shared_reward, terminal_flags, training_only_info
```

Copy renderer buffers when collecting observations. A subsequent engine step must not mutate images already stored in replay.

### P1.3 Implement task family 1: gate-and-relay navigation

**Mechanic:** One player holds a plate that opens a route for the teammate. The teammate reaches a latch that permanently releases the route, allowing both players to reach extraction.

- Prevent the plate-holder from completing both roles alone through travel timing or geometry.
- Give both roles persistent, locally visible cues: plate activation, gate status, latch confirmation, and extraction marker. Avoid an instantaneous hidden event that a local policy could never infer.
- Randomize the role-to-spawn assignment, route lengths, obstacle placement within valid bounds, and textures using the training manifest.
- Define success as both living players reaching extraction after the latch is set.
- Use a provisional deadline of **400 joint decisions**. Either player's death is terminal failure, although lethal hazards may be absent in the first mechanics fixture.
- Reward: `+1` for team success, `-1` for terminal failure, `-0.001` per decision, and `+0.1` once when the latch first activates. Do not reward repeated plate occupancy or repeated latch use.

### P1.4 Implement task family 2: cover-and-activate combat

**Mechanic:** One player operates an exposed console for a visible dwell interval while the teammate suppresses an enemy approach. A second stage changes the approach/console location, followed by joint extraction.

- Start with two activation stages, a **20-decision dwell interval**, and a provisional **600-decision deadline**.
- Use a small fixed weapon set and bounded ammunition sufficient for scripted completion. Keep health/ammo visible locally. Match enemy counts, damage, and spawning rules across variants.
- Accumulate progress only while a living player is inside the interaction zone and executes `USE`. The chosen action set makes console operation incompatible with attacking during that decision. Pause accumulated progress when either condition fails, display progress locally, and make completion an irreversible one-time event.
- Place enemies, cover, and the exposed operator so cover is useful. Provide persistent local cues for stage completion and extraction.
- Reward: `+1` for team success, `-1` for terminal failure, `-0.001` per decision, and `+0.05` for each first console completion. No per-damage, per-kill, resupply, or repeated-use reward initially.
- End the team episode on either player's death and disable automatic respawning.

For both families, evaluate human/scripted completion and run an inactive-teammate sanity check. These validate task mechanics; they are not additional learned comparison variants. Adjust difficulty during pilots if tasks are trivial, unsolvable, or fail to require cooperation. Freeze reward rules and task versions before final B/D runs. A failure reward takes precedence on death; otherwise success attained by the deadline takes precedence over deadline failure.

### P1.5 Instrument privileged targets and authoritative events

Maintain a versioned registry of fields, units, validity rules, and normalization. Initial targets:

- Per player: position, heading represented by sine/cosine, planar velocity, health, ammo, and alive status.
- Per task: stage, plate occupancy, gate/latch/console state, success/failure flags, and event counters.
- Stable task-object identifiers authored in the map; do not assume transient object IDs match across clients.

Use authoritative cumulative event counters and derive deltas between decisions so an event between two rendered observations is not lost. Calculate the shared reward once and verify both participants receive the same value. Export task variables through the supported ACS/game-variable mechanism; verify fixed-point conversions and reserve the engine's reward variable correctly. [Custom environments](https://vizdoom.farama.org/environments/creating_custom/), [game variables](https://vizdoom.farama.org/api/python/enums/), [GameState](https://vizdoom.farama.org/api/python/game_state/).

### P1.6 Specify termination, truncation, and splits

For the initial tasks, success, death, and the declared deadline are genuine terminals and receive zero value bootstrap. A collection-block cutoff is an external truncation with a valid final observation. Reset at the next collection block and record that boundary without labeling it task failure.

Capture final reward/events before resetting. If a final image is unavailable, mark it invalid and mask its image reconstruction target. Never use a fabricated zero or repeated frame as a true terminal observation. Engine crashes and synchronization failures produce error records, not reward-bearing transitions.

Provisional layout allocation per family: **six training templates, two validation templates, and two held-out test templates**, with versioned spawn/texture/enemy randomization banks. Procedurally generated layouts must pass reachability and mechanic checks. Hold out geometry, not just random seeds, for claims about unseen layouts. Publish the exact scope of the resulting test distribution.

Keep environment-validation trajectories separate from training. Any diagnostic/scripted data later used for learning must be entered into the training interaction ledger.

### P1.7 Acceptance tests and outputs

Complete these meaningful checks before model training:

- 10,000 joint decisions with zero unexplained tic mismatches or stale responses.
- 100 coordinated resets without cross-episode transitions or residual task state.
- Fixed-seed, fixed-joint-action replay reproduces event/reward/state traces on the pinned platform; investigate frame-hash differences and report the achieved determinism level.
- Explicit success, death, deadline, collection-cutoff, and worker-error tests follow their distinct paths.
- Each family has successful scripted/human executions and a functioning cooperation dependency.
- Holding a local input/history fixed while changing global diagnostic metadata leaves that actor's outputs unchanged.
- Benchmark one and two joint environments: transitions/s, tics/s, CPU/RAM use, and p50/p95 decision latency. Choose the stable collector count from measurements.

**Outputs:** environment adapters, scenario sources/assets, action/observation/reward contracts, split manifests, event registry, mechanics videos, and `phase1_acceptance.json` with the measured results.

## Phase 2 — Build the shared world-model foundation

### P2.1 Collect and store real replay

For each task-family/training-seed pair, collect **50,000 initial joint transitions**, attributed in full to both B and D. Physically collect this initial dataset once and share an immutable copy.

Provisional collection mixture: 70% temporally correlated random exploration and 30% scripted mechanic exploration to cover latch/console, success, failure, and combat events. Scripts may use training-only state and are used to collect model data, not to supply actor imitation targets. Document this initialization as part of the method. All script interactions count toward the 50,000 budget. Keep the mixture common to B and D, and freeze it after pilots.

Store episodes in chunked uint8 arrays with light compression and JSON manifests; benchmark the storage/compression setting against the four-core limit. Use one stored observation sequence with transition indices, rather than duplicating every image as both `obs` and `next_obs`.

```text
Episode metadata:
  episode_id, task_family, layout_id, split, scenario_hash, engine_version
  environment_seed, run_seed, variant, collector_kind, policy_version
  initial_tic, end_reason, valid_length, dataset_version

Transition fields:
  transition_id, episode_id, decision_index, start_tic, end_tic, actual_tics
  observation_index, next_observation_index, next_obs_valid[2]
  actions[2], team_reward, terminated, truncated, termination_reason
  privileged_player_state[2,...], privileged_validity_mask
  task_state, cumulative_event_counters, event_deltas
  collector_policy_version, collection_block
```

Store player/task **state** arrays alongside the observation sequence: a target indexed by t describes the state corresponding to `o_t`, and auxiliary heads on `Z_t` predict that state. Reward, termination, and event-delta targets indexed by t describe the transition from t to t+1. Preserve independent validity masks: terminal events/reward may remain available when the terminal image or state target is missing.

Raw images require approximately **73.7 GB per million joint observations** at this resolution, before compression. Forty million-transition runs would approach 3 TB of raw images before shared-initialization savings, checkpoints, or compression. Measure actual storage and provision archiving before the final campaign; keep only an active cache in RAM.

Partition real episodes deterministically into 90% model-fitting and 10% model-diagnostic data using episode IDs. All collection still counts. Exclude diagnostic episodes from model fitting and policy imagination anchors. Maintain separate stores for real replay and synthetic PPO trajectories; do not fit the world model on its own generated images.

### P2.2 Specify the shared stochastic Transformer model

**Implementation status:** completed. See [architecture, interfaces, and checks](docs/phase22_architecture.md)
and [P2.2 verification](artifacts/phase22/phase22_acceptance.json). This verifies the
untrained architecture. P2.3 objectives are completed below; P2.4 fitting and
P2.5 quality acceptance remain pending.

Use a recurrent state-space design with Transformer state updates. At time t, maintain deterministic shared memory `M_t` and a shared categorical latent `z_t`. This provides temporal memory and uncertainty without requiring a separate large video-generation model.

| Component | Pilot specification |
|---|---|
| Image encoder | Shared CNN, channels 32/64/128/256, four stride-2 blocks; retain a 6×8 spatial feature grid per view. |
| Visual tokens | 48 tokens/view, width 256, spatial and agent-identity embeddings. |
| Deterministic memory `M` | 16 shared tokens × width 256. |
| Dynamics backbone | Four pre-normalized Transformer blocks, eight heads, FFN width 1,024. |
| Action tokens | One embedding per agent with action and agent identity. |
| Stochastic latent `z` | 32 categorical variables × 32 categories, sampled once for the joint world step. |
| Posterior | Cross-attend shared memory to both views' visual tokens, then predict categorical logits. |
| Prior | Predict categorical logits from the updated deterministic memory. |
| Observation decoder | Agent-specific spatial queries attend shared state; shared upsampling CNN outputs `[3,96,128]` per agent. |
| Reward head | Small MLP predicting the milestone-reward component on the source/action/next-state transition representation; combine with known step/outcome rewards. |
| Outcome/termination head | Categorical head for continue, success, or non-deadline failure; enforce the known deadline in the imagined wrapper. |
| Auxiliary heads | Normalized continuous player states, binary events, and categorical task stage. |

Represent `z` using straight-through categorical samples during model fitting. Mix a small uniform component into categorical probabilities if needed for stability; use a provisional 1% mixture and apply it consistently to prior/posterior KL calculations. Keep sampling reproducible through explicit RNG streams.

One concrete transition construction is:

$$
\bar M_t=M_t+P(z_t),\qquad
U_t=\operatorname{CrossAttn}(\bar M_t,[e(a_t^1,1),e(a_t^2,2)]),
$$
$$
M_{t+1}=F_\theta(\bar M_t+U_t),\qquad
p_\theta(z_{t+1}\mid M_{t+1}).
$$

`P` projects the flattened categorical sample into the 16 memory tokens. Add an embedding of normalized episode progress to the transition/posterior context; it is omitted from the equations for brevity. The posterior adds observation evidence:

$$
q_\theta(z_{t+1}\mid M_{t+1},E(o_{t+1}^1),E(o_{t+1}^2)).
$$

Construct the shared decoder/teacher representation `Z_t` from `M_t` and embeddings of `z_t`. Both local view decoders use the **same** `Z_t` and sample identity, with their own view identifier. Reward and termination may also depend on source state and joint action so transition events are not forced to be functions of the next observation alone.

Use a small amount of known task structure to handle deadlines exactly. The learned outcome head predicts `continue`, `success`, or `failure` for non-deadline events. A real deadline-only transition has outcome target `continue`; its terminal status is supplied by the clock rule. In imagination, sample one outcome for the joint transition, then apply this precedence: sampled failure, sampled success, otherwise deadline failure if the clock expires, otherwise continuation. Define

$$
\hat r_t=\hat b_t-0.001+R(c_t,j_{t+1}),
$$

where `b_t` is the bounded one-time milestone bonus and `R` is +1 for success, -1 for failure or deadline failure, and zero otherwise. Fit the bonus head to `b_t = r_t + 0.001 - R(true_outcome, j_(t+1))`, clipped only for assertion tolerance rather than to hide bad labels. Constrain its output to the task's per-transition bonus range. The shared sampled outcome determines both the terminal reward component and termination, preventing independent reward/done samples from contradicting those fixed rules. Accuracy of predicted outcomes, bonuses, and views still requires validation.

At episode reset, use a learned reset memory and infer `z_0` from the first synchronized observations. During fitting and replay-anchor initialization, observations update the posterior. After an imagination anchor, use only the prior, executed joint actions, and generated outputs. No future replay observations or true engine states may enter that rollout.

MARIE and the original world-model references provide architectural starting points; these exact dimensions and the recurrent shared-state construction are project design choices. [MARIE](https://arxiv.org/html/2406.15836v2), [MATWM](https://arxiv.org/abs/2506.18537).

### P2.3 Define model objectives and loss reductions

**Implementation status:** completed. See [objective definitions and interfaces](docs/phase23_objectives.md)
and [P2.3 verification](artifacts/phase23/phase23_acceptance.json). Fitting-only
normalization is frozen for all four approved pilot stores; forward/backward
checks performed no optimizer updates. P2.4 fitting and P2.5 model-quality
acceptance remain pending.

Use an explicitly reduced total loss:

$$
L_{WM}=100L_{RGB}+L_r+L_d+L_{aux}+0.5L_{dyn}+0.1L_{rep}.
$$

- `L_RGB`: mean squared error for `[0,1]` images, averaged over valid pixels, views, timesteps, and batch. The factor 100 is a starting scale, to be inspected during pilots.
- `L_r`: Huber loss on the milestone-reward component defined above, averaged over valid transitions; initial delta 1. Also log error in the reconstructed total team reward.
- `L_d`: cross-entropy for continue/success/non-deadline-failure. Deadline-only and external collection-cutoff transitions use `continue`; the separate clock rule determines deadline termination.
- `L_aux`: mean of per-field losses after normalizing continuous fields using fitting-data statistics; use Huber for continuous values, BCE for binary fields, and cross-entropy for stage. Mask unavailable fields. Give each declared field equal weight initially.
- `L_dyn = max(1 nat, KL(stopgrad(q) || p))` and `L_rep = max(1 nat, KL(q || stopgrad(p)))`. Sum the categorical KL over all 32 variables before applying the free-nat floor, then average over valid samples.

Do not compute a posterior KL or next-image loss when the required final observations are missing. Still train valid reward/termination targets using the transition prior. Initial-observation reconstruction and state supervision can be trained without a preceding reward target.

Inspect component losses, gradient norms, latent entropy, posterior/prior gap, and event imbalance. If sparse events are ignored, first improve collection coverage and inspect target alignment. Any event weighting or loss-scale change is a documented pilot change applied to both variants before protocol freeze.

### P2.4 Train, validate, and refresh the model

**Implementation status:** completed; research training is left to the GPU
server operator. See the [training and recovery runbook](docs/phase24_training.md)
and [CPU implementation checks](artifacts/phase24/README.md). The user-selected
launch layout is four independent parallel workers sharing one GPU, with one
task/seed per worker. Each effective batch uses exact component/per-field
denominators across memory-saving microbatches. Fitting-only frozen normalizers,
versioned checkpoint/RNG recovery, held-out diagnostic evaluation, separate real
validation-geometry banks, 16/32/64-step burn-in comparison and later branch
refresh are implemented. The default 5,000-update fits and P2.5 quality gate have
not been run locally; no usable learned model is claimed by implementation tests.

**Server status (user report, 2026-11-05):** P2.4 training is still running.
The local checkout/checkpoint copies may not be synchronized with the server;
actual optimizer counts, preserved checkpoints and completion remain unverified
here. P2.5 implementation work does not require stopping that training.

| Setting | Pilot default |
|---|---:|
| Optimizer | AdamW |
| Learning rate / weight decay | `1e-4` / `1e-4` |
| Effective sequence batch | 32 |
| Sequence length | 64 decisions: 16 burn-in + 48 loss-bearing steps |
| Gradient clipping | Global norm 100 |
| Initial model updates | 5,000 |
| Model updates per later 25,000 real transitions | 500 |
| Sampling mixture after initialization | 50% valid anchors from all fitting replay; 50% from most recent 100,000 transitions |

Keep sequences within episodes, mask padding, and preserve reset boundaries. The initial fitting implementation may use truncated burn-in from a window; explicitly treat this as an approximation and compare 16/32/64-step burn-in on validation before freezing. Behavior-generation anchors use complete-prefix reconstruction as specified in Phase 3. Cache states only with their exact model-version key.

Use gradient accumulation if microbatches are necessary; preserve effective batch size and loss scaling. Record model update count separately from real and imagined transition counts. Shared initialization means the same checkpoint and normalization statistics within a B/D seed pair; later model updates use each branch's own replay.

### P2.5 Model validation and acceptance gate

**Implementation complete locally:** compact evaluator, source/provenance
checks, baseline metrics and reports are implemented. CPU verification passes
55 targeted tests; the existing four replay cohorts are audited separately.
See [implementation evidence](artifacts/phase25/README.md). Trained-checkpoint
quality, server timing and exploratory readiness still require the server run.

**Approved pilot scope:** use a compact engineering gate before expanding policy
experiments. This checks the common world-model foundation for B and D; it is
not another experimental arm. Retain **relay and cover, seeds 0 and 1**: evaluate
the four shared initial checkpoints once each, rather than separately for B/D.
See the [compact validation guide](docs/phase25_validation.md).

1. Reuse recorded transition-indexing, action-identity, shared-sample,
   future-leakage and optimizer/overfit checks where available. A structural
   action path does not by itself establish learned action-effect accuracy.
2. Select the same explicit optimizer update for all four models before
   examining their scores. Intermediate checkpoints may be screened; screening
   does not finalize an unfinished P2.4 schedule or create B/D bindings.
3. Time four to eight windows on one checkpoint first, including replay I/O,
   complete-prefix reconstruction and inference. Then evaluate **at most 64
   windows per model**: up to 32 uniform and up to 32 distinct event-targeted
   anchors. Preserve the sampling seed and report the actual selected cohort.
   No half-day or shorter runtime has been measured or guaranteed.
4. Reconstruct each anchor from its complete real prefix without gradients.
   Reuse prefix states only under their exact checkpoint, replay/episode,
   prefix and sampling identities. From each anchor, run one **prior-only,
   recorded-action rollout up to 15 steps** and extract metrics at 1, 5, 10 and
   15 steps. Future real observations and privileged labels are scoring targets
   only; never correct the autonomous rollout with them.
5. Compare RGB predictions with image persistence, outcomes/milestones with
   fitting-only frequencies, and reward with their task-rule baseline, including
   the exact deadline. Report reward error, outcome/termination behavior,
   stage/milestone prediction and image error. Player-state errors remain
   deferred formal validation work. Keep uniform and event-targeted strata
   separate; report valid support, unique episodes and distinct positive event
   instances. Overlapping windows are not independent trials. Missing terminal
   RGB does not remove valid reward/outcome supervision. Early model termination
   must appear in termination/error/coverage diagnostics, not be hidden by
   dropping the difficult rows or forcing the model to continue.
6. Inspect sparse latch/console/success/death labels explicitly. Zero held-out
   death examples means **death prediction is not validated**, not a pass.
   Low reconstruction loss and majority-class accuracy are insufficient.
7. Record `pilot_readiness` separately from `P2.5_quality_acceptance: false`.
   Existing model-diagnostic episodes are held out from fitting but use training
   geometry; this engineering screen cannot establish held-out geometry or
   final-test performance. Agree on a supported operating horizon shared by
   B/D before policy pilots; any change from the configured 15 is an explicit
   common pilot change, not an automatic configuration edit.

Initially defer new simulator collection, burn-in sweeps, videos, ablations and
broad tuning. Existing overfit evidence can be reused instead of repeating
training. Record absent evidence and unsupported classes as unresolved. The
earlier fixture targets of event/terminal F1 ≥0.9 and five-step cumulative
reward error ≤0.1 remain provisional debugging references, not universal pass
thresholds for this small screen.

**Outputs:** verified source/checkpoint/replay/normalizer identities, selected
windows and support, timing, baselines, separate-stratum rollout diagnostics and
an explicit pilot-readiness recommendation. A ready engineering screen permits
small exploratory Phase 3 integration work. Reports distinguish
`ready_for_exploratory_phase3`, `needs_review` and `failed`; four or eight timing
windows alone are not sufficient for advancement. A completed four-model
campaign also writes a receipt-bound `pilot_model_selection.json`, retaining
all four checkpoint identities and readiness even when review is needed. It
does not finalize an unfinished P2.4 run or replace its B/D bindings.
Treat the final campaign report and its SHA receipt as the completion marker;
check its selection receipt before using that separate pilot selection.

The provisional 15-step readiness checks are fixed in `compact.PILOT_THRESHOLDS`:
at least 16 uniform windows and 8 full-length uniform suffixes; at least 75%
prediction coverage and at most 25% false-early-termination windows in each
stratum; coverage-adjusted RGB MSE within persistence +0.02 and outcome Brier
within fitting baseline +0.05; both final cumulative and cumulative-path reward
MAE no greater than 1.25 × their respective baselines +0.05. In the event stratum,
each task milestone needs at least five distinct positives, recall ≥0.5,
precision ≥0.25 and coverage-adjusted Brier within its fitting baseline +0.02.
There must also be five distinct successes, success argmax recall ≥0.5 and
sampled-terminal recall ≥0.5 in that stratum. All four models must meet these
engineering alarms; no statistically calibrated quality acceptance is implied.

**Deferred formal acceptance:** retain the full requirement for transition and
action-effect correctness, overfit task-event reproduction, held-out
validation-geometry comparisons with persistence/constant predictors, autonomous
1/5/10/15-step image/player-state fidelity with supported rare-event/termination diagnostics, and
shared-sample/future-leakage checks. Freeze the supported common B/D operating
horizon before final experiments. Complete training diagnostics, appropriate
qualitative rollout evidence and `phase2_acceptance.json` remain required before
expanding beyond exploratory policy work or making confirmatory B/D claims.
Do not choose checkpoints or tune the horizon using final-test data.

## Phase 3 — Implement variant B: local MAPPO in imagination

Begin with a small end-to-end pilot after the compact P2.5 engineering screen
supports proceeding. Verify real-environment local-policy behavior before
expanding training; screen readiness is not evidence of policy competence.
Keep the common foundation and local MAPPO opportunities matched across B/D.
Phase 3 implementation and runtime are additional work, not included in the
P2.5 timing measurement.

### P3.1 Implement local actors and the centralized critic

Use an actor-specific CNN with the same four channel widths as the visual model encoder, followed by a flattened spatial projection to 256 features. These actor weights are separate from the world-model encoder. Concatenate the current image embedding with the previous action embedding/one-hot, then update a GRU with hidden size 256. Use a two-layer categorical action head on the resulting memory.

Include a distinct previous-action START symbol at reset; it is not an executable action. Set recurrent state to zero at a real episode reset. Use no actor dropout or batch-dependent normalization that would make replayed log-probabilities depend on minibatch composition; GroupNorm/LayerNorm are suitable starting choices.

$$
h_t^i=\operatorname{GRU}_\phi(h_{t-1}^i,[C_\phi(o_t^i),a_{t-1}^i]),
\qquad \pi_\phi^i=\operatorname{Categorical}(g_\phi(h_t^i)).
$$

The centralized team critic pools `Z_t`, concatenates detached actor memories `h_t^1,h_t^2`, and predicts one shared value through a 512/512 MLP. Use the same critic input convention for B and D students. The value function should account for what the local actors remember, not just the current physical/world state. [Scaffolder](https://arxiv.org/html/2405.14853).

Start with raw value targets and Huber value loss; bounded task rewards make this a simple baseline. Normalize advantages, not reward definitions. If value normalization becomes necessary during pilots, apply the same implementation to both variants and test normalized/unnormalized target conversions.

### P3.2 Construct valid imagination anchors

1. Sample 256 nonterminal anchor states from fitting replay. Use balanced anchor sampling from the current replay distribution, not privileged success selection available to only one arm.
2. Reconstruct the world posterior from the episode prefix under the current frozen model.
3. Reconstruct each actor's memory from **its own complete observation and executed-action prefix**, without gradients. Do not initialize local memory from `Z` or a stored hidden state from a different policy version.
4. Cache prefix results only under `(episode_id, prefix_length, world_model_version, actor_version)`. Invalidate after relevant parameters change. Reuse calculations across anchors from the same episode where possible.
5. At the anchor, the actor can use the real local observation. At future imagined steps, decode RGB from the sampled shared state and pass it through the same local actor encoder.

Process the anchor observation exactly once: `h_t` already includes `o_t` when sampling the first action. Initialize `j_t` from the anchor's recorded decision index. Never sample an anchor at or after its task deadline.

Use full prefixes as the correctness reference. If their cost is prohibitive, evaluate a fixed capped local burn-in against full-prefix logits/returns during pilots, document the approximation, and apply the same cap in B and D. Do not introduce a cap silently in final runs.

### P3.3 Generate imagined on-policy trajectories

Freeze the world model and behavior actor for each rollout/update cycle. For each imagined joint step:

1. Sample each local action from its categorical policy using independent actor RNG streams.
2. Advance the shared transition once using both actions; sample one shared next latent from the prior.
3. Advance the episode clock, sample the shared non-deadline outcome, and apply the clock/outcome reward and termination rules in P2.2. Record outcome probabilities and the resulting terminal cause for diagnostics.
4. Decode both local next images from that state. Convert to the same uint8-equivalent image interface used by real actors, then normalize identically.
5. Update local memories using those observations and the actions actually executed.
6. Record critic values and stop a trajectory at a sampled or clock-induced terminal, or at the fixed imagination horizon.

Imagination uses no policy-gradient path through the world model. Store local images, actions, behavior log-probabilities, old values, rewards, terminal/cutoff masks, relevant shared states and detached critic-memory inputs, anchor/prefix IDs, and all model/policy versions. Store these synthetic rollouts separately from real replay.

### P3.4 Implement recurrent MAPPO correctly

Use one team advantage and individual actor likelihood ratios:

$$
\rho_t^i=\frac{\pi_\phi(a_t^i\mid\text{local history}_t^i)}
{\pi_{old}(a_t^i\mid\text{local history}_t^i)}.
$$

$$
L_{actor}=-\mathbb E_{t,i}\left[
\min(\rho_t^i\hat A_t,
\operatorname{clip}(\rho_t^i,1-\epsilon,1+\epsilon)\hat A_t)
\right]-\beta\,\mathbb E_{t,i}H(\pi_\phi^i).
$$

Do not use the product of the two likelihood ratios. Normalize advantages across valid joint samples, and average the actor objective across valid player decisions. Compute the team critic loss once per joint sample.

For GAE, use:

$$
\delta_t=r_t+\gamma(1-d_t)V_{old}(t+1)-V_{old}(t),
\qquad A_t=\delta_t+\gamma\lambda m_t A_{t+1}.
$$

Here `m_t` is zero at a terminal or at the end of the stored trajectory. At a nonterminal imagination cutoff, the final TD residual includes `V(t+1)` even though the advantage recursion stops. Compute critic return targets as **raw GAE + raw old value before normalization**. Normalize a separate copy of GAE only for the actor objective.

For recurrent minibatches, preserve whole imagined sequences and their prefix references. Recompute current-policy memory from the stored local histories under the current actor parameters. Keep old log-probabilities immutable. At the first update before any parameter change, recomputed log-probabilities must reproduce the stored values. Prefix computations can run without gradients; keep gradients through the loss-bearing sequence. Stored critic-memory features and old values are fixed rollout data for that update.

| MAPPO setting | Pilot default |
|---|---:|
| Imagination horizon | 15 joint steps, subject to Phase 2 pilot gate |
| Starts per PPO collection cycle | 256 |
| PPO epochs / recurrent minibatches | 4 / 4 |
| Discount / GAE lambda | 0.99 / 0.95 |
| PPO clip epsilon | 0.2 |
| Actor and critic learning rates | `3e-4` |
| Optimizer | Adam, epsilon `1e-5`, no weight decay |
| Entropy coefficient | 0.01 |
| Value-loss coefficient / Huber delta | 0.5 / 1.0 |
| Gradient clipping | Norm 0.5, applied to actor and critic separately |

Log approximate KL, clip fraction, entropy, value error/explained variance, gradient norms, active trajectory length, and predicted termination frequency. Use a common fixed epoch count initially. Numerical divergence is a failure to diagnose, not a reason to silently grant one variant a different optimizer schedule. The official MAPPO implementation is a reference for update and recurrent-buffer behavior. [Official MAPPO repository](https://github.com/marlbenchmark/on-policy).

### P3.5 Define the real-data / model / policy cycle

The default per-run schedule is:

```text
Collect the shared initial 50,000 real joint transitions.
Fit the common initial world model for 5,000 updates.
Initialize paired local actors/critics.
For B: perform 100 local PPO collection/update cycles in imagination.
Save the 50,000-interaction evaluation checkpoint.

Repeat until total real count reaches 1,000,000:
    Collect 25,000 real joint transitions using the current local actor pair.
    Add valid real episodes/segments to this branch's replay.
    Fit the world model for 500 updates; freeze the resulting snapshot.
    Perform 20 local PPO collection/update cycles in imagination.
    Save a completed-block checkpoint and evaluation metadata.
```

There are 38 online collection blocks and **860 local PPO collection/update cycles** under these defaults. B and D students receive the same schedule. The nominal local imagined volume is `860 × 256 × 15 = 3,302,400` joint steps before early termination, with two local decisions per joint step. Log actual volume and optimizer steps as well.

Collection uses stochastic local policies at temperature 1.0. Real data are used for world-model fitting and imagination anchors; the initial comparison does not additionally perform PPO updates directly on real episodes. Stop collection at the exact block budget, recording a nonterminal external cutoff if necessary, then start a fresh episode for the next block.

### P3.6 Acceptance gate

- Hand-calculated GAE fixtures verify terminal, nonterminal cutoff, and multi-step returns.
- Clock fixtures verify exact deadline termination, the success/failure precedence at the deadline, and reward decomposition without double-counting terminal penalties.
- Frozen-behavior replay reproduces old log-probabilities within numerical tolerance; the initial PPO ratio is approximately one.
- Locality holds for both actor outputs and memory updates; changing global metadata cannot change fixed-local-history outputs.
- Actor/critic updates leave the world-model parameter hash unchanged.
- Padding, episode resets, and independently sampled actor RNG streams behave correctly.
- A small mechanics task shows trainable policy behavior and valid real evaluation. Diagnose persistent chance-level behavior before a large campaign, without requiring B to reach a particular final research score.

**Outputs:** working B pipeline, invariant tests, real-evaluation checkpoints, local-actor export, learning diagnostics, and `phase3_acceptance.json`.

## Phase 4 — Implement variant D: teacher, distillation, and local refinement

### P4.1 Implement and train the globally informed teacher

Use the local actor's CNN/GRU dimensions for the teacher's local history branch. Add cross-attention from the teacher's current memory plus agent identity to the current shared world tokens:

$$
c_t^i=\operatorname{CrossAttn}(Q(h_{T,t}^i,i),Z_t),\qquad
\pi_T^i=\operatorname{Categorical}(g_T([h_{T,t}^i,c_t^i])).
$$

Global context enters the action head, not the teacher's recurrent memory. This makes its memory reconstructible on student-visited trajectories. Use an initial four-head teacher attention module and a width-256 head; exact teacher capacity is recorded separately from deployed student capacity.

Train teacher MAPPO with the same reward, environment dynamics, termination convention, and PPO implementation. Teacher actions generate teacher training rollouts. Initially perform **100 teacher PPO collection/update cycles** on the common model. Refresh the teacher for **20 cycles per online block** while student guidance remains active, using the newly frozen model snapshot.

The initial experiment collects subsequent real experience with local students only. Any later decision to collect real teacher experience must be made before final protocol freeze, counted fully, and reflected in the method description.

### P4.2 Warm-start local students through action distillation

1. Initialize the local student and its critic from the paired initialization used by B, before B's learning. Do not copy the larger teacher's encoder, critic, hidden state, or global-attention module into the student.
2. Generate a fixed distillation buffer of **512 teacher trajectories** of the selected imagination horizon from valid real-prefix anchors.
3. Store soft teacher action distributions and the corresponding local observation/executed-action histories. Use temperature 1.0 initially.
4. Perform **200 supervised minibatch updates**, each using 64 full trajectory sequences sampled from this buffer. Reconstruct student prefix memory locally; teacher actions in these replay histories are the actions that were actually executed.
5. Optimize forward KL from the stopped-gradient teacher distribution to the student. Keep the world model and teacher frozen. Do not initialize the student critic from privileged teacher values.

This is an additional supervised warm-start budget in D and must be reported as such. It does not replace any of the 100 initial local PPO cycles that B receives.

### P4.3 Refine on student-generated imagined trajectories

After warm-start, use student actions to generate the rollouts used for student MAPPO. Run the frozen teacher alongside the same local observation/action history to provide distributions at student-visited states.

The teacher's previous-action input is the **executed student action**, not an independently sampled teacher action. Initialize teacher memory from the same real prefix, using its local encoder, and update it along the actual imagined student trajectory. Teacher queries may use the current shared state, while student inputs remain local.

$$
L_S=L_{actor}^{PPO}+\lambda_k\,
\mathbb E_{t,i}D_{KL}\left(
\operatorname{stopgrad}[\pi_T^i(\cdot\mid h_{T,t}^i,Z_t)]
\parallel \pi_S^i(\cdot\mid h_{S,t}^i)
\right).
$$

Use the same value loss as B. The teacher KL is an auxiliary supervised term; it does not change which student probabilities enter the PPO ratio. Save teacher target distributions with the rollout, so targets remain fixed through its PPO epochs.

Provisional guidance schedule, indexed by completed local PPO collection/update cycles:

$$
\lambda_k=1.0\max\left(0,1-\frac{k}{0.75K}\right),
\qquad K=860\text{ under the default budget}.
$$

The final quarter of local cycles therefore uses zero distillation weight. Stop teacher refresh after guidance reaches zero. Compute `K` from the frozen training schedule if pilot budgets change; do not extend guidance in response to final-test results.

Privileged teachers can choose actions that cannot be inferred from a student's local history. Local RL refinement is the mechanism for adapting to that information constraint; matching teacher performance is not an assumption. [Privileged-information analysis](https://arxiv.org/html/2412.00985).

### P4.4 Integrate D into the common schedule

```text
Load the paired 50,000-transition dataset and initial world-model checkpoint.
Train teacher for 100 imagined PPO cycles.
Warm-start the paired-initialized local students through action distillation.
Perform the same 100 initial local PPO cycles as B, with scheduled guidance.
Save the 50,000-interaction local-student checkpoint.

For each later 25,000-transition block:
    Collect real data with the current local students.
    Update the branch's world model for the same 500 updates as B.
    If guidance is active, refresh the teacher for 20 imagined PPO cycles.
    Freeze model and teacher snapshots.
    Perform the same 20 local PPO cycles as B, using student actions.
    Save checkpoint and accounting records.
```

Teacher/model snapshots remain fixed within each collection/update cycle; the simpler initial implementation freezes them across the complete local-learning block. Teacher/model refresh happens only between blocks. D uses the same real interaction and local MAPPO schedule as B, with additional teacher/distillation computation.

### P4.5 Acceptance gate and diagnostics

- Verify teacher global attention influences teacher logits on a controlled fixture, while the local student's interface contains no global inputs.
- Verify the teacher's shadow history uses executed student actions.
- Recompute stored teacher targets under the frozen snapshot and confirm agreement.
- Verify teacher and model parameters receive no gradients from student PPO/distillation.
- Confirm local actor/critic parameter counts and optimizer defaults match B.
- Confirm guidance reaches exactly zero for the final planned stage.
- Run the exported student pair without loading teacher/world-model modules and reproduce its local-policy outputs.
- Plot real student success/return alongside teacher reference performance and distillation KL. Low KL is not an acceptance proxy for successful cooperation.

**Outputs:** D teacher/student implementation, action-distillation buffers, local exports, transfer diagnostics, and `phase4_acceptance.json`. Engineering correctness is required; a D advantage remains the research hypothesis.

## Phase 5 — Run the controlled B-versus-D study

### P5.1 Stage the experiments and freeze the protocol

| Stage | Default scope | Selection use |
|---|---|---|
| Mechanics and math checks | Small fixtures and fixed batches | Correct implementation and task definitions. |
| End-to-end pilots | Two disjoint training seeds per variant/family; up to 200,000 real transitions/run | Choose workable common settings and D guidance settings using validation data. |
| Confirmatory runs | Ten paired training seeds per family; B and D | Estimate the frozen B/D effect. |

Allow up to three documented policy-training configuration candidates per variant on the same pilot seeds, after the common world-model foundation is chosen. Match tuning effort. Tune task design, shared model sizes, and horizons using joint feasibility/model diagnostics; then apply the selected common foundation to both variants. Log all tried candidates, failed pilots, and the selection rule.

If neither variant learns useful behavior within the pilot budget, investigate environment mechanics, reward coverage, imagination errors, and optimization before extending experiments. If one million transitions is unsuitable, revise the budget and derived schedules for both arms before confirmatory runs. Avoid selecting settings only because they maximize D-minus-B.

Create a protocol manifest containing code/config/task hashes, initial-data procedure, model/actor sizes, losses, budgets, seed lists, split manifests, checkpoints, evaluation rule, primary metric, statistical analysis, and infrastructure retry policy. Freeze it before starting the final seed set. Final runs use the final scheduled checkpoint, not whichever checkpoint has the best test score.

### P5.2 Final run matrix and shared initialization

Default final matrix: **2 families × 10 seed pairs × 2 variants = 40 training runs**, with **1,000,000 total real joint transitions per run**. This is a total campaign count, not a concurrency request.

Use provisional pilot master seeds 0 and 1 and final master seeds 1001–1010. Derive actual RNG streams from `(task_family, master_seed, component, branch)` using a recorded deterministic derivation. Component namespaces cover collection, model initialization/fitting, local actor initialization, teacher, student action sampling, imagination, and evaluation. Share only the intended initial streams within a B/D pair; subsequent branch streams and the two families are independent. Use separate per-agent sampling streams and separate train/validation/test environment seed namespaces.

For each family/seed pair:

1. Generate the initial 50,000-transition dataset and fit the initial world model once.
2. Save its data/model/normalization hashes and paired local actor/critic initialization.
3. Branch B and D into separate run directories and replay manifests.
4. Attribute the full 50,000 initial interactions to each run's learning-curve axis. Report physical shared collection once in experiment-wide resource totals.
5. Thereafter allow each local policy to generate its own data. Keep architectures, data field availability, model-update schedules, local PPO budgets, and evaluation conditions matched.

A paired initialization reduces irrelevant variation. Later data/model divergence is part of the pipeline treatment; it does not justify describing the later model weights as identical.

For standalone per-method compute accounting, attribute the initial model-fitting cost to each method; for actual campaign consumption, count the physically shared fit once. Label these accounting views explicitly.

### P5.3 Define and preserve the interaction ledger

Maintain monotonic counters for:

```text
real_joint_transitions_total
real_joint_transitions_valid_for_replay
real_simulator_tics_total
real_initial / real_local_collection / real_teacher_collection
imagined_joint_transitions_local / imagined_joint_transitions_teacher
world_model_optimizer_steps
local_PPO_cycles / local_actor_steps / local_critic_steps
teacher_PPO_cycles / teacher_optimizer_steps / distillation_optimizer_steps
evaluation_joint_transitions / evaluation_simulator_tics
```

Reusing replay does not increase the real interaction count. A joint decision contributes one count, independent of the number of players. Commit action-dispatch IDs to a durable ledger so a crash cannot roll counters backward. Count executed decisions even if their outcomes are unusable for learning; record uncertain interrupted dispatches conservatively and flag them. Never turn missing results into task-failure rewards.

Evaluation episodes never enter training replay or update model/normalization statistics. Exclude them from the training-budget axis, but include their real simulator usage in resource accounting. Likewise report development/pilot collection separately from confirmatory sample budgets.

### P5.4 Predefine checkpoints and evaluation policy

Primary curve checkpoints are **0; 50,000; 100,000; then every 100,000 through 1,000,000** real joint transitions, giving 12 checkpoints.

- At zero, evaluate the paired initialized local actors, before learning.
- At 50,000, evaluate after each arm's prescribed initial training block, including D's teacher and student preparation.
- At later checkpoints, evaluate the local actor after the completed block at that count.
- Never restart D's horizontal axis when distillation finishes. Additional learning at fixed real count adds compute, not samples.
- Save intermediate block checkpoints for recovery; they do not alter the predefined primary AUC grid.

Use stochastic sampled-policy evaluation with temperature 1.0 and independent RNG streams for the two actors. Pair the environment and per-actor sampling seed lists across B and D. Actual trajectories naturally differ after different actions.

For each family, create a **50-episode held-out test bank** shared by all scheduled checkpoints and runs, balanced across the two test layouts. Evaluate final performance additionally on **200 test episodes**, including that 50-episode bank. Use the fixed 50-episode estimates at every AUC checkpoint, including the final one; the 200-episode result is a secondary endpoint.

During pilots use validation banks with the same evaluation mechanics. For final runs, save checkpoints and evaluate test curves using a separate evaluator after configurations are frozen, preferably after training is complete. Keep test scores unavailable to tuning and training decisions. Report any exceptional test inspection and resulting protocol change.

### P5.5 Primary analysis and uncertainty

For task family `f`, variant `v`, training seed `s`, checkpoint budget `n_j`, and success fraction `p_fvsj`, calculate normalized trapezoidal AUC per run:

$$
A_{fvs}=\frac{1}{N}\sum_{j=0}^{J-1}
(n_{j+1}-n_j)\frac{p_{fvsj}+p_{fvs,j+1}}{2},\qquad N=1{,}000{,}000.
$$

The result is in `[0,1]`. Form paired differences `d_fs = A_fDs - A_fBs`. Average differences within each family, then give the two family means equal weight. Do not pool all episodes into an apparent sample size of thousands of independently trained agents.

Use 10,000 paired bootstrap resamples of training-seed pairs **within each fixed family**. Retain both variants and all checkpoints for every sampled pair. Report the mean difference, a 95% bootstrap interval, family-specific differences, and individual seed results. The uncertainty pertains to these fixed task families/test banks and training randomness, not all possible cooperative tasks.

A positive interval excluding zero supports the primary hypothesis. An interval crossing zero is inconclusive at the achieved precision; it does not prove equivalence. Use **0.02 normalized AUC** as a provisional practical-effect reference, frozen before final runs, and report the actual magnitude rather than only a sign or significance decision. Pointwise learning-curve bands are descriptive, not separate hypothesis tests at every checkpoint.

Implementation checks for the analysis script: a constant success curve has the same AUC as its constant value; identical arms have zero paired difference; swapping B/D negates the result; unequal checkpoint spacing is weighted correctly; incomplete curves cannot silently become complete runs. [Statistical reporting reference](https://arxiv.org/abs/2108.13264).

### P5.6 Secondary metrics and attribution

Report:

- Final 200-episode team success rate, mean return, completion time on successful episodes, and death/deadline failure frequencies.
- Per-family AUC and final performance, with layout-specific summaries to expose concentrated gains.
- Teacher/student gap on the same diagnostic scenario distribution when teacher evaluation is performed. Supply the teacher through its trained shared-model posterior interface, not an untrained direct engine-state shortcut.
- Imagination reward/termination error and drift on branch-specific real holdouts, plus a common diagnostic bank where appropriate.
- Actual real and synthetic volume, local/teacher/distillation optimizer steps, GPU-hours, CPU-hours, elapsed time, and peak memory/storage.

Teacher scores are privileged references, not guaranteed upper bounds on decentralized performance. Higher D sample efficiency can coexist with higher computation. If local PPO budgets, task definitions, or privilege availability diverge, disclose the deviation before attributing a gain to the planned treatment.

### P5.7 Schedule work on four GPUs and four CPU cores

Start with one active two-player collector/evaluator. Train other runs from their cached replay on available GPUs while collection rotates across runs. Expand to two joint environments only if Phase 1 profiling supports it. Avoid scheduling four multiplayer collectors simply because four GPUs are available.

Use single-GPU learners initially; spread independent seeds or teacher/model jobs across GPUs only when their dependencies and memory demands allow it. Keep update order deterministic at the logical block level. Record active training time separately from queue/wait time so scheduling does not masquerade as algorithmic efficiency.

Estimate remaining runtime from measured transitions/s and updates/s:

```text
training collection time ≈ required real transitions / sustained collection rate
evaluation time ≈ total planned evaluation decisions / evaluation collection rate
learner time ≈ sum of measured model, local-PPO, teacher, and distillation work
```

Do not infer a calendar duration before measuring these rates. Allocate storage for archived replay, checkpoints, and logs before launching the full seed matrix.

### P5.8 Recovery, failed runs, and completion gate

Save model/actor/critic/teacher parameters, optimizer states, RNG states, counters, configuration hash, replay manifest, and teacher/model version identifiers at completed collection blocks. Also journal consumed real interactions independently of model checkpoints. At a crash, resume from a coherent checkpoint and replay ledger without subtracting consumed interactions; restart the simulator at a fresh episode and document any discarded partial rollout.

Journal each block's **absolute real-count target** and completed stage/update IDs. If a crash occurs partway through a 25,000-transition block, collect only the remaining amount to that original boundary. Resume model, teacher, and local stages from consistent snapshots so logical updates are not applied twice. Re-executing work from a stage-start snapshot is permissible if it reconstructs the intended parameter state; log the additional physical compute separately. If exact logical recovery is unavailable, record a protocol deviation rather than silently changing the 860-cycle schedule or evaluation budgets.

Predefine up to two retries for reproducible infrastructure failures using the same seed/config. Numerical divergence or poor performance is an algorithmic outcome, not a reason to replace the seed. Preserve the last valid actor; if the protocol specifies continued evaluation of it after divergence, clearly label the stalled run. Report unresolved missing runs explicitly; do not silently calculate the main result from only successful pairs. If necessary, provide bounded AUC sensitivity using the `[0,1]` range and label the campaign incomplete.

**Outputs:** protocol manifest, all run/evaluation records or explicit failure accounting, analysis scripts, learning curves, paired effect estimates, resource table, and `phase5_acceptance.json`. Completion depends on auditable evidence, not the direction of the result.

## Phase 6 — Interpret results and assemble research deliverables

### P6.1 Diagnose outcomes using the recorded evidence

For every family, inspect representative successful and failed episodes from both variants using a predefined sampling rule, not only favorable examples. Classify failures into mechanics/observation ambiguity, missing local information, model exploitation, inaccurate rewards/termination, unstable policy learning, or unsuccessful teacher transfer.

Relate failures to model rollout diagnostics, student entropy/KL, teacher reference behavior, and real task-event traces. Treat these as explanations supported by observations, not causal attribution to individual components that the two-arm experiment did not isolate.

| Observed result | Supported interpretation |
|---|---|
| D improves AUC and final success | Teacher-guided training helps sample efficiency and final deployment performance in the tested setting. |
| D improves AUC but final scores converge | D learns useful cooperation earlier under the tested budget. |
| D improves final score without AUC improvement | Benefit appears late; the primary sample-efficiency hypothesis is not supported by that fact alone. |
| Improvement occurs in one family only | Benefit is task-dependent; report the family-level evidence and aggregate uncertainty. |
| No clear difference or D is worse | Added guidance has no demonstrated benefit, or harms learning, under this protocol. |

### P6.2 Produce the reproducible release package

Prepare:

1. Source, dependency lock, scenario sources, compiled task assets where redistributable, and version/hash manifests.
2. Frozen B/D configurations and a run matrix that expands into the recorded experiments.
3. Commands for environment smoke tests, initial collection, world-model training, B training, D training, evaluation, and analysis.
4. Local deployment exports with input/action contracts and a minimal two-player evaluation example.
5. Checkpoints and replay manifests; document how to obtain or regenerate larger datasets when distributing the full archive is impractical.
6. Episode-level evaluation records, paired metric tables, learning-curve plots, resource measurements, and representative videos.
7. A failure/deviation log, including pilots, missing runs, protocol changes, and hardware/software limitations.

Proposed CLI contract to implement, not commands claimed to work today:

```text
python -m scripts.validate_env --task relay --config configs/base.yaml
python -m scripts.collect_initial --task relay --seed <seed> --budget 50000
python -m scripts.train_world_model --manifest <initial-manifest> --config <config>
python -m scripts.train --variant B --task relay --seed <seed> --config <frozen-config>
python -m scripts.train --variant D --task relay --seed <seed> --config <frozen-config>
python -m scripts.evaluate --checkpoint-manifest <manifest> --bank <frozen-test-bank>
python -m scripts.analyze --protocol <protocol-manifest> --runs <run-manifest>
```

### P6.3 Draft the research report and verify claims

Write the report around the central question, information constraints, integrated method, B/D protocol, primary result, secondary results, resource cost, and limitations. Include the actual computational and real-interaction budget, initial scripted-data procedure, task-distribution scope, and the distinction between shared initial models and later branch-specific models.

Use prior work to explain the components and the integration context. Avoid claims that the project invents Transformer world models, MAPPO imagination, privileged learning, or global-to-local distillation. State every performance claim with its measured setting and uncertainty.

Regenerate reported tables and figures from archived evaluation records in a clean environment. Verify that exported local actors require only their declared local inputs and that the reproduction instructions resolve to the recorded configuration/model/task versions.

**Outputs:** research report, reproducible release package, final claim-to-evidence checklist, and `phase6_acceptance.json`.

## 7. Cross-phase risk register and review checkpoints

| Risk | Early evidence | Planned response before final protocol freeze |
|---|---|---|
| Multiplayer desynchronization | Tic mismatches, blocked workers, inconsistent rewards | Fix barrier/reset handling before fitting any model. |
| Four-core collection bottleneck | High CPU use and poor scaling beyond one environment | Serialize/stagger collection and evaluate from saved checkpoints; retain GPU work on cached replay. |
| Sparse event coverage | Missing success/latch/console/death targets | Adjust the shared initialization collector within a documented counted budget. |
| Good one-step images but poor rollout control | Drift in event/reward predictions | Improve model fitting and shorten the common horizon during pilots. |
| Local information leakage | Metadata perturbation changes local logits | Repair interfaces; rerun affected results. |
| Privileged teacher behavior is unlearnable locally | Large real teacher/student gap despite low training loss | Inspect task observability and guidance strength in pilots; retain the final unguided RL stage. |
| Teacher/model changes invalidate PPO | Changed target distributions or model hashes within a cycle | Freeze snapshots and invalidate stale rollout buffers. |
| Dataset/storage overload | Slow reads, compression saturation, full disks | Benchmark chunking/compression and provision archive capacity before the campaign. |
| Test-driven tuning or seed selection | Configuration changes after test inspection | Preserve the deviation record and use a fresh untouched test protocol for revised claims. |

Review at each phase gate. Numerical defaults may change during pilots with a recorded rationale. After protocol freeze, a substantive correction requires versioning the protocol and consistently rerunning the affected B/D comparisons; it must not be applied selectively to unfavorable runs.

## 8. Reference map

| Reference | Role in this plan |
|---|---|
| [ViZDoom official repository](https://github.com/Farama-Foundation/ViZDoom) and [API](https://vizdoom.farama.org/api/python/doom_game/) | Simulator integration, reproducibility, and supported control interfaces. |
| [MARIE](https://arxiv.org/html/2406.15836v2) | Transformer multi-agent world modelling, centralized aggregation, and imagined MAPPO-like learning. |
| [MATWM](https://arxiv.org/abs/2506.18537) | Multi-agent Transformer world-model design and sample-efficiency context. |
| [Official MAPPO implementation](https://github.com/marlbenchmark/on-policy) | PPO/recurrent buffer and training implementation reference. |
| [Scaffolder](https://arxiv.org/html/2405.14853) | Locally informed policies trained through privileged world modelling and memory-aware critics. |
| [PTDE](https://ijcai.org/proceedings/2024/4) | Prior work on global-to-local information transfer for decentralized execution. |
| [Privileged-information analysis](https://arxiv.org/html/2412.00985) | Limits of privileged-teacher distillation under partial observability. |
| [IRIS](https://arxiv.org/abs/2209.00588), [TransDreamer](https://arxiv.org/abs/2202.09481), [Dreamer 4](https://arxiv.org/abs/2509.24527) | Visual imagination architecture and training-loop context; this plan keeps its own shared recurrent Transformer and MAPPO choices. |
| [Deep RL at the Edge of the Statistical Precipice](https://arxiv.org/abs/2108.13264) | Motivation for uncertainty-aware experimental reporting. |

The exact architecture dimensions, task specifications, schedules, gates, and budgets in this document are project choices. They require the stated pilot checks and are not presented as reproduced results from these references.

