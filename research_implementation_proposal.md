# Research implementation proposal

**Working title:** Shared Imagination and Policy Distillation for Decentralized Visual Cooperation

**Status:** Preliminary proposal for review. The detailed implementation plan will follow after approval.

This proposal consolidates the decisions in [conversation_transcript.md](conversation_transcript.md) and the latest instruction to evaluate only **variant B and variant D**. It proposes a systems and empirical research contribution built around Transformer world modelling, MAPPO, shared world tokens, policy distillation, and cooperative ViZDoom.

## 1. Research question and intended contribution

**Central research question**

> Does globally informed teacher training followed by policy distillation and local MAPPO refinement improve the real-environment learning efficiency of two decentralized visual agents, compared with training local MAPPO actors directly in the same shared-world-model framework?

**Primary hypothesis**

> At an equal total budget of real environment interactions, variant D will achieve higher area under the deployed agents' cooperative success-rate learning curve than variant B.

Final team success rate and return will provide secondary evidence. Evaluation will measure the two deployed local actors together in the real simulator. Teacher performance will be a diagnostic reference.

The intended contribution is an integrated training framework and a controlled evaluation of whether its teacher-guided training procedure improves decentralized cooperation. The comparison measures the complete procedure: teacher learning, action distillation, and subsequent local refinement. It does not separately establish the benefit of imagination, shared tokens, or distillation alone, nor demonstrate synergy between imagination and distillation.

**Proposed contribution statement**

> We develop an integrated framework combining shared Transformer world modelling, imagined MAPPO, and centralized-to-local policy distillation. Through a controlled comparison in two-player cooperative ViZDoom, we investigate whether teacher-guided training improves real-environment learning efficiency over direct local policy learning within the same world-model framework, while preserving strictly decentralized execution.

These are proposed contributions and hypotheses; the experiments will determine the resulting claims.

## 2. Agreed scope and proposed design

| Component | Proposal |
|---|---|
| Environment | Two distinct cooperating ViZDoom players, partial visual observations, and a shared team reward. |
| Deployment | Each actor uses its own local observations, previous actions, and recurrent memory. There is no explicit inter-agent communication channel or global actor input. |
| World representation | One shared world representation containing a flexible number of tokens, with temporal memory and a representation of uncertainty. |
| Joint transition | Preserve agent identity when encoding actions; aggregate both players' actions through cross-attention before updating the shared world state. |
| Imagination | Advance the shared state autonomously and generate both local observations, team reward, and episode continuation from that imagined world. |
| Privileged information | Permit engine variables during training. Use auxiliary state/event prediction targets as the initial proposed design; determine any direct encoder inputs in the detailed plan and apply the same choice to B and D. |
| Policy optimization | MAPPO inside imagination, with centralized critics and the appropriate teacher or local actor inputs. |
| Transfer in D | Soft action-distribution distillation into local recurrent students, followed by student-generated MAPPO refinement with decreasing teacher guidance. |
| Resources | Four H200 GPUs and four CPU cores, with sufficient time and manpower. Simulator throughput will be measured before setting collection concurrency. |

Action distillation is the recommended interpretation of the previously discussed global-to-local transfer. Reconstructing the complete global token array is not required by this proposal.

The world model and teacher operate during training. The deployed artifact is the local actor pair. Actors may share weights, but each maintains its own memory, initialized and updated solely from its local history. Generated actor inputs must reproduce the local observation interface; an unrestricted projection of shared tokens would not meet this requirement.

### The two experimental variants

| | Variant B: direct local learning | Variant D: teacher-guided learning |
|---|---|---|
| Shared Transformer world model | Yes | Yes |
| Actor trained first | Local recurrent actors | Globally informed teacher actors |
| Teacher access to shared tokens | No teacher | Cross-attention to the shared representation |
| Distillation | None | Teacher action distributions supervise local students |
| Local MAPPO learning | Directly in imagination | In student-generated imagination after transfer, with guidance reduced to zero |
| Deployed agents | Local recurrent actor pair | Equivalent local recurrent student pair |

Only these two variants are included in the proposed comparison. Both use the same world-model architecture and training protocol. They begin from a common initial model checkpoint within each paired experimental seed. Subsequent checkpoints can diverge when the two policies collect different experiences.

## 3. Major implementation phases

### Phase 1 — Establish cooperative tasks and the evaluation interface

**Purpose:** Create a reliable two-player experimental setting with a clear local-information boundary.

Implement synchronized joint actions and observations, shared rewards, resets, episode termination, and consistent handling of player death. Record engine variables for permitted training supervision and diagnostics.

Propose a compact task suite with two complementary families: coordinated interaction/navigation, such as operating a mechanism while a teammate traverses a route; and cooperative combat/survival. Final scenarios should reward actual cooperation and remain solvable using the permitted local histories. Information asymmetry must leave useful local cues or opportunities to gather information.

Establish separate training, validation, and held-out test scenarios or seeds. Define success criteria, evaluation episodes, and interaction counting. Benchmark headless synchronized collection and CPU use before choosing the number of concurrent environments.

**Deliverable:** A working two-player environment interface, provisional task definitions, evaluation scaffolding, and a throughput report.

**Completion criterion:** Repeated episodes produce synchronized transitions and correct rewards/resets; each actor's input contains only permitted local information; the proposed tasks and collection setup are feasible.

### Phase 2 — Build the shared world-model foundation

**Purpose:** Provide the common imagination engine for B and D.

Collect an initial dataset using a documented exploration procedure and store synchronized joint transitions. Build the visual encoders, shared token representation, agent-labelled action aggregation, temporal dynamics, and observation/reward/continuation prediction components.

During model fitting, observations update the shared state. During imagination, the model advances from its current state and joint actions without future simulator observations or engine truth. Both players' predictions must come from the same sampled next world state, while retaining their different viewpoints.

Use privileged auxiliary targets to support learning and diagnose errors in positions, health, objects, or interaction events. These targets provide game-dynamics supervision; they do not automatically guarantee physical consistency.

Evaluate held-out prediction and free-running rollouts, including reward, termination, action responsiveness, and shared interaction outcomes. Use these checks to select a practical initial imagination horizon.

**Deliverable:** A trained initial shared model, synchronized replay pipeline, and model-quality diagnostics.

**Completion criterion:** The model can generate autonomous joint rollouts through the local actor interface, with sufficient fidelity to proceed to policy-learning pilots. The detailed plan will define acceptance thresholds.

### Phase 3 — Implement variant B: local MAPPO in imagination

**Purpose:** Establish the direct-learning baseline and reusable policy-training infrastructure.

Train two local recurrent actors inside the shared world model. Each actor receives its own generated observation and local memory. A centralized critic may use shared model state and the relevant actor memories.

Initialize actor memory from local-history burn-in. Freeze the world-model snapshot within each PPO collection/update cycle, and compute policy updates from the actions and behavior-policy probabilities that generated those rollouts.

Alternate real experience collection, world-model updates, and imagined policy learning using a documented schedule. Evaluate the local pair in the real simulator at predefined interaction checkpoints.

**Deliverable:** An end-to-end variant B implementation with checkpoints, learning curves, and repeatable real-environment evaluation.

**Completion criterion:** The full training cycle runs reliably, policy behavior responds to training, and model error can be distinguished from environment or policy-update failures.

### Phase 4 — Implement variant D: teacher, distillation, and local refinement

**Purpose:** Add the proposed teacher-guided training procedure to the common foundation.

First train teacher actors using imagined MAPPO. Each teacher can combine its local history with cross-attention to shared world tokens. This actor attention serves a different role from the action cross-attention used to update world dynamics.

Then initialize local recurrent students with the same deployment architecture and inputs as B. Warm-start them using the teachers' soft action distributions. Continue training on student-generated imagined trajectories, querying teacher guidance at states the students visit.

Combine local MAPPO learning with an auxiliary distillation objective. Reduce the guidance weight over training, including a final stage of local MAPPO with zero distillation weight. Keep model and teacher snapshots fixed within each student PPO collection/update cycle; PPO ratios must use the student's own behavior probabilities.

Refresh real experience and the world model using the same overall protocol as B. If additional real teacher experience is collected, include it in D's interaction budget. Evaluate the student pair throughout; teacher performance cannot substitute for deployment performance.

**Deliverable:** A complete variant D training pipeline and independently executable local students.

**Completion criterion:** Students execute without global inputs, learn from their own rollouts, and can be compared with B through the same evaluation interface.

### Phase 5 — Run the controlled B-versus-D study

**Purpose:** Test the hypothesis under a fair and reproducible comparison.

Use multiple independent training seeds, paired initial datasets/model checkpoints, common evaluation scenarios, and equal total real-interaction budgets. Freeze the experimental protocol after pilot validation and before final test evaluation.

Match deployed actor architecture, local inputs, rewards, world-model architecture, model-update schedules, centralized critic information, and local MAPPO update opportunities. Give both variants comparable tuning effort. Report D's additional teacher and distillation computation separately.

The primary study evaluates the complete training pipelines. When their policies generate different data, resulting differences in model checkpoints are part of that comparison; describe them explicitly rather than claiming that both runs always use an identical world model.

| Measure | Purpose |
|---|---|
| Primary: success-rate learning-curve area | Compare cooperation learned over a common real-interaction budget. Normalize by the budget and use the same evaluation checkpoints. |
| Final success rate and team return | Assess the quality of the deployed local pair at the final budget. |
| Held-out scenario performance | Assess transfer to the predefined test distribution. |
| Teacher–student performance gap | Diagnose loss of performance when global actor information is removed. |
| World-model rollout and reward errors | Help explain failures or differences between pipelines. |
| GPU/CPU time and elapsed time | Quantify the cost of teacher guidance and the practical effect of limited CPU resources. |

Count real data in **joint environment transitions**, documenting frame skip and underlying simulator frames. Include initial collection and all later teacher, student, or exploration collection; reused replay is counted once. Use the same counting convention for both variants. Keep evaluation interactions separate and report their cost.

Predefine a common learning-curve origin and checkpoint schedule that includes the initial collection/training period. Do not reset D's interaction counter when its student becomes ready. Evaluate the available local actor at the scheduled checkpoints, and account for teacher warm-up in the resource report.

Report uncertainty across independent training seeds and the paired D-minus-B difference. Individual evaluation episodes are repeated measurements within a training run, not independent training seeds. Select configurations on validation data and reserve the final test evaluation for frozen selections.

**Deliverable:** Matched learning curves, final performance tables, uncertainty estimates, and resource accounting.

**Completion criterion:** The data support a reproducible answer about whether D improves on B under the stated budgets and task distribution, including null or negative findings.

### Phase 6 — Interpret results and assemble research deliverables

**Purpose:** Turn the implementation and comparison into an evidence-based research contribution.

Analyze whether D's improvement, if observed, is consistent across tasks and seeds, whether it survives strict local execution, and what computation it requires. Examine failures involving missing local information, teacher-dependent coordination, or inaccurate imagination.

If D improves the primary outcome, the supported claim is that the complete teacher-guided procedure benefits direct local learning in this shared-world-model setting. If benefits occur only on some tasks, characterize those conditions. If D does not improve on B, report that outcome and its implications for the integrated design.

Package task definitions, configuration records, training/evaluation procedures, checkpoints, and representative rollouts. Draft the report around the tested research question and measured results.

**Deliverable:** A reproducible implementation package and research report documenting the B-versus-D findings and their limits.

## 4. Decisions for proposal review

The proposed defaults are action-distribution distillation, privileged auxiliary targets, a compact two-family cooperative task suite, and the B-versus-D experiment above. These are the main design choices to vet before the detailed plan.

After approval, the detailed plan will specify exact scenarios and rewards; image/token representations; model and actor sizes; permitted privileged variables; optimization objectives and schedules; rollout horizons; data budgets; seed counts; acceptance thresholds; and repository structure. Implementation scheduling will use milestones and measured throughput rather than an assumed fixed deadline.

## 5. Technical starting points

The existing project discussion provides the literature basis. These works inform implementation choices; the contribution here is the proposed integration and its controlled evaluation.

- [MARIE](https://arxiv.org/html/2406.15836v2): multi-agent Transformer world modelling, centralized aggregation, and MAPPO-like learning in imagination.
- [MATWM](https://arxiv.org/abs/2506.18537): Transformer world models for sample-efficient multi-agent reinforcement learning.
- [Scaffolder](https://arxiv.org/html/2405.14853): privileged world-model training for policies with restricted observation access; relevant to variant B.
- [PTDE](https://ijcai.org/proceedings/2024/4): global-to-local information distillation for decentralized execution.
- [Provable Partially Observable Reinforcement Learning with Privileged Information](https://arxiv.org/html/2412.00985): limitations of transferring privileged teacher behavior to locally informed students.
- [IRIS](https://arxiv.org/abs/2209.00588) and [TransDreamer](https://arxiv.org/abs/2202.09481): visual Transformer world-model design references.
- [Dreamer 4](https://arxiv.org/abs/2509.24527): further inspiration for learning in visual imagination; this project retains the agreed MAPPO optimizer and its own resource scale.

