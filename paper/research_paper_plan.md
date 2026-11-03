# Research paper writing plan

**Project:** Shared Imagination and Policy Distillation for Decentralized Visual Cooperation  
**Draft date:** 3 November 2026  
**Purpose:** Plan the manuscript and the evidence needed to support it. This document does not start experiments or change the approved implementation protocol.  
**Scope:** Variant B versus variant D only, using the seven requested paper sections.

The authoritative project specifications are the [approved proposal](../research_implementation_proposal.md), [detailed implementation plan](../detailed_research_implementation_plan.md), and [research contract](../docs/research_contract.md). This writing plan follows them. Numerical training and evaluation settings remain pilot defaults until the Phase 5 protocol freeze; the manuscript must describe the settings actually used.

No target conference has been selected. Prepare a venue-neutral empirical AI paper, then adapt its template, length, anonymity, disclosure, and supplementary-material requirements to the selected conference and year. Use the [NeurIPS paper checklist](https://nips.cc/public/guides/PaperChecklist) as a reproducibility reference. The section allocations below are editorial suggestions, not conference rules.

## Working title, abstract, and central argument

**Proposed neutral title:** *Can Globally Informed Teachers Improve Decentralized Visual Cooperation through Shared Imagination?*

**Central research question:** Does globally informed teacher training followed by action-distribution distillation and local MAPPO refinement improve real-environment learning efficiency over direct local MAPPO learning in the same shared-world-model framework?

**Primary hypothesis:** At matched total real joint-transition budgets, D achieves higher normalized area under the deployed local agents' team-success learning curve than B.

The scientific issue is whether information available during training can produce more efficient learning when that information is unavailable to the deployed actors. The shared world model supplies a controlled training setting for investigating that issue. The contribution must be expressed through the question, procedure, and measured answer, rather than the list of components used.

Place an unnumbered abstract before Section 1 in the eventual paper. Draft its motivation, question, and method now; complete the result and implication sentences only after confirmatory analysis. Include the primary effect estimate and uncertainty, task scope, and compute qualification. Use explicit `[RESULT PENDING]` placeholders in drafts; do not insert illustrative performance values. Finalize the title after the findings are known.

## Evidence available at the start of writing

| Project work | Verified current status | Appropriate manuscript use |
|---|---|---|
| Phase 0 and Phase 1 | Research contracts, synchronized two-player environment, tasks, splits, and environment validation completed | Setting, task definitions, reproducibility, and environment diagnostics |
| P2.1 | Four pilot stores: two families × seeds 0 and 1 × 50,000 joint transitions | Initial-data procedure and pilot data characterization |
| P2.2 | Shared stochastic Transformer architecture and interfaces implemented and verified; model untrained | Architecture description, without predictive-performance claims |
| P2.3 | Objectives, fitting-only normalizers, and forward/backward checks completed | Objective and normalization description, without learning claims |
| P2.4–P2.5 | Pending | Model fitting and autonomous-rollout quality evidence still required |
| Phases 3–4 | Pending | B policies, D teachers/students, distillation, and trained deployment audits still required |
| Phases 5–6 | Pending | Confirmatory results, interpretation, and final research release still required |

Status sources: [project README](../README.md), [P2.2 architecture](../docs/phase22_architecture.md), and [P2.3 measured report](../artifacts/phase23/README.md). Current checks performed zero optimizer updates and establish no learned-policy result or measured H200 training performance. Earlier phase reports are historical evidence, not a replacement for the latest status.

## 1. Introduction

**Purpose:** Make the problem, unresolved question, proposed approach, and contribution clear before describing implementation details. Allocate roughly 15% of the main narrative.

### 1.1 Build the argument in five paragraphs

1. **Problem:** Two cooperating agents must coordinate from separate visual histories. Each sees only part of the environment, while real interaction is costly. Explain the practical distinction between joint information available in training and local information available at execution.
2. **Established approaches:** Briefly introduce world-model imagination, centralized training with decentralized execution, and privileged teacher transfer. Acknowledge that each already has substantial prior work; Section 2 establishes the closest overlaps.
3. **Unresolved issue:** A teacher using joint information can recommend behavior that a local student cannot infer from its history. Better teacher performance therefore need not produce better local cooperation. Explain why this makes transfer a research question rather than an automatic benefit.
4. **Our approach:** Introduce one shared stochastic visual world model, B's direct local imagined MAPPO, and D's teacher learning, action distillation, and student-generated local refinement. State the strict local execution constraint and the complete-procedure comparison.
5. **Evidence and scope:** Describe the two task families and controlled paired study. Add the actual primary finding only when available, including uncertainty and additional compute. End with the contributions.

### 1.2 Planned contribution statements

- **Training procedure:** A specified integration of shared visual imagination, globally informed teacher learning, and action-distribution transfer into local recurrent actors, including student-generated refinement and eventual removal of guidance.
- **Controlled empirical answer:** A paired B/D study testing whether the complete teacher-guided procedure improves real-interaction learning efficiency while matching local actor capacity, local PPO opportunities, and the world-model training protocol.
- **Analysis and reproducibility:** Evidence about the teacher–student gap, model error, task-dependent outcomes, and resource cost, supported by reproducible tasks, configurations, accounting, and evaluation artifacts.

These are intended contributions. Claim a new algorithmic principle only if the final literature analysis and implementation support one. Do not claim invention of Transformers, MAPPO, shared representations, policy distillation, or multiplayer world modelling. Releasing task code is useful infrastructure; it does not by itself establish a broadly validated new benchmark.

**Section completion check:** A reader can identify the research question, treatment, comparator, deployed artifact, primary outcome, and claim boundary without reading the appendix.

## 2. Review of existing literature

**Purpose:** Establish what is already known and explain the precise question our experiment addresses. Organize by research issue rather than a chronological list. Allocate roughly 15% of the main narrative.

### 2.1 Organize the review into four themes

- **Visual world models and imagination:** Dreamer, IRIS, TransDreamer, and Dreamer 4. Explain how representations and behavior-learning procedures differ. Our model is a categorical variational recurrent Transformer without a VQ codebook; our policy optimizer is MAPPO.
- **Multi-agent world models:** MARIE and MATWM are core comparisons. Discuss whose observations/actions condition the dynamics, how joint consistency is represented, what actors receive, and how policies learn. Include Solaris and Gamma-World when discussing multiplayer visual generation; generating joint views is not, on its own, our novelty.
- **Privileged training and decentralized transfer:** Scaffolder, PTDE, policy distillation, and Offline Multi-Agent Reinforcement Learning with Knowledge Distillation. Distinguish privileged model/critic learning, latent or feature transfer, and action-distribution supervision.
- **Transfer limits and reliable evaluation:** Privileged-information theory and coordination-mode failures motivate why transfer can fail. Statistical RL evaluation motivates reporting run-level uncertainty and explicit resource budgets.

### 2.2 Prepare a closest-work comparison table

Use the following as a starting point; verify the full methods and experimental sections before writing definitive comparisons.

| Prior work | Important overlap | Distinction to establish in our manuscript |
|---|---|---|
| [MARIE](https://arxiv.org/abs/2406.15836) | Multi-agent Transformer world models, centralized aggregation, imagined policy learning | Its local autoregressive modelling/aggregation versus our one shared stochastic state; our focal experiment is teacher-guided versus direct local learning |
| [MATWM](https://arxiv.org/abs/2506.18537) | Transformer world modelling for sample-efficient MARL | Its local imagination/teammate modelling and behavior learning versus our joint imagined transition and matched B/D transfer study |
| [Scaffolder](https://arxiv.org/abs/2405.14853) | Training-time privileged world modelling for restricted policies | Privileged training already exists; our comparison concerns an added globally informed policy teacher and action transfer in two-player visual cooperation |
| [PTDE](https://ijcai.org/proceedings/2024/4) | Distillation of global information into decentralized execution | Personalized feature transfer versus our soft action supervision and subsequent student-generated imagined MAPPO |
| [Offline MARL with knowledge distillation](https://papers.nips.cc/paper_files/paper/2022/hash/01d78b294d80491fecddea897cf03642-Abstract-Conference.html) | Centralized Transformer teacher and decentralized students | Offline feature/relation distillation versus our online real-data/model cycle and complete-procedure sample-efficiency question |
| [Solaris](https://arxiv.org/abs/2602.22208) and [Gamma-World](https://arxiv.org/abs/2605.28816) | Joint multiplayer visual world modelling | Their visual simulation objectives differ from the deployed local-policy learning-efficiency endpoint; Gamma-World's teacher/student transfer concerns the world model |
| [Privileged-information analysis](https://arxiv.org/abs/2412.00985) and [support-preserving distillation](https://arxiv.org/abs/2610.10087) | Limits and failure modes of transferring centralized behavior | Motivate diagnostics and limitations; our categorical action distillation does not inherit their theoretical guarantees or solve all coordination-mode failures |

For the eventual manuscript, add columns for observation modality, teacher information, transfer target, real versus imagined training, execution information, and evaluation objective. Mark a feature “not established” when the source does not substantiate it; do not infer an absence from an abstract alone.

### 2.3 Literature review tasks and novelty decision

1. Read the closest papers in full and record the exact passages supporting each comparison, including any negative overlap with our proposed contribution.
2. Search recent work on multi-agent world models, asymmetric teacher/student RL, and decentralized distillation again before submission. The current focused review is a starting set, not an exhaustive priority search.
3. Write a short gap paragraph: existing world modelling and privileged transfer motivate studying whether this complete teacher-guided procedure improves strictly local cooperation under matched real-interaction budgets.
4. If a close paper already answers the same question under comparable conditions, position this work as a replication/extension or reassess the contribution with the project team. Do not manufacture a novelty claim by listing a different software stack.

**Scope:** Prior papers are literature comparisons. This plan does not add learned experimental arms beyond B and D or promise state-of-the-art superiority without matched experiments.

## 3. Methodology

**Purpose:** Define the method precisely enough to reproduce it and assess the information restrictions. Allocate roughly 30% of the main narrative; move exhaustive settings to the appendix.

### 3.1 Problem formulation and information access

Introduce a two-agent partially observable cooperative decision process with shared reward. Let local history be \(h_t^i=(o_{0:t}^i,a_{0:t-1}^i)\), and let the deployed joint policy factorize as

\[
\pi(a_t^1,a_t^2\mid h_t^1,h_t^2)
=\pi_\theta(a_t^1\mid h_t^1)\pi_\theta(a_t^2\mid h_t^2).
\]

The actors share parameters and retain separate recurrent memories. Use independent per-agent action-sampling streams. Parameter sharing does not provide online access to teammate history.

Include a compact information-access table:

| Component | Inputs or supervision |
|---|---|
| Deployed B actor / D student | Own RGB, own previous action, own recurrent memory |
| World-model posterior | Both synchronized visual histories, executed joint actions, known episode clock |
| Autonomous world-model prior | Current shared state, joint action, deterministically advanced clock |
| D teacher | Local recurrent history plus attention to the current shared world representation at the action head |
| Centralized critic | Shared state, relevant actor memories, and remaining time |
| Auxiliary training heads | Declared engine-state and event targets |

Engine variables are auxiliary targets only in the approved design; they are not encoder inputs. The teacher's joint context is inferred world-model information, not a direct engine-state oracle. Local actors receive no explicit communication, teammate features, global tokens, actor ID, or team reward as an input. In-world visual interaction remains possible.

### 3.2 Shared stochastic world model

Describe the implemented architecture from [P2.2](../docs/phase22_architecture.md): shared CNN visual encoding; agent-labelled joint-action cross-attention; recurrent deterministic memory; categorical stochastic latent; and two local-view reconstructions plus shared transition predictions.

Give the state/update abstraction, with indexing matched to the code:

\[
m_{t+1}=F_\phi(m_t,z_t,a_t^1,a_t^2,j_t),\qquad
z_{t+1}\sim p_\phi(z\mid m_{t+1}),
\]

with \(j_{t+1}=j_t+1\). The transition uses the source clock; the observation-conditioned posterior uses the destination clock when fitting real sequences. Explain that one selected joint latent supplies both views and all relevant predictions. This couples their generation; it does not guarantee learned physical consistency.

Report the current dimensions in an architecture table: 16 memory tokens, width 256, four Transformer blocks, eight attention heads, 32 categorical variables with 32 classes, and 128×96 RGB per player. Describe the exact-clock deadline rule and coherent outcome/reward decomposition. Keep current dimensions labelled as implementation settings, not claimed architectural innovations.

### 3.3 Model learning and data use

Present the implemented objective:

\[
L_{WM}=100L_{RGB}+L_{bonus}+L_{outcome}+L_{aux}
+0.5L_{dynKL}+0.1L_{repKL}.
\]

Define the KL stop-gradient directions, one-nat per-sample floor after summing latent variables, valid-label reductions, and equal averaging over supported auxiliary fields. State that normalization uses fitting episodes only. Explain terminal missing-view masks and why an external collection cutoff is not a learned task failure.

Document the initial 50,000-transition procedure per family/seed: 70% random and 30% scripted exploration. Scripts may access privileged training state for data collection; their actions are not actor imitation targets. Attribute all initialization interactions to each arm. Reserve diagnostic episodes from fitting and normalization; report their exact counts.

Add the actual optimizer, sequence/burn-in lengths, update schedule, replay sampling, and selected imagination horizon after P2.4–P2.5. Architecture and objective checks alone cannot establish usable dynamics.

### 3.4 Variant B: direct local learning

Explain how real prefixes initialize the shared model and each local recurrent memory. Roll out both local actors on their generated local RGB views. Optimize recurrent MAPPO with a centralized critic, behavior-policy probabilities, correct terminal/cutoff bootstrapping, and frozen world-model snapshots within collection/update cycles.

Local inputs are generated images, not unrestricted projections of global tokens. Alternate counted real collection, model fitting, and imagined policy updates according to the common schedule. Do not imply that this implementation is already completed.

### 3.5 Variant D: teacher, transfer, and refinement

Describe three stages: globally informed teacher MAPPO; supervised warm-start from soft action distributions; and local MAPPO on student-generated trajectories with decaying teacher guidance.

Use the intended student objective:

\[
L_S=L_{PPO}+\lambda_k\mathbb{E}_{t,i}
D_{KL}\!\left(\operatorname{stopgrad}[\pi_T^i]\;\|\;\pi_S^i\right).
\]

Specify that the teacher is queried on the student-visited shared state and the executed student action history. Its recurrent memory is local-history based; global context enters the action head. Teacher/model snapshots and saved soft targets remain fixed through the corresponding PPO updates. PPO ratios use the student's behavior probabilities.

Guidance reaches zero for the final quarter of local cycles under the current schedule. The deployed student retains B's architecture and local interface and requires no world model, teacher, or centralized critic. Claim this deployment property only after testing the actual exported trained actors.

### 3.6 State exactly what the comparison identifies

Show a B/D table of matched and additional resources. Match local actor capacity/initialization, initial replay/model, permitted auxiliary fields, real budget, local PPO opportunities, critic convention, and model-update schedule. D adds teacher and distillation computation.

After shared initialization, each branch collects its own experience and updates its own model. The study estimates the complete pipeline effect, including resulting data-distribution changes. It does not isolate distillation, shared tokens, imagination, or a Transformer contribution individually.

**Planned main visual:** Figure 1, a B/D training diagram with explicit training-only inputs and a small deployment panel. Include Algorithm 1, a concise outer loop showing common initialization, each branch's data/model/policy cycle, D's extra steps, and interaction accounting.

## 4. Experiments and results

**Purpose:** Answer the primary question, establish that the comparison is valid, and characterize costs and failures. Allocate roughly 30% of the main narrative. Every result below is planned unless linked to completed evidence.

### 4.1 Experimental setting

Describe gate-and-relay navigation and cover-and-activate combat with representative local screenshots and simple task diagrams. Include rewards, deadlines, console dwell, action set, frame skip, death/reset rules, and local cues.

Report six training, two validation, and two test geometry templates per family. Holdouts change lengths, gate positions, or pillar arrangements within the authored topologies; they do not establish generalization to arbitrary Doom maps. In cover, finite moving enemies may be cleared before console operation. Do not describe simultaneous suppression or mandatory role swapping as requirements.

Summarize Phase 1 mechanics evidence as environment validation. Scripted completion rates and fire/no-fire diagnostics are not learned-policy results or additional research variants.

### 4.2 Confirmatory design and fairness

| Protocol item | Current planned specification |
|---|---|
| Experimental arms | B and D only |
| Independent training runs | 2 families × 10 paired seeds × 2 arms = 40 runs |
| Seeds | Pilot 0/1; confirmatory 1001–1010, with derived component streams |
| Training budget | 1,000,000 real joint transitions per run, including 50,000 initialization |
| Curve checkpoints | 0, 50k, 100k, then every 100k through 1M: 12 points |
| Primary evaluation | Fixed 50-episode held-out bank per family at each checkpoint |
| Secondary final evaluation | 200 episodes, including the primary 50-episode bank |
| Evaluation policy | Sampled local policies at temperature 1.0; paired case and per-agent seed lists |
| Local learning opportunities | Matched B/D local MAPPO schedule; D's extra training compute reported |
| Selection | Validation-only pilots; freeze protocol before final seed runs; final scheduled checkpoint |

Count a simultaneous two-player decision once. Include scripted initialization and all later training collection. Count physically shared initialization once in total collection cost while assigning its full cost to each arm's learning-curve axis. Report simulator tics, evaluation interactions, development data, retries, and compute separately. Do not reset D's sample counter after teacher preparation.

Keep test data and scores out of fitting, normalization, tuning, and stopping decisions. Record all pilot candidates, unsuccessful runs, and deviations. The 40 runs are a planned campaign, not current completed experiments.

### 4.3 Primary endpoint and statistical analysis

For family \(f\), arm \(v\), seed \(s\), checkpoint \(n_j\), and team success fraction \(p_{fvsj}\), compute

\[
A_{fvs}=\frac{1}{N}\sum_j(n_{j+1}-n_j)
\frac{p_{fvsj}+p_{fvs,j+1}}{2},\qquad N=10^6.
\]

Form paired differences \(d_{fs}=A_{fDs}-A_{fBs}\). Average within each family and weight the two family means equally. Report the effect, a 95% interval from 10,000 paired bootstrap resamples within each family, family-specific estimates, and every seed's result. Training seeds are the independent units; evaluation episodes are repeated measurements within runs.

Use the 50-episode estimate at every AUC checkpoint, including the final one. Do not substitute the more precise secondary 200-episode estimate into the primary curve. The planned 0.02 normalized-AUC practical reference must be frozen before final runs. A positive 95% paired interval excluding zero supports the primary hypothesis. An interval crossing zero is inconclusive, not proof of equivalence. Uncertainty concerns these fixed task families/test banks and training randomness.

Incomplete or divergent runs must remain visible under the predefined failure policy. Never report only successful seed pairs. Describe sensitivity analyses for missing outcomes where required, and label an incomplete campaign accordingly. This reporting approach is informed by [Agarwal et al.](https://arxiv.org/abs/2108.13264).

### 4.4 Results presentation order

1. **Primary answer:** Per-family B/D success curves and paired AUC differences. State the magnitude and uncertainty before offering an explanation.
2. **Final local performance:** Success, return, successful completion time, death/deadline frequencies, and layout summaries. Explain any difference between earlier learning and final quality.
3. **Model-quality evidence:** Held-out one-step and free-running errors at horizons 1/5/10/15, including rewards, outcomes, events, views, and shared interaction consistency. Separate common diagnostic banks from branch-specific holdouts. Persistence and constant-frequency predictors are diagnostic baselines, not new learned policy arms.
4. **Transfer and failure diagnostics:** Teacher/student performance gap, student distillation KL, behavior after guidance reaches zero, and representative success/failure traces. Low KL is not itself evidence of good cooperation. Any association between these diagnostics and performance is descriptive, not a component-level causal result.
5. **Resource cost:** Real and imagined volumes, model/local/teacher/distillation updates, GPU-hours, CPU-hours, active/elapsed time, peak memory, and storage. Report pilot and failed-run cost as well as confirmatory cost.

### 4.5 Figures, tables, and evidence to prepare

| Item | Main message | Evidence required |
|---|---|---|
| Figure 1: B/D method and deployment | What changes during training; what local actors receive | Implemented data flow and trained export audit |
| Figure 2: two task families | Why coordination and partial observations matter | Authored map diagrams and authentic local frames |
| Figure 3: success learning curves | Learning efficiency over the same real budget | All scheduled confirmatory local-pair evaluations |
| Figure 4: paired AUC effects | Effect magnitude, uncertainty, and seed variation | Run-level AUCs and frozen analysis |
| Table 1: closest literature | Precise relationship to earlier methods | Full-text literature comparison |
| Table 2: B/D controls and protocol | Validity and scope of comparison | Frozen manifest and implementation audits |
| Table 3: primary and final outcomes | Main quantitative answer | AUC and final evaluation records |
| Table 4: compute and interaction accounting | Price paid for any sample-efficiency change | Measured resource and interaction logs |
| Appendix figures | Model rollouts, teacher/student traces, failure categories | Fixed selection rule and linked episode records |

Combine figures/tables if the target venue is short. Generate numerical tables and plots from saved analysis outputs; retain links to source records. Keep all performance cells `[PENDING]` until populated from actual runs. Add source/sampling descriptions to captions; select qualitative episodes by a declared rule rather than only showing favorable examples.

## 5. Discussion

**Purpose:** Explain what the evidence supports, what remains uncertain, and where the approach applies. Allocate roughly 10% of the main narrative, including a short concluding paragraph.

### 5.1 Interpret the outcome without changing the question

| Observed result | Defensible interpretation |
|---|---|
| D improves AUC and final local success | Complete teacher-guided procedure benefits both learning efficiency and final performance in the tested setting |
| D improves AUC but final outcomes converge | Useful cooperation is learned earlier under the tested budget |
| Only final success improves | A late benefit; the primary AUC hypothesis still needs its own evidence |
| Benefits appear in one family only | Task-dependent result that requires per-family explanation |
| Primary interval crosses zero | Evidence is inconclusive at the achieved precision; report effect size and uncertainty |
| D reliably underperforms | Additional guidance can be costly or harmful here; diagnose transfer and model failures without discarding the finding |

Discuss information mismatch, ambiguous local histories, multimodal coordination, teacher dependence, world-model error, and branch-specific replay as candidate explanations. Teacher scores are diagnostic references, not guaranteed upper bounds. Do not assert that local refinement solves information-theoretic restrictions.

### 5.2 Limitations that must remain explicit

- Only two cooperating agents, two authored task families, and narrow geometry holdouts; no evidence for arbitrary games, more agents, or other task topologies.
- B/D identifies the full teacher-guided procedure. No approved component ablations or model-free comparisons establish isolated benefits or broad superiority.
- Equal real-interaction budgets do not equal compute budgets. Any efficiency conclusion must identify which resource improves.
- Scripted, privileged initialization is part of the training-data procedure. Its usefulness and cost cannot be hidden behind a “pixels only” label.
- Current pilot replay has sparse milestones and only five fitting non-deadline failure transitions. Assess event coverage and calibrated prediction rather than aggregate accuracy alone.
- A common latent sample does not enforce correct physics, and decentralized policy distillation cannot recover unavailable information automatically.
- Final actor locality requires auditing trained exports; a generic interface probe and unit tests do not establish it alone.

### 5.3 Publication assessment and conclusion

A useful combination of known components can motivate research, but implementation alone does not establish a strong conference contribution. The paper needs a clear distinction from the closest work, a rigorous primary result, and an explanation of when training-time joint information helps or fails. B/D can answer the approved complete-procedure question; it cannot substantiate every component-level claim a reviewer might request.

After results, assess whether the evidence supports a conference submission, a narrower empirical report, or further research. Any additional baselines or ablations require a separate project decision; this plan does not silently expand the approved study. End with a brief answer to the research question, the principal limitation, and a concrete future direction. Avoid presenting planned work as an achieved contribution.

## 6. References

**Purpose:** Maintain an accurate, relevant bibliography and attach each comparison to a primary source. The following is a seed reading list, not a claim that all papers have been reproduced or exhaustively reviewed.

### 6.1 Initial reference list

1. Zhang et al. *Decentralized Transformers with Centralized Aggregation are Sample-Efficient Multi-Agent World Models* (MARIE; 2024 preprint, revised 2025). [Primary source](https://arxiv.org/abs/2406.15836).
2. Deihim, Alonso, and Apostolopoulou. *Transformer World Model for Sample Efficient Multi-Agent Reinforcement Learning* (MATWM; 2025). [Primary source](https://arxiv.org/abs/2506.18537).
3. Hu et al. *Privileged Sensing Scaffolds Reinforcement Learning* (Scaffolder; 2024). [Primary source](https://arxiv.org/abs/2405.14853).
4. Chen et al. *Personalized Training with Distilled Execution for Multi-Agent Reinforcement Learning* (PTDE; IJCAI 2024). [Proceedings](https://ijcai.org/proceedings/2024/4).
5. Tseng et al. *Offline Multi-Agent Reinforcement Learning with Knowledge Distillation* (NeurIPS 2022). [Proceedings](https://papers.nips.cc/paper_files/paper/2022/hash/01d78b294d80491fecddea897cf03642-Abstract-Conference.html).
6. Cai et al. *Provable Partially Observable Reinforcement Learning with Privileged Information* (2024). [Primary source](https://arxiv.org/abs/2412.00985).
7. Lee et al. *Multi-Agent Coordination via Support-Preserving Distillation* (2026). [Primary source](https://arxiv.org/abs/2610.10087).
8. Micheli, Alonso, and Fleuret. *Transformers are Sample-Efficient World Models* (IRIS; 2022 preprint / ICLR 2023). [Primary source](https://arxiv.org/abs/2209.00588).
9. Chen et al. *TransDreamer: Reinforcement Learning with Transformer World Models* (2022 preprint). [Primary source](https://arxiv.org/abs/2202.09481).
10. Hafner et al. *Dream to Control: Learning Behaviors by Latent Imagination* (2019 preprint / ICLR 2020). [Primary source](https://arxiv.org/abs/1912.01603).
11. Hafner, Yan, and Lillicrap. *Training Agents Inside of Scalable World Models* (Dreamer 4; 2025). [Primary source](https://arxiv.org/abs/2509.24527).
12. Yu et al. *The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games* (2021 preprint / NeurIPS 2022). [Paper](https://arxiv.org/abs/2103.01955); [official implementation](https://github.com/marlbenchmark/on-policy).
13. Rusu et al. *Policy Distillation* (2015 preprint). [Primary source](https://arxiv.org/abs/1511.06295).
14. Savva et al. *Solaris: Building a Multiplayer Video World Model in Minecraft* (2026). [Primary source](https://arxiv.org/abs/2602.22208).
15. Liu et al. *Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players* (2026). [Primary source](https://arxiv.org/abs/2605.28816).
16. Kempka et al. *ViZDoom: A Doom-based AI Research Platform for Visual Reinforcement Learning* (2016). [Paper](https://arxiv.org/abs/1605.02097); [official platform](https://github.com/Farama-Foundation/ViZDoom).
17. Agarwal et al. *Deep Reinforcement Learning at the Edge of the Statistical Precipice* (2021). [Primary source](https://arxiv.org/abs/2108.13264).

### 6.2 Resolve the originally supplied references

The original [preliminary note](../preliminary_plan.md) also names [AOAD-MAT](https://arxiv.org/abs/2510.13343), *Transformer-based multi-agent deep reinforcement learning model considering agents' order of action decisions*. Read it as actor/decision-order context; do not describe it as a world model or imply our simultaneous joint transition learns an agent ordering.

The original [OpenReview reference sVrzVAL90sA](https://openreview.net/forum?id=sVrzVAL90sA) could not be identified reliably through the accessible page during this review. Recover its title/authors and PDF from the original materials or an accessible primary record before citing or comparing it. Keep it as an unresolved reading-list item rather than guessing its identity.

### 6.3 Bibliography workflow

Create `references.bib` when manuscript drafting begins. Export metadata from the paper or publisher, preserve full author names, identify the exact version read, and distinguish preprint dates from publication dates. Check every citation against the claim it supports. Refresh recent literature before making priority claims. Do not use “first” or “state of the art” on the strength of this seed list.

## 7. Appendix

**Purpose:** Supply enough detail to reproduce and audit the work without burying the main scientific argument. The main paper must still contain the central hypothesis, information contract, B/D treatment, primary endpoint, uncertainty, and compute qualification.

### 7.1 Planned technical appendix contents

| Appendix | Material to prepare | Existing source or dependency |
|---|---|---|
| A. Tasks and local interface | Geometry, cues, action mapping, rewards, deadlines, mechanics, splits, and exact generalization scope | Phase 1 sources; [split contract](../docs/phase1_splits.md) |
| B. Initial replay and provenance | Random/scripted procedure, episode partitions, schema, validity masks, event counts, immutable identities, interaction ledger | [P2.1 runbook](../docs/phase21_replay.md) and sealed pilot stores |
| C. Architecture and objectives | Shapes, categorical sampling, KL directions/reductions, decoder/heads, masks, normalizers, reward/deadline rules | [P2.2](../docs/phase22_architecture.md) and [P2.3](../docs/phase23_objectives.md) |
| D. Model learning and imagination | Optimizer, burn-in approximation, prefix reconstruction, model versions, horizons, quality diagnostics, leakage checks | P2.4–P2.5 and Phase 3, pending |
| E. MAPPO and distillation | Recurrent buffers, actor/critic inputs, PPO/GAE details, teacher history, KL targets, guidance schedule, pseudocode | Phases 3–4, pending |
| F. Full experimental protocol | Frozen manifest, seed streams, checkpoint/evaluation banks, tuning candidates, failure/recovery rules, export-locality tests | Phase 5, pending |
| G. Complete results | Every run/checkpoint, seed-level metrics, family/layout breakdowns, failed runs, uncertainty procedure, qualitative selection rule | Phase 5 analysis, pending |
| H. Resources and reproduction | Measured hardware/software, CUDA/driver, actual commands, time/memory/storage, dataset access, licenses, artifact hashes | [Linux setup guide](../linux_gpu_server_setup_through_p23.md) plus final measured run records |

The current transfer guide reproduces the project through P2.3; it is not a final training/reproduction guide for unpublished later phases. Archive dataset/checkpoint access instructions and licensing information before claiming an open reproducible release. Check the selected venue's requirements for an impact statement, author disclosures, acknowledgements, and reproducibility checklist.

### 7.2 Writing sequence aligned with implementation phases

| Writing milestone | When | Concrete output and completion condition |
|---|---|---|
| W1. Establish the story | Now | Draft Introduction and literature comparison; articulate the single question and claim limits; resolve outstanding source identities |
| W2. Describe the implemented foundation | Now | Draft task, replay, architecture, and objective subsections using actual code/docs; clearly mark unimplemented training/policy sections |
| W3. Complete model evidence | After P2.4–P2.5 | Add actual fitting recipe, operating horizon, trained prediction diagnostics, and qualitative autonomous rollouts |
| W4. Complete policy method | During Phases 3–4 | Verify B/D pseudocode against code; document teacher/student schedules and audit trained local exports |
| W5. Lock the analysis | Before confirmatory Phase 5 | Freeze metric definitions, tables/figures, test-bank handling, uncertainty, and failure policy with the protocol; no claimed outcomes |
| W6. Write results and discussion | After the complete confirmatory campaign | Generate all metrics/plots from records; report primary result, costs, exceptions, and negative findings; align Discussion with evidence |
| W7. Finalize submission | Phase 6 | Write final abstract/title/contributions, refresh literature, run a claim audit, complete reproducibility material, and apply the selected venue's current rules |

### 7.3 Claim and evidence audit before submission

Maintain a small ledger linking every numerical or comparative manuscript claim to its run IDs, metric definition, analysis output, and evidence location. Keep “implemented,” “validated interface,” “trained model,” “pilot observation,” and “confirmatory finding” distinct.

- [ ] Every claimed gain comes from the specified deployed local pair on the appropriate held-out evaluations.
- [ ] No scripted collector outcome, untrained loss, unit-test count, or teacher score is presented as learned local-policy performance.
- [ ] The primary AUC, checkpoint grid, paired uncertainty, and missing-run handling match the frozen protocol.
- [ ] Family-level results and individual seeds accompany aggregate conclusions.
- [ ] All matched resources and D's additional compute are disclosed; later branch models are not described as identical.
- [ ] Methods match the final code and configurations, with pilot changes and deviations identified.
- [ ] Actual exported B/D actors pass the local-information audit and run without the training stack.
- [ ] Every plot/table has reproducible input records and an analysis path; placeholders and fabricated examples are absent from final results.
- [ ] Literature comparisons are supported by primary sources, and the unresolved original reference has been resolved or explicitly omitted.
- [ ] Limitations, task scope, licenses, reproduction instructions, and venue-specific requirements are complete.

**Ready-to-draft now:** Introduction, literature structure, implemented setting/model/objective descriptions, and empty result/table specifications. **Ready-to-submit later:** only after the remaining implementation, confirmatory experiments, and evidence audit substantiate the manuscript's claims.
