# Can Globally Informed Teachers Improve Decentralized Visual Cooperation through Shared Imagination?

**Working manuscript — Sections 1–3. Updated 4 November 2026.**

> **Draft status.** This manuscript follows the [paper-writing plan](research_paper_plan.md) and [research specification](../detailed_research_implementation_plan.md). The environment, initial pilot replay, world-model architecture, and learning objectives are implemented through P2.3. The model remains untrained. Model fitting, policy optimization, teacher transfer, and their numerical schedules below describe the proposed procedure and provisional pilot settings. No empirical advantage is claimed. The [architecture](../docs/phase22_architecture.md) and [objective specification](../docs/phase23_objectives.md) document the implemented components.

## 1. Introduction

Cooperative agents often act from different views of the same environment. One player may observe an interaction mechanism while another encounters the route or threat affected by it. Effective behavior therefore depends on relating local visual evidence to past actions and remembering information that is no longer visible. Training can draw on both agents' observations, but a deployed actor may have access only to its own history. The challenge is to use richer training information to improve learning without making execution depend on information unavailable to the actor.

Learned world models provide a way to reuse real experience through imagined interaction. Dreamer demonstrated behavior learning inside latent visual dynamics, while Transformer-based approaches and multi-agent world models extended the representations and settings in which imagination can support policy optimization. MARIE and MATWM show that Transformer world modelling and multi-agent imagined learning already form an established research direction. Their success motivates studying how information available within an imagination framework should be used to train decentralized policies. [Hafner et al., 2020](https://arxiv.org/abs/1912.01603), [Zhang et al., 2024/2025](https://arxiv.org/html/2406.15836v2), [Deihim et al., 2025](https://arxiv.org/abs/2506.18537).

One possibility is to train a teacher that can use a joint representation, then transfer its action preferences into local students. Privileged training and centralized-to-local distillation have substantial precedents. However, a teacher may distinguish states that are indistinguishable from a student's local history, so strong teacher behavior does not guarantee effective local transfer. A teacher can also select coordinated joint behavior whose separate local approximations fail to preserve its coordination. These limitations make the performance of the deployed students the relevant endpoint. [Hu et al., 2024](https://arxiv.org/abs/2405.14853), [Tseng et al., 2022](https://papers.nips.cc/paper_files/paper/2022/hash/01d78b294d80491fecddea897cf03642-Abstract-Conference.html), [Cai et al., 2024](https://arxiv.org/abs/2412.00985), [Lee et al., 2026](https://arxiv.org/abs/2610.10087).

We investigate the following question: **Does globally informed teacher learning, followed by action-distribution distillation and local policy refinement, improve the real-interaction learning efficiency of decentralized visual cooperation over direct local learning in the same shared-world-model framework?** Our setting contains two cooperating ViZDoom players with a common reward and separate first-person observations. A shared stochastic Transformer model generates both local views from one selected joint state. The planned comparison uses direct local imagined MAPPO, denoted B, and a teacher-guided procedure, denoted D. D adds globally informed teacher learning, action distillation, and student-generated MAPPO with guidance reduced to zero. Both procedures deploy the same local recurrent actor architecture.

The primary hypothesis is that D achieves greater area under the real-environment team-success learning curve at matched real-interaction budgets. The comparison also matches the initial replay and model checkpoint, deployed actor capacity and initialization, local MAPPO opportunities, and model-update schedule. D receives additional teacher and distillation computation, which must be reported separately. Once the branches collect their own experience, their replay distributions and model weights may diverge; those changes belong to the complete training procedure being compared.

The study is organized around three contributions. First, we specify a teacher-guided training procedure with an explicit boundary between joint training information and local execution. Second, we define a paired comparison against direct local learning that tests the complete procedure's sample-efficiency effect. Third, we construct cooperative visual tasks and auditable replay interfaces that support this comparison through synchronized observations, authoritative events, and explicit interaction accounting. The scientific contribution sought is a controlled answer to the teacher-guidance question, including its costs and failures. The design does not separately identify the benefits of Transformers, shared tokens, imagination, or distillation, and no favorable experimental outcome is assumed.

## 2. Review of existing literature

### 2.1 Visual world models and imagined learning

Dreamer learns visual control through a latent dynamics model and improves behavior using imagined trajectories. IRIS uses a discrete image autoencoder with an autoregressive Transformer, while TransDreamer investigates stochastic Transformer world models. Dreamer 4 further develops learning inside scalable visual world models. These methods establish imagination as a training substrate, but differ in representation, dynamics, and behavior optimization. Our procedure uses categorical latent-state dynamics with MAPPO; it is not an implementation of Dreamer 4 or its policy optimizer. [Hafner et al., 2020](https://arxiv.org/abs/1912.01603), [Micheli et al., 2023](https://arxiv.org/html/2209.00588v2), [Chen et al., 2022](https://arxiv.org/abs/2202.09481), [Hafner et al., 2025](https://arxiv.org/abs/2509.24527).

The distinction between visual features and stochastic state is relevant to our architecture. IRIS quantizes image features with a learned codebook and models the resulting discrete tokens. We instead extract spatial features with a CNN and infer categorical latent states through a learned prior and posterior. The complete model remains variational despite using an ordinary CNN encoder. This choice avoids a separate VQ image-token codebook, but the B/D comparison does not evaluate its advantage over VQ-based alternatives. [Micheli et al., 2023](https://arxiv.org/html/2209.00588v2).

### 2.2 Multi-agent world modelling

MARIE combines decentralized autoregressive Transformer dynamics with centralized aggregation of agent representations and policy learning from imagined local observations. MATWM combines decentralized imagination, teammate prediction, and a semi-centralized critic, and supports both visual and vector observations. These are close precedents for Transformer-based multi-agent imagination. Our common model instead advances a shared stochastic state using both actions and decodes the players' distinct views from that state. This construction supplies the setting for the teacher-guidance comparison rather than establishing novelty through Transformer dynamics alone. [Zhang et al., 2024/2025](https://arxiv.org/html/2406.15836v2), [Deihim et al., 2025](https://arxiv.org/html/2506.18537v1).

Joint visual simulation also has direct precedents. Solaris studies multiplayer world modelling in Minecraft, and Gamma-World develops multi-agent visual generation with hub-mediated interaction and distillation of a world-model teacher into a causal student. These works demonstrate why joint views, shared interaction tokens, and teacher–student terminology alone cannot distinguish our contribution. Our teacher supervises policy actions, and the focal outcome is real-interaction learning efficiency of the deployed local pair. [Savva et al., 2026](https://arxiv.org/abs/2602.22208), [Liu et al., 2026](https://arxiv.org/abs/2605.28816).

### 2.3 Privileged training and policy transfer

Scaffolder uses privileged training components to support a policy restricted to deployment observations. PTDE supplies personalized global information during training and distills it into locally predicted features. Offline Multi-Agent Reinforcement Learning with Knowledge Distillation trains a centralized Transformer teacher and transfers both features and inter-agent structural relations into decentralized students. Policy distillation more broadly transfers an expert's action preferences through supervised learning. Together these works establish training privilege and centralized-to-local transfer as existing mechanisms. [Hu et al., 2024](https://arxiv.org/html/2405.14853v1), [Chen et al., 2024](https://ijcai.org/proceedings/2024/4), [Tseng et al., 2022](https://papers.nips.cc/paper_files/paper/2022/hash/01d78b294d80491fecddea897cf03642-Abstract-Conference.html), [Rusu et al., 2015](https://arxiv.org/abs/1511.06295).

Our proposed transfer target is the teacher's soft action distribution. Students subsequently generate their own imagined trajectories and receive local MAPPO updates with decreasing teacher guidance. This differs from reconstructing personalized global features or transferring a world model itself. It is a specific design to evaluate, rather than a claim that action transfer is generally superior to representation transfer.

Privileged-information theory shows that partial observability can prevent teacher distillation from recovering a strong locally informed policy without additional assumptions. Work on support-preserving multi-agent distillation also identifies coordination-mode failures in centralized-to-decentralized transfer. These results motivate examining local performance and teacher–student gaps rather than treating imitation accuracy as sufficient evidence of useful cooperation. Our procedure does not inherit their guarantees or eliminate all information restrictions. [Cai et al., 2024](https://arxiv.org/abs/2412.00985), [Lee et al., 2026](https://arxiv.org/abs/2610.10087).

### 2.4 Position of the present study

| Closest approach | Relevant mechanism | Relationship to our research question |
|---|---|---|
| MARIE | Local autoregressive world models, centralized aggregation, imagined multi-agent learning | Establishes Transformer MARL imagination; motivates the common model-based training foundation |
| MATWM | Decentralized imagination and teammate prediction | Establishes visual/vector multi-agent world modelling; differs from our shared joint-state construction |
| Scaffolder | Privileged model/critic and restricted target policy | Establishes that training privilege can support restricted execution; B is related but is not a reproduction |
| PTDE | Personalized global-feature transfer to local features | Distinguishes feature distillation from our proposed action-distribution transfer |
| Offline MARL with knowledge distillation | Centralized Transformer teacher, feature and relational transfer | Establishes centralized-to-local Transformer distillation; differs from our online data refresh and imagined MAPPO |
| Proposed B/D study | Common shared visual model; D adds teacher learning, action transfer, and local refinement | Tests the complete added training procedure under matched real budgets and local learning opportunities |

We use MAPPO as an established cooperative policy optimizer. The planned empirical analysis will distinguish sample efficiency from computation and use paired training-run uncertainty, consistent with the need for reliable statistical reporting in deep RL. The contribution is therefore framed as a controlled investigation of training-time joint information, with its scope determined by the actual experiments. [Yu et al., 2022](https://arxiv.org/html/2103.01955v4), [Agarwal et al., 2021](https://arxiv.org/abs/2108.13264).

## 3. Methodology

The world-model architecture and objectives are implemented but untrained. Model fitting and the policy-learning procedures below remain proposed.

### 3.1 Setting and information constraints

Two cooperating ViZDoom players receive separate 128×96 RGB observations and a shared team reward. Each chooses among 12 actions, normally repeated for four simulator tics. Relay navigation requires plate–gate–latch coordination before joint extraction; cover-and-activate requires two exposed console activations and joint extraction. The latter permits either player to operate either console and allows finite enemies to be cleared before activation.

For local history $\tau_t^i=(o_{0:t}^i,a_{0:t-1}^i)$, the deployed policy factorizes as

$$
\pi_\theta(a_t^1,a_t^2\mid\tau_t^1,\tau_t^2)
=\pi_\theta(a_t^1\mid\tau_t^1)\pi_\theta(a_t^2\mid\tau_t^2).
$$

Actors share parameters, maintain separate recurrent memories, and sample actions independently given their histories. Each receives only its own image, previous action, and memory; global features, actor identity, and reward are excluded as actor inputs. There is no explicit communication channel. Both visual histories may condition the training-time world model, while privileged engine variables supply auxiliary targets only. The teacher's global context is therefore inferred from joint visual experience, rather than supplied as true simulator state.

### 3.2 Shared stochastic dynamics

The recurrent world state is $x_t=(M_t,z_t,j_t)$: deterministic memory tokens, categorical stochastic variables, and the exact episode clock. A shared CNN encodes the two views. Identity-labelled joint-action embeddings $A_t=[e(a_t^1,1),e(a_t^2,2)]$ are aggregated by cross-attention before a Transformer updates the memory:

$$
M_{t+1}=F_\phi(M_t,z_t,A_t,j_t),\qquad
j_{t+1}=j_t+1,\qquad
z_{t+1}\sim p_\phi(\cdot\mid M_{t+1}).
$$

During real conditioning, a posterior $q_\phi(z_{t+1}\mid M_{t+1},o_{t+1}^1,o_{t+1}^2,j_{t+1})$ incorporates both destination views. The transition uses the source clock; posterior conditioning uses the destination clock. The decoder/teacher representation $Z_t$ combines memory tokens and embeddings of $z_t$. View-specific queries decode both local images from the same selected joint state. Autonomous imagination uses the prior without future real observations. This couples the generated views without guaranteeing learned physical consistency.

Transition heads use source state, joint action, and destination state to predict a milestone bonus and one shared outcome: continue, success, or non-deadline failure. The sampled outcome and exact clock jointly determine termination and terminal reward:

$$
\hat r_t=\hat b_t-0.001+R(\hat c_t,j_{t+1}),
$$

where $R$ is +1 for success, −1 for failure, and zero for continuation. A continue outcome becomes failure at the deadline; success at the deadline remains success. Accordingly, deadline-only replay transitions train the learned outcome as continue. This keeps the prescribed reward and termination rules coherent while leaving dynamics accuracy to empirical validation.

### 3.3 Real replay and model learning

Each task-family/seed pair starts from 50,000 shared real joint transitions, comprising 70% random and 30% scripted exploration. Scripts may use privileged state to collect mechanics coverage, but their actions are not actor-imitation targets. Episodes are deterministically assigned to fitting or model-diagnostic subsets using a 90%/10% rule. Diagnostic episodes are excluded from fitting, normalization, and imagination anchors. Synthetic policy trajectories remain separate from real model-training replay.

The implemented objective is

$$
\mathcal L_{WM}=100\mathcal L_{RGB}+\mathcal L_{bonus}
+\mathcal L_{outcome}+\mathcal L_{aux}
+0.5\mathcal L_{dyn}+0.1\mathcal L_{rep}.
$$

The terms comprise pixel MSE, milestone-bonus Huber loss, outcome cross-entropy, auxiliary state/event prediction, and balanced prior–posterior KL regularization. Dynamics KL stops gradients through the posterior; representation KL stops gradients through the prior, following established world-model objectives. [Hafner et al., 2023](https://arxiv.org/html/2301.04104v2).

Continuous auxiliary targets are normalized using fitting data only; supported fields contribute equally after their own valid-label reductions. Missing targets are masked. If either required destination image is unavailable, the destination remains prior-based and joint next-image/posterior-KL supervision is skipped, while valid reward, state, and event labels remain usable. Initial normalization statistics are frozen and shared within each B/D pair. The common imagination horizon will be selected using held-out autonomous-rollout diagnostics before policy experiments.

### 3.4 Variant B: direct local learning

B uses shared CNN–GRU actors with separate memories and encoder weights independent of the world model. A centralized critic receives pooled shared tokens, detached memories of both actors, and remaining task time, and predicts one team value.

Imagined trajectories begin at nonterminal fitting-replay anchors. The model state is reconstructed from the real episode prefix, while each actor's memory is reconstructed from its complete own observation/executed-action history. After the anchor, actors consume decoded local RGB through the same observation interface used in the simulator. They receive no direct latent-state input.

Recurrent MAPPO optimizes individual clipped policy surrogates using each actor's own stored behavior probabilities and a shared team advantage. The surrogates are averaged over valid player decisions; likelihood ratios are not multiplied across players. The critic loss is counted once per joint sample. True terminals bootstrap zero, while nonterminal imagination cutoffs retain a next-value bootstrap. [Yu et al., 2022](https://arxiv.org/html/2103.01955v4).

The world model remains fixed throughout each rollout/PPO cycle, and policy gradients do not propagate through its dynamics. PPO updates current actor/critic parameters against the stored behavior data. Current actor memories are recomputed from stored local histories, while behavior probabilities remain fixed. Real episodes supply model-training data and anchors; they do not receive additional direct PPO updates in this design.

### 3.5 Variant D: teacher-guided local learning

D first trains a globally informed teacher through imagined MAPPO. Its recurrent branch uses local observation/action history, while its action head cross-attends to $Z_t$. Global tokens do not enter teacher memory. Local students and their critic begin from the paired initialization used by B, without copying teacher parameters or privileged state.

Transfer has two stages. A supervised warm-start matches soft teacher action distributions on teacher-generated imagined trajectories. Students then generate their own trajectories and receive local MAPPO updates with auxiliary forward-KL guidance:

$$
\mathcal L_S=\mathcal L_{PPO}
+\lambda_k\mathbb E_{t,i}
D_{KL}\!\left[
\operatorname{sg}\!\left(\pi_T^i(\cdot\mid h_{T,t}^i,Z_t)\right)
\,\|\,\pi_S^i(\cdot\mid h_{S,t}^i)
\right].
$$

Here $\mathcal L_{PPO}$ is the student actor objective, including entropy regularization; the value objective is unchanged from B. During refinement, the teacher is queried on student-visited states and updates its local history using the students' executed actions. Teacher targets are recorded with each rollout and remain fixed through its PPO epochs. Teacher and world-model parameters stay fixed during warm-start and within each student rollout/PPO cycle. Student likelihood ratios use student behavior probabilities; teacher guidance is a separate supervised term.

Guidance decays over the $K$ local PPO cycles as $\lambda_k=\max(0,1-k/(0.75K))$, where $k$ counts completed cycles, leaving the final quarter unguided. The teacher is refreshed between training blocks while guidance is active. Warm-start training does not replace the local PPO opportunities provided to B. Teacher performance is diagnostic; matching it is not assumed under local information restrictions.

### 3.6 Matched training and scope of inference

B and D share their initial replay, world-model checkpoint, normalization statistics, and local actor/critic initialization. They use the same model architecture, auxiliary information, real-interaction budget, model-update schedule, critic input convention, and local MAPPO opportunities. Both alternate real collection by their local policies, branch-specific model updates, and imagined policy learning. Later replay and model weights may diverge as their policies visit different states.

A simultaneous two-player decision counts once. Shared initialization is charged fully to each arm's learning-curve axis, and evaluation data remain outside learning. D adds teacher and distillation computation, which is reported separately from real sample efficiency. The comparison measures the complete teacher-guided procedure, including downstream data-distribution changes, rather than individual component effects. Deployment retains only the local actor pair; trained exports must be audited for this information boundary. Detailed architecture, optimization settings, and implementation procedures will be provided in the appendix.
