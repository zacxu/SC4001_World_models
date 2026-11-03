# Research contract — Phase 0

The approved study asks whether globally informed teacher training, soft action
distillation, and local MAPPO refinement improve sample efficiency compared with
direct local MAPPO in the same shared Transformer world-model framework.

The primary hypothesis is a positive paired D-minus-B difference in normalized
team success-rate learning-curve AUC over the total real joint-transition budget.
This measures the complete teacher-guided procedure. It does not separately
identify benefits of imagination, Transformer dynamics, shared tokens, or the
distillation loss, and does not establish component synergy.

Only B and D are experimental arms. Relay and cover are separate task-family
training settings. Within each family/seed pair, initial real replay, world-model
weights, and local actor/critic initialization are paired. Each branch later
collects its own local-policy experience and updates its own model. Shared model
architecture does not imply identical evolving checkpoints.

Both arms have the same local actor capacity, local inputs, critic interface,
model architecture/update schedule, real budget, and local MAPPO schedule.
D adds teacher and supervised distillation computation. The comparison supports
sample-efficiency claims; report total computation and elapsed time separately.

Count initial scripted/random exploration and all later training collection.
Replay reuse counts once. Count a two-player joint transition once, not twice.
Track simulator tics independently. Evaluation data never enter learning replay
or normalization updates and their simulator cost is reported separately.

Pilot defaults: two seeds/family, 200,000 transitions/run, eight total B/D runs,
220 local PPO cycles. Final defaults: ten paired seeds/family, 1,000,000
transitions/run, 40 runs, 860 local PPO cycles. These are planned runs, not results.
The final grid has 12 checkpoints; D's teacher warm-up never resets the sample
axis. Use paired training-seed uncertainty and equal task-family weighting.

Configuration selection uses validation layouts/seeds. Freeze the complete
protocol before final experiments, then evaluate saved policies on the untouched
test bank. Negative, inconclusive, diverged, or missing outcomes remain visible;
never replace a seed merely because it performs poorly.

All numeric defaults remain provisional until pilots. Exact invariants and
derived budgets are checked by `world_models.infrastructure.config`. Configuration
hashes and source hashes identify the settings actually used.
