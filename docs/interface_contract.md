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
