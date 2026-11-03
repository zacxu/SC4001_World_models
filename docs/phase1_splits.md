# Phase 1 split and trajectory provenance contract

`configs/environment_splits.yaml` is the versioned source of the formal layout
allocation. `world_models.env.splits.split_manifest()` publishes its canonical
SHA-256, complete nuisance banks, realized geometry hashes and static mechanic
checks. Each family has **six training, two validation and two test templates**.
All ten realized geometry hashes per family are distinct. Templates are fixed
geometry definitions: changing the randomization seed changes nuisance draws,
not which geometry belongs to which split.

The initial distribution is deliberately narrow:

| Family | Training geometry | Validation geometry | Test geometry |
| --- | --- | --- | --- |
| Relay | Route end x=768 through 1088 in 64-unit increments; gate x=256/320/384; template-specific obstacles | Route ends 1152/1216; gates 448/512; new obstacle configurations | Route ends 1280/1344; gates 576/640; new obstacle configurations |
| Cover | Arena widths 960 through 1280; six pillar arrangements | Widths 1344/1408; two new pillar arrangements | Widths 1472/1536; two new pillar arrangements |

Relay retains its two separated routes, plate dependency and latch. Cover
retains the two consoles, two finite moving enemies, ordinary combat and joint
extraction. The declared deadlines, rewards and dwell interval remain common.
This test set measures generalization to held-out lengths/gate locations/pillar
arrangements **within these authored task topologies**. It does not establish
generalization to arbitrary Doom maps, different task topologies, new weapons or
new enemy classes. The longer validation/test layouts introduce an explicit
geometry extrapolation; they are not interchangeable samples of training
geometry.

## Versioned nuisance banks

The same family-specific nuisance bank is shared across splits so the geometry
holdout is explicit rather than confounded with a separately unseen texture or
enemy-class distribution. Both B and D select the same template and bank for a
paired evaluation case.

- `relay-randomization-v1`: three role-specific spawn pairs, seeded role swapping,
  three wall textures, and no enemies.
- `cover-randomization-v1`: three role-specific spawn pairs, seeded role swapping,
  three wall textures, and three two-enemy spawn-position pairs. Enemy counts,
  classes, stages, health and combat rules are not randomized by these banks.

Role/spawn/texture/enemy choices use separate deterministic hash-derived streams
from `(bank_version, family, component, seed)`. Template ID is omitted from these
nuisance streams so nuisance conditions can be matched across geometries. The
bank version/hash and selected indices enter every generated scenario manifest.
This deterministic selection does not claim the native multiplayer engine is
bitwise reproducible; achieved replay determinism is measured separately.

## Generation and access

```python
from world_models.env.splits import build_split_scenario
from world_models.env.coop import TwoPlayerCoopEnv

manifest = build_split_scenario(
    "relay", "train", template_index=0, seed=123, purpose="training"
)
env = TwoPlayerCoopEnv("relay", seed=123, scenario_manifest=manifest)
```

`purpose="training"` rejects validation/test templates. Validation and test
purposes must select their respective splits. Diagnostic purpose permits an
explicit audit of any template and marks its data learning-ineligible. An
unknown template, inconsistent task/split, or substituted bank is rejected.
Legacy `build_scenario(task, seed)` remains available as a **diagnostic-only**
developer fixture; its geometry generation is preserved. Formal training and
evaluation require an explicitly allocated template.

Formal builds use a separate path under
`artifacts/phase1/split_scenarios/<template>/<seed>_f<repeat>/` unless an output
directory is supplied. The scenario records exact WAD/ACS/geometry hashes,
registry version/hash, authored object IDs, split and bank provenance. The
environment's layout seed must equal the scenario manifest seed; an engine
reset seed can subsequently reseed that fixed geometry independently.

## Validation and data boundaries

Every build checks reachable spawn/interaction/extraction routes, the relay's
closed-gate and plate-holder dependency, and equality of the marked extraction
footprint with its success rectangle. Fresh spawn centers must be outside the
task's interaction/extraction zones, including the pressure plate, so reset
warm-up cannot begin with occupancy, progress, or entry counters already set.
Relay spawn-bank x coordinates are 64, 80 and 96; its plate starts at x=104.
Publishing the split manifest additionally
checks all role/spawn/enemy-bank combinations: 60 combinations across relay
templates and 180 across cover templates. Texture changes do not alter this
collision graph. Native scripted executions in the Phase 1 acceptance campaign
validate actual collision/action mechanics on all twenty templates; static graph
checks alone are not reported as native executions.

Diagnostic, validation and test trajectories remain separate from training and
are never inserted into replay by these APIs. Metadata contains purpose,
learning eligibility, and the requirement for a counted training-ledger import.
If a later collector imports train-template diagnostic data, it must record all
attributable interactions in the training ledger before insertion. This Phase 1
split API does not create or authorize a Phase 2 replay import, and recording a
diagnostic trajectory does not silently charge or populate a training dataset.
Validation/test trajectories remain excluded from learning even if they contain
successful scripted behavior.

Termination contracts remain in `world_models.contracts` and `TwoPlayerCoopEnv`:
death, success and the declared deadline are true terminals; external collection
cutoffs retain a valid final observation and bootstrap value while stopping
sequence recursion. Final reward and event deltas belong to the transition
before reset. Missing terminal images remain invalid/masked; infrastructure
failures produce durable errors and no task-failure transition. The acceptance
report records actual explicit-path tests rather than equating these cases.
