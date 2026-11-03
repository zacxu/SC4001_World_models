# P1.5: privileged targets and authoritative events

The runtime implementation is `src/world_models/env/targets.py`. Its canonical
registry is exported as `configs/privileged_registry_v1.json`, version
`privileged-targets-v1`. The export includes field names, sources, units,
encodings, validity, normalization, all USER assignments, event semantics and a
SHA-256 of the canonical definition. A test requires the exported JSON to equal
the runtime definition. Each generated scenario pins this version/hash and the
wire encoding, and carries its authored object registry. The wrapper rejects a
mismatched registry, USER assignment, encoding or object definition.

These fields are exclusively under `training_only_info`; they cannot be passed
through `LocalActorInput`. No actor input, action mask or reward rule changes.
Diagnostic trajectories remain separate from learning data.

## Published data

Both reset and step return:

```text
training_only_info.privileged_targets
  schema_version, registry_sha256
  players[0:2]: object_id, raw, normalized, valid
  task: raw, normalized, valid
  objects: map-authored IDs, bounds/tags/TIDs, and enemy state

training_only_info.events
  cumulative, deltas, normalized_cumulative, valid
  native_raw_counters, native_counter_baseline
  boundary_events
```

A step additionally publishes `reward_components` with `step`, `terminal`,
`milestone`, `total`, and the identical two-element `shared_rewards`. Reset has
no reward-bearing transition and reports this field as `None`.

`image_reconstruction_valid` reports each actual next-image presence. The
`StepResult.image_reconstruction_mask` property converts the boundary's matching
validity pair to floats for reconstruction losses. A missing terminal image
stays `None` with mask zero; the other player's valid image can retain mask one.
Reset requires both real images and has mask `(1.0, 1.0)`. External collection
cutoffs likewise require valid final images, retain image masks one and value
bootstrap one. A later reset does not modify an earlier terminal result/mask.

Per-player state includes x/y/z position, planar velocity, sine/cosine heading,
health, armor, selected pistol ammunition and alive status. Native ViZDoom
positions/velocities are already floating-point map units / map units per tic;
ANGLE is converted from degrees to sine/cosine once. Native player death is the
alive label's authority. Final state variables remain valid independently of
whether a terminal RGB image is available.

Normalization is explicit and uses each generated map's actual extent: x/y map
to `2*coordinate/extent-1`; z divides by 128; velocity by 32; nonnegative health
and armor by 100; pistol ammunition by 200. Raw values are retained. Values are
not clipped to a presumed maximum, except negative overkill health/armor are
clamped to zero in normalized targets. A negative ammunition sentinel is invalid
for supervision. Missing/inapplicable targets use `None` and a false mask, not
a fabricated zero. Strict versioned scenarios require every declared native
player source. Minimal unversioned unit fixtures explicitly expose missing masks.

Task targets include stage, plate occupancy, actual gate fully-open flags,
latch, console dwell/completion state, extraction lock, milestone count,
success/failure flags and elapsed decisions. Progress divides by the configured
dwell; elapsed by the deadline; stage/milestones by the family's declared range.
Relay-only targets are masked in cover and cover-only targets in relay. The task
flags describe ACS state; the published boundary and `boundary_events` apply
death/success/deadline precedence to the actual transition outcome.

Enemy x/y fields are signed ACS **16.16 fixed point**, exported without an
integer truncation and decoded by division by 65536 exactly once. The conversion
is tested against the pinned engine's `doom_fixed_to_float`, including negative
and fractional values. Enemy positions are valid while spawned and alive;
health is valid after spawning, including zero on death. The second enemy's
slot is known before spawning, with alive/spawned false and position masked.

## Authored object identity

Player TIDs are 100/101 and enemy TIDs are 200/201. Relay gate sector tags are
11/12 and extraction sectors use tag 31. Plate, latch, console and extraction
interaction-region names/bounds are authored logical IDs in the generated ACS
parameters and manifest. They are not transient renderer-label/object indices.
Role-to-spawn randomization does not change player IDs. The generated registry
must agree with these definitions and the map geometry.

## ACS field allocation

USER1-20 preserve existing task/diagnostic names; USER13/14/17/18 now declare
16.16 positions rather than truncated integer positions. This wire change is
covered by the new scenario/registry versions.

| USER | Field |
| --- | --- |
| 21/22 | Cumulative per-player plate entries/exits, summed over both players |
| 23/24 | Cumulative physical gate fully-open/fully-closed arrivals, summed over both gates |
| 25 | First latch activation count |
| 26/27 | First console A/B completion counts |
| 28/29 | ACS success/failure transition counts |
| 30/31 | ACS player 0/1 death counts |
| 32/33 | Enemy 1/2 death counts |
| 34/35 | Gates 1/2 actually fully open (`ceiling >= 128`) |
| 36/37 | Console A/B complete |
| 38/39 | ACS player 0/1 alive |
| 40 | Extraction unlocked |

Gate fully-closed arrival means `ceiling <= 0`; the initial closed state is not
counted as an event. Partially open gates have fully-open false. Gate counters
describe physical arrivals, not open/close commands. No reward is assigned for
plate/gate toggling, enemy hits/kills or pickups.

The monitors execute every native tic, including the remaining sub-tics after
an ACS terminal flag. Thus a plate entry and exit between two rendered decision
observations still increment both cumulative counters. The wrapper subtracts
the previous committed cumulative snapshot, not the two endpoint booleans.
Regressions, schema changes and inconsistent milestone totals enter durable
`NEEDS_RECOVERY` without committing state or returning a transition.

Native DEATHCOUNT, HITCOUNT, HITS_TAKEN, DAMAGECOUNT, DAMAGE_TAKEN, KILLCOUNT and
ITEMCOUNT are also retained. They are baselined at each validated reset so their
published cumulative values are episode-relative even if an engine statistic
persists across a reset strategy. Native raw counts and the baseline are retained
for audit. With automatic respawn disabled, the canonical player-death event is
the maximum of the ACS count, episode-relative native count and irreversible
native dead flag. This captures a final-sub-tic death even when the ACS monitor
has not run again. Native engine counters remain separately named for diagnosis.

Reset emits zero deltas and requires zero ACS event counters. Native baselines
and previous events commit only after the full reset observation/info validates.
On steps the same atomic rule applies. `boundary_events` distinguishes success,
death, failure, deadline and external collection cutoff without changing the
task's ACS event history. Counter normalization divides by
`deadline * frame_skip * 2`; applicability masks exclude absent family mechanics.

## One shared reward

ACS global **0** is explicitly reserved and set to `0.0`. It is never assigned
to a task flag or counter. Both native living/death rewards remain zero. A
nonzero native reward in a versioned scenario is an integrity failure, preventing
accidental double counting.

The coordinator computes one shared task reward after boundary resolution.
Milestone bonuses must equal the authoritative latch/console event deltas and
their sum must agree with the cumulative milestone counter. Reward components
must sum to the published total, and both participants receive that same total.
The approved bonuses, step cost and terminal reward schedule are preserved.

## Verification

`tests/test_targets.py` checks registry/export/hash agreement, stable IDs,
engine-utility fixed-point conversion, variable map normalization, heading,
terminal state/image validity, sentinels/missing fields, transient event
retention, multiple event deltas, native counter baselines, final-sub-tic death,
counter-regression recovery, and native reward isolation. The Phase 1 native
acceptance campaign records the matching scenario/registry and actual exported
values, events and shared rewards; unit fixtures are not substitutes for those
native traces.

Authoritative API references: [custom scenarios and reserved reward](https://vizdoom.farama.org/environments/creating_custom/),
[game variables / ACS USER1-USER60](https://vizdoom.farama.org/api/python/enums/),
and [fixed-point conversion](https://vizdoom.farama.org/api/python/utils/).
