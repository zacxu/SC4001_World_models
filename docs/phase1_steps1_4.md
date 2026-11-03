# Phase 1, first implementation step — P1.1 through P1.4

This implementation covers synchronous two-player integration, the local visual
and action interface, and the relay/cover task mechanics in the detailed plan.
The generated evidence is in
[`artifacts/phase1_requested_fixes/acceptance.json`](../artifacts/phase1_requested_fixes/acceptance.json).
See also the [readable verification summary](../artifacts/phase1_requested_fixes/README.md).
The original v1 evidence and matching source/assets are archived under
`artifacts/phase1_steps1_4/before_requested_fixes/`; those results describe the
earlier mechanics, not the revised moving-enemy task.
This historical report covers the first implementation step. P1.5's full target
registry, P1.6's formal split/termination interface and P1.7's acceptance campaign
are documented in the [completion runbook](phase1_complete.md), with separate
current evidence. Minimal flags and terminal handling in this earlier step were
required to run and check mechanics.

## P1.1 — Tested runtime and process coordination

The development simulator stack is pinned in `requirements-phase1.lock`:
ViZDoom 1.3.2, NumPy 2.5.4, Pillow 13.0.0, and their installed dependencies.
The evidence records the actual Python/platform and hashes the engine, base IWAD,
PK3, native module, and supporting runtime files. This is a development-host
simulator validation; it does not establish CUDA/PyTorch training or the separate
four-H200 allocation. The parent explicitly selects the installed
`freedoom2.wad`, pins its SHA-256, and passes both to each worker. Each worker
sets the engine IWAD path, reads it back before and after initialization, and
checks the hash. Both observed paths/hashes are included in reset/smoke evidence;
listing installed WADs alone is not treated as proof of selection.

`TwoPlayerCoordinator` creates two spawned workers. Each owns its `DoomGame`,
working directory, and request/reply pipe. The coordinator checks a UDP port by
binding it and holds a cross-process advisory lease throughout the environment's
lifetime. The probe socket closes before engine binding. A competing external
process can still race that handoff, so automatically selected ports retry
recognized bind collisions up to three startup attempts. An occupied explicit
port fails rather than silently changing. Recovery/close releases the lease.
The coordinator initializes host/guest concurrently and dispatches both actions before waiting.
All replies carry participant, episode, decision, dispatch, and substep identity.
The games use cooperative PLAYER mode, off-screen rendering, no automatic respawn,
disabled friendly fire/autoaim, and zero engine living/death reward; task reward is
computed once from authoritative mechanics events.

Actual testing found that native four-tic `make_action` chunks could deadlock a
peer. The coordinator now barriers **every native tic**, retaining one four-tic
logical action and one consumed joint-decision count. It checks matching tics and
expected advancement, journals dispatches durably before execution, records
watchdog/error context, and supports fresh-process recovery. Worker failures are
exceptions with error records, not fabricated failure rewards.

The sandbox denied Windows message-queue creation required by ViZDoom. Native
probes were run with the necessary execution permission. The multiplayer probe
subsequently completed synchronized actions and reset; this is recorded in
`multiplayer_probe/report.json`. Both participants run on this machine and the
guest joins `127.0.0.1`; the permission enables the required native IPC/networking.

The official ZDoom ACC 1.60 compiler is installed under `.tools/acc-1.60`.
`python -m scripts.bootstrap_acc` verifies the pinned Windows archive and
executable SHA-256 values. Compiler versions/binary hashes also enter scenario
manifests. See `scenarios/README.md` for Linux setup and source provenance.

## P1.2 — Local policy boundary

`actions.py` maps the 12 categorical actions to the eight declared engine buttons;
the reset START symbol cannot execute. No state-derived action mask is supplied.

`observations.py` accepts only an actual uint8 160×120 RGB buffer in the declared
layout, copies it, resizes it with Pillow bilinear interpolation, and returns an
immutable `[3,96,128]` RGB frame. Normalization produces an independent float32
array in `[0,1]`. Missing terminal images remain missing. Native local health/ammo
HUD and scenario indicators are rendered; automap, depth, labels, object metadata,
and scoreboards are not actor inputs.

`TwoPlayerCoopEnv.reset()` returns two local frames separately from
`training_only_info`. `reset_result.local_input(player, previous_action, hidden)`
constructs the strict local policy payload. The task indicators/countdown are an
explicit rendered public environmental cue; they are not engine-variable vectors
passed into the actor. Positions and mechanics state used by diagnostic scripts
remain in the training/diagnostic namespace.

The ordinary native co-op spawn effect can briefly obscure the first rendered
frame; it is retained as a real observation rather than replaced with synthetic
pixels. Later milestone frames show the unobscured scene. The compact task HUD
uses explicit virtual-pixel positions so its status/countdown and instructions
remain inside the actor-visible frame.

## P1.3 — Relay

The builder generates two separated routes. One contains a pressure plate and
the other a physical gate and latch. Holding the plate opens the runner's gate;
leaving closes it. The runner holds USE in the latch zone to release both gates
permanently. Both players then reach extraction. Geometry prevents the operator
from accessing the far side alone before the latch.

Seeded generation changes route length, gate position, obstacle placement,
texture, and role-to-spawn assignment while preserving the center paths used by
the mechanics fixture. ACS owns plate/latch/success state and persistent local
indicators. A cumulative first-latch counter pays the +0.1 bonus once. Success
pays +1; failure pays -1; each decision costs 0.001. Deadline: 400 decisions.

## P1.4 — Cover and activate

Two exposed interaction zones require 20 decisions of USE while a living player
remains in the zone. Progress pauses when USE/occupancy stops, and a completion
is irreversible. Either player may operate either console, including both
consoles in one episode. A bounded-health enemy spawns at each stage with ordinary
engine movement and combat behavior. They have distinct stable identifiers
(200/201). Completing console A does not remove its surviving enemy. Finite
enemies may be cleared before operating a console; simultaneous covering is not
a task rule.

Players begin with the same pistol, bounded ammunition, health, and armor. The
partner can suppress the shooter while the operator uses the console. After both
stages, both players must reach extraction. The bonuses are +0.05 per first
completion, +1 team success, -1 team failure, and -0.001 per decision. Deadline:
600 decisions. Either player's death ends the wrapper's team episode; automatic
respawn is disabled. No guard-proximity or player-identity condition gates USE.

## Physical extraction cues and diagnostic comparisons

Both generated tasks define one extraction rectangle. The same rectangle drives
the complete marked-floor footprint and exact fixed-point, half-open
player-center containment used by ACS success checks. Its amber checker floor
and wall-mounted EXIT/LOCKED sign turn green and show EXIT/OPEN when the relay
latch or both cover consoles complete. This marker style is identical across
randomized layouts. A visible footprint refers to the player's center; the
sprite/collision radius can extend over its edge without the center entering.

The revised pilot extraction bounds are aligned to the existing 64-unit floor
grid so the visible region has no invisible inset. This changes the earlier
unmarked rectangles; it is an explicit geometry change, not just new artwork:

| Task | Earlier bounds `[x1,y1,x2,y2]` | Revised bounds |
| --- | --- | --- |
| Relay | `[common+24,280,common+120,424]` | `[common,256,common+128,448]` |
| Cover | `[width-112,616,width-24,728]` | `[width-128,640,width,768]` |

Both living player centers must enter the revised rectangle after unlocking.
The same generated geometry applies to B and D; stages, dwell, deadlines, and
rewards are preserved. The prior rectangles and matching WADs remain archived.

The marker audit saves actual resized 128-by-96 local actor images, camera
positions/headings, task state, and scenario hashes at several approaches.
Privileged positions only place the diagnostic cameras through ordinary actions;
this is a human visual audit, not a trained image-navigation result. Relay's
plate-holder cannot see locked extraction before the latch through its closed
route; swapped role assignments allow both player indices to supply locked views
across layouts. Both players supply unlocked views without bypassing the task.

`cover-diagnostics` pairs identical layouts/engine seeds with covering fire
enabled and withheld. The operator begins USE as soon as its own position and
momentum allow it; the guard's arrival does not gate progress. Stage two starts
extraction navigation regardless of surviving enemies. Withholding fire strips
only attack components from actions; navigation, turning, USE, and stage rules
are shared, although resulting trajectories may differ under combat.

The report measures success, activation-time operator health/armor loss, deaths,
completion decisions/seconds, and ammunition expenditure. It checks that the
enabled treatment fires actual pistol rounds during activation, and that the
withheld treatment uses both consoles and attempts extraction with zero shots.
Actual ammunition decreases and ATTACK-command counts are reported separately.
These are descriptive development diagnostics: a null or negative covering
effect is a valid outcome, not a reason to alter task rules or reject the task.
Damage reduction supports protection; higher success is stronger evidence.
The research comparison remains B versus D only.

## State and clock implementation details

ACS globals may survive `new_episode` until the level's OPEN script executes.
Every wrapper reset crosses one initialization boundary before trusting ready or
completion flags. These initialization decisions are recorded separately and
remain included in the coordinator's durable consumed-dispatch ledger. Both
players then start with fresh task state and local images.

ACS sets its start at a native frame-skip boundary. Under the tested settings,
the ready observation is engine tic 5 with ACS start tic 4 and elapsed decision
zero; the next actor decision reaches tic 9 with elapsed one. The smoke check
asserts ACS/wrapper clock agreement. The wrapper enforces the task deadline and
death/success/deadline precedence, bounds one-time milestone rewards, and rejects
unexplained engine completion, mismatched task state, or counter regression.

The wrapper explicitly tracks UNINITIALIZED, READY, STEPPING, EPISODE_ENDED,
NEEDS_RECOVERY, and CLOSED states. Raw identifiers, flags/counters, boundary,
reward, metadata, both RGB conversions, and serializable info are staged before
committing the clock/previous transition. Any post-dispatch integrity error is a
typed `EnvironmentIntegrityError`, recorded through the public coordinator
`invalidate()` interface and its durable journal/error file. It blocks both
`step()` and ordinary `reset()` until `recover()` restarts both participants and
validates a fresh episode. Reset/recovery info explicitly requests both actor
memories reset. Consumed attempts and initialization counts remain cumulative.
Invalid transitions never become task-failure rewards. Unit fault injections
cover disagreeing flags, counter regression, malformed images/rewards/metadata,
unexpected engine completion, and failed reset/recovery; the native recovery
smoke also injects a conversion failure after an actual engine dispatch.

## Commands

Run from the project root. On Linux substitute `.venv/bin/python` for the Windows
interpreter and provide the official compatible ACC compiler.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m scripts.bootstrap_acc
.\.venv\Scripts\python.exe -m scripts.phase1 build --task both --seeds 0 1 2
.\.venv\Scripts\python.exe -m scripts.phase1 smoke --task both
.\.venv\Scripts\python.exe -m scripts.phase1 mechanics --task relay --seeds 0
.\.venv\Scripts\python.exe -m scripts.phase1 mechanics --task cover --seeds 0
.\.venv\Scripts\python.exe -m scripts.phase1 mechanics --task cover --seeds 0 --operator-policy same --no-mechanics-probes
.\.venv\Scripts\python.exe -m scripts.phase1 cover-diagnostics --seeds 0 1 2 --output-dir artifacts/phase1_requested_fixes/paired
.\.venv\Scripts\python.exe -m scripts.phase1 recovery-smoke --output-dir artifacts/phase1_requested_fixes/recovery
.\.venv\Scripts\python.exe -m scripts.phase1 marker-audit --task both --seeds 0 1 2 --output-dir artifacts/phase1_requested_fixes/markers
.\.venv\Scripts\python.exe -m scripts.phase1 verify --output-dir artifacts/phase1_requested_fixes
```

`verify` runs the unit suite, live task smoke/resets, scripted completions on
three seeds per family, each-player-inactive sanity checks, a same-operator cover
completion, paired fire/no-fire cases, marker captures, and native recovery.
It records source and runtime-asset hashes before and after execution and rejects
drift, while each worker also verifies its selected IWAD. Progress checkpoints
retain completed checks if a later fixture fails.
Its controllers
use privileged geometry/positions solely to validate mechanics. They use the
declared actions and do not teleport, issue kill commands, supply human policy
scores, or enter learning replay implicitly. JSON traces and exact actor-visible
PNG frames accompany results. This targeted verification is not P1.7's formal
10,000-decision/100-reset/throughput acceptance campaign.

## Provenance

Generated PWADs contain project-authored UDMF geometry and ACS scripts; sprites,
textures, and actors are referenced from the separately installed Freedoom IWAD.
Preserve upstream Freedoom license/credits when redistributing that IWAD. Original
ViZDoom code is MIT; bundled engine components retain their separate licensing.
The runtime manifest links upstream license statements rather than assigning one
license to the entire engine bundle.
