# Fixed-action replay and fresh multiplayer reset

P1.7 checks an externally fixed sequence of joint categorical actions. Each run
receives identical action pairs, even if the observed states diverge. This
separates simulator repeatability from a closed-loop controller that can amplify
one small difference into a different future action sequence.

## Observed failure and correction

The initial probe reproduced a 114-action cover sequence exactly across three
fresh environment instances, including player variables, task fields, reward,
termination and actor RGB bytes. Reusing the same native participants through
`new_episode()` did not reproduce the trace: ready variables matched, but the
first movement result and later combat/event traces differed. Ready images also
differed. Those failed traces are retained under
`artifacts/phase1_remaining/determinism_initial/`.

The default coordinator reset now closes both participants and starts a fresh
pair with the requested seed. It preserves the episode-ID progression and
cumulative consumed-interaction/initialization counts. Every native startup has
its own `episode-N/player-I` working directory so an earlier engine's generated
INI or other mutable files cannot be loaded implicitly. The journal records the
reset strategy and actual dispatch port. `CoordinatorConfig.reset_strategy`
defaults to `fresh_process`; `native_map` remains an explicit diagnostic option
for reproducing the old behavior.

After healthy startup, the parent captures each worker's exact installed native
engine child as an immutable PID, creation-time, executable-path and instance-ID
identity. `coordinator.native_processes` exposes these identities and the journal
retains them. Cleanup uses those captured identities even if a Python worker has
already died, so its native engine cannot be silently orphaned. A reused PID or
different executable is never signaled; the identity is checked again before a
forced kill. Cleanup failures remain durable errors and prevent a silent restart.

One later fresh-pair replay still showed a small movement difference. The
coordinator now gives both participant backends a shared input-readiness barrier.
Each backend calls `set_action`, waits until both inputs have been assigned,
then calls `advance_action(1)`. The existing parent barrier still waits for both
snapshots on every native tic. Barrier failure is an infrastructure error, not
a transition. The four-tic public action repetition remains unchanged.

A four-repeat cover probe then matched exactly, but a later relay repeat still
showed a one-tic-scale local pose difference while authoritative events matched.
Inspection of the pinned engine source identified client prediction as an
additional state layer: `P_PredictPlayer` can advance the console player's pose
toward the count of buffered input commands. The worker now pins
`+cl_noprediction 1` for both peers so local RGB and queried poses reflect the
completed authoritative state. Replay reports read this recognized CVAR back
from each generated engine INI, rather than trusting an unknown-option string.

Further final acceptance results are reported in `phase1_acceptance.json`;
measured replay equality is not a mathematical guarantee for every future
map/platform. The earlier closed-loop calibration's specific root cause is not
assigned retrospectively.

## Comparison and evidence

`world_models.env.replay_diagnostics.run_fixed_action_replay()` accepts action
pairs and an optional scenario manifest. `reset_mode='fresh_process'` creates a
new Python wrapper per repeat; `reset_mode='coordinated_reset'` reuses the wrapper
and exercises its public reset, which uses fresh native participants by default.
Reports include this distinction and the underlying coordinator reset strategy.

The comparison uses exact canonical numeric/state/event/reward equality, without
rounding or tolerance. It checks scenario bytes, selected IWAD provenance and
unchanged environment source hashes. Wall time, worker instance names, episode
IDs and cumulative cross-episode accounting are retained as provenance rather
than compared as physical state. Native and derived privileged state, event
deltas and reward components are compared. RGB SHA-256 values are checked
separately; if they differ, the first differing real frames and channel-error
statistics are saved for investigation. RGB difference alone does not hide a
state/event difference or become an invented task failure.

```powershell
.\.venv\Scripts\python.exe -m world_models.env.replay_diagnostics cover --trace artifacts/phase1_requested_fixes/mechanics/cover_0_inactiveNone_fire1_swap.json --repeats 3 --reset-mode coordinated_reset --output-dir artifacts/replay_check
```

The trace file only supplies a frozen action list; the simulator is run again.
Diagnostic transitions are kept outside training and are not inserted into a
learning ledger implicitly. Reset startup cost is measured separately in the
collector benchmark rather than hidden in steady-state step throughput.
Each repeat records both cumulative counters and consumed/initialization deltas;
sum the deltas, rather than repeated cumulative values, for the replay campaign's
physical interaction total. Coordinator-ledger paths identify the underlying
dispatch evidence.

## Primary sources

The [official DoomGame API](https://vizdoom.farama.org/api/python/doom_game/)
defines the input assignment, advancement, seed and multiplayer reset methods.
The pinned [1.3.2 multiple-instance example](https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/1.3.2/examples/python/multiple_instances.py)
initializes peers concurrently. The [1.3.2 controller source](https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/1.3.2/src/lib/ViZDoomController.cpp)
shows that multiplayer map restart follows a distinct path from engine startup,
including seed commands and native restart tics. These sources motivate testing
the actual pinned setup; the measured traces establish the achieved level.
The pinned [player prediction source](https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/1.3.2/src/vizdoom/src/p_user.cpp)
defines `cl_noprediction` and the prediction path; the
[network loop](https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/1.3.2/src/vizdoom/src/d_net.cpp)
calls that path after native tick processing.
