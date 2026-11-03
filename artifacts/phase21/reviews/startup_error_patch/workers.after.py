"""Isolated ViZDoom participants and their tagged request/reply protocol."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.util
import os
from pathlib import Path
import traceback
from typing import Any, Protocol


@dataclass(frozen=True)
class WorkerConfig:
    player_index: int
    scenario_path: str
    port: int
    seed: int
    work_dir: str
    map_name: str = "map01"
    frame_skip: int = 4
    episode_timeout_tics: int = 100000
    user_variable_count: int = 16
    iwad_path: str | None = None
    iwad_sha256: str | None = None
    action_barrier: Any | None = None
    action_barrier_timeout_seconds: float = 20.0


@dataclass(frozen=True)
class RawParticipantState:
    player_index: int
    tic: int
    screen_buffer: Any | None
    variables: dict[str, float] = field(default_factory=dict)
    dead: bool = False
    finished: bool = False
    engine_reward: float = 0.0
    instance_id: str | None = None
    iwad_provenance: dict[str, str] | None = None


def resolve_iwad(path: str | None = None, expected_sha256: str | None = None) -> tuple[str, str]:
    """Select one explicit IWAD and reject changes to a previously pinned file."""
    if path is None:
        spec = importlib.util.find_spec("vizdoom")
        if spec is None or spec.origin is None:
            raise RuntimeError("ViZDoom is unavailable; an explicit IWAD is required")
        selected = Path(spec.origin).parent / "freedoom2.wad"
    else:
        selected = Path(path)
    selected = selected.resolve(strict=True)
    digest = hashlib.sha256(selected.read_bytes()).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256.lower():
        raise ValueError(f"IWAD SHA256 mismatch for {selected}: expected {expected_sha256}, got {digest}")
    return str(selected), digest


class ParticipantBackend(Protocol):
    def start(self) -> RawParticipantState: ...
    def reset(self, seed: int | None = None) -> RawParticipantState: ...
    def step(self, action: int) -> RawParticipantState: ...
    def close(self) -> None: ...


class ViZDoomBackend:
    """A worker-local game: no DoomGame object crosses a process boundary."""

    def __init__(self, config: WorkerConfig):
        self.config = config
        self.game: Any = None
        self.vzd: Any = None
        self._variable_names: tuple[str, ...] = ()
        self._iwad_provenance: dict[str, str] | None = None

    def start(self) -> RawParticipantState:
        import vizdoom as vzd
        from world_models.env.actions import button_enums

        self.vzd = vzd
        game = self.game = vzd.DoomGame()
        iwad_path, iwad_sha256 = resolve_iwad(self.config.iwad_path, self.config.iwad_sha256)
        game.set_doom_game_path(iwad_path)
        configured_iwad = str(Path(game.get_doom_game_path()).resolve())
        if configured_iwad != iwad_path:
            raise RuntimeError("engine did not retain the explicitly selected IWAD")
        game.set_doom_scenario_path(str(Path(self.config.scenario_path).resolve()))
        game.set_doom_map(self.config.map_name)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.CRCGCB)
        game.set_window_visible(False)
        game.set_sound_enabled(False)
        game.set_render_hud(True)
        game.set_render_messages(False)
        game.set_render_crosshair(True)
        game.set_episode_start_time(0)
        game.set_episode_timeout(self.config.episode_timeout_tics)
        game.set_seed(self.config.seed)
        game.set_living_reward(0.0)
        game.set_death_penalty(0.0)
        game.set_available_buttons(button_enums())
        for method in ("set_labels_buffer_enabled", "set_depth_buffer_enabled", "set_automap_buffer_enabled", "set_objects_info_enabled", "set_sectors_info_enabled"):
            setter = getattr(game, method, None)
            if setter is not None:
                setter(False)
        basic = ("POSITION_X", "POSITION_Y", "POSITION_Z", "ANGLE", "VELOCITY_X", "VELOCITY_Y", "HEALTH", "ARMOR", "SELECTED_WEAPON_AMMO",
                 "DEATHCOUNT", "HITCOUNT", "HITS_TAKEN", "DAMAGECOUNT", "DAMAGE_TAKEN", "KILLCOUNT", "ITEMCOUNT")
        self._variable_names = basic + tuple(f"USER{i}" for i in range(1, self.config.user_variable_count + 1))
        game.set_available_game_variables([getattr(vzd.GameVariable, name) for name in self._variable_names])
        network = (f"-host 2 -port {self.config.port}" if self.config.player_index == 0
                   else f"-join 127.0.0.1:{self.config.port}")
        # Cooperative play; do not inherit the deathmatch example's flags.
        # Match rendered/privileged local state to the completed authoritative
        # tic. Client-side prediction otherwise advances the console player's
        # pose toward a variable number of buffered inputs (p_user.cpp), which
        # made exact replay depend on native packet scheduling.
        game.add_game_args(network + " +deathmatch 0 +sv_forcerespawn 0 +teamdamage 0 +sv_noautoaim 1 +autoaim 0 +cl_noprediction 1 -noautoload -nosound -nomusic")
        game.add_game_args(f"+name ResearchPlayer{self.config.player_index + 1} +colorset {self.config.player_index * 3}")
        game.init()
        loaded_iwad, loaded_hash = resolve_iwad(game.get_doom_game_path(), iwad_sha256)
        if loaded_iwad != iwad_path:
            raise RuntimeError("engine loaded a different IWAD")
        self._iwad_provenance = {"path": loaded_iwad, "sha256": loaded_hash,
                                 "configured_path": configured_iwad}
        return self._snapshot()

    def _snapshot(self) -> RawParticipantState:
        state = self.game.get_state()
        # Copy before any subsequent engine step can mutate its shared buffer.
        image = state.screen_buffer.copy() if state is not None else None
        variables = {name: float(self.game.get_game_variable(getattr(self.vzd.GameVariable, name))) for name in self._variable_names}
        instance_method = getattr(self.game, "get_instance_id", None)
        return RawParticipantState(
            player_index=self.config.player_index,
            tic=int(state.tic if state is not None else self.game.get_episode_time()),
            screen_buffer=image,
            variables=variables,
            dead=bool(self.game.is_player_dead()),
            finished=bool(self.game.is_episode_finished()),
            engine_reward=float(self.game.get_last_reward()),
            instance_id=str(instance_method()) if instance_method is not None else None,
            iwad_provenance=dict(self._iwad_provenance) if self._iwad_provenance else None,
        )

    def reset(self, seed: int | None = None) -> RawParticipantState:
        if seed is not None:
            self.game.set_seed(seed)
        # Both participants must enter this call; the coordinator sends both first.
        self.game.new_episode()
        return self._snapshot()

    def step(self, action: int) -> RawParticipantState:
        from world_models.env.actions import encode_action
        if self.config.action_barrier is None:
            self.game.make_action(encode_action(action), self.config.frame_skip)
        else:
            # make_action combines input assignment and advancement. In a
            # multiplayer pair one process could advance before its peer has
            # assigned the current input. Stage both inputs first, then release
            # both native advance calls at the same decision/tic barrier.
            self.game.set_action(encode_action(action))
            self.config.action_barrier.wait(timeout=self.config.action_barrier_timeout_seconds)
            self.game.advance_action(self.config.frame_skip)
        return self._snapshot()

    def close(self) -> None:
        if self.game is not None:
            self.game.close()
            self.game = None


def participant_worker(connection: Any, config: WorkerConfig, backend_factory: Any) -> None:
    """Worker main; echoes all IDs and rejects stale commands before stepping."""
    backend = None
    episode, decision = -1, -1
    next_substep = 0
    active_substep_count = 1
    operation = "initialize_worker"
    try:
        if os.name != "nt":
            # Give watchdog cleanup a process group containing engine children.
            os.setsid()
        Path(config.work_dir).mkdir(parents=True, exist_ok=True)
        os.chdir(config.work_dir)
        operation = "construct_backend"
        backend = backend_factory(config)
        while True:
            operation = "receive_request"
            try:
                request = connection.recv()
            except (EOFError, BrokenPipeError):
                # Parent/control transport closure is normal shutdown. EOF
                # raised by a backend operation below is a simulator error.
                break
            command = request["command"]
            if command == "close":
                break
            operation = command
            requested_episode, requested_decision = request["episode_id"], request["decision_index"]
            substep = request.get("substep_index", 0)
            substep_count = request.get("substep_count", 1)
            if command == "start":
                if episode != -1 or requested_decision != 0:
                    raise ValueError("invalid duplicate start request")
                state = backend.start()
                episode, decision, next_substep = requested_episode, requested_decision, 0
            elif command == "reset":
                if requested_episode <= episode or requested_decision != 0:
                    raise ValueError("stale reset episode/decision")
                state = backend.reset(request.get("seed"))
                episode, decision, next_substep = requested_episode, requested_decision, 0
            elif command == "step":
                if requested_episode != episode or requested_decision != decision + 1 or substep != next_substep:
                    raise ValueError("stale or out-of-order step request")
                if not isinstance(substep_count, int) or substep_count < 1 or not 0 <= substep < substep_count:
                    raise ValueError("invalid native-tic substep")
                if substep == 0:
                    active_substep_count = substep_count
                elif substep_count != active_substep_count:
                    raise ValueError("substep count changed within a joint decision")
                state = backend.step(request["action"])
                next_substep += 1
                if next_substep == substep_count:
                    decision, next_substep = requested_decision, 0
            else:
                raise ValueError(f"unknown participant command: {command}")
            operation = "send_reply"
            try:
                connection.send({"status": "ok", "player_index": config.player_index,
                                 "episode_id": requested_episode, "decision_index": requested_decision,
                                 "substep_index": substep,
                                 "dispatch_id": request["dispatch_id"], "state": state})
            except (EOFError, BrokenPipeError):
                break
    except BaseException as exc:
        try:
            connection.send({"status": "error", "player_index": config.player_index,
                             "error": repr(exc), "error_type": type(exc).__name__,
                             "operation": operation, "traceback": traceback.format_exc(),
                             "episode_id": episode, "decision_index": decision})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        if backend is not None:
            try:
                backend.close()
            except Exception:
                pass
        connection.close()
