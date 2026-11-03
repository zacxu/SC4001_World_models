"""Barrier-stepped, watchdog-protected two-participant process coordinator."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import json
import importlib.util
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
from pathlib import Path
import signal
import socket
import subprocess
import tempfile
import time
from typing import Any
import uuid

import psutil

from world_models.contracts import validate_action
from world_models.env.workers import RawParticipantState, ViZDoomBackend, WorkerConfig, participant_worker, resolve_iwad


class CoordinatorError(RuntimeError):
    def __init__(self, message: str, record: dict):
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class CoordinatorConfig:
    scenario_path: str
    work_root: str
    seed: int = 0
    map_name: str = "map01"
    port: int | None = None
    frame_skip: int = 4
    timeout_seconds: float = 20.0
    startup_timeout_seconds: float = 60.0
    episode_timeout_tics: int = 100000
    user_variable_count: int = 16
    iwad_path: str | None = None
    iwad_sha256: str | None = None
    reset_strategy: str = "fresh_process"

    def __post_init__(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("seed must be a uint32 integer")
        if type(self.frame_skip) is not int or self.frame_skip < 1:
            raise ValueError("frame_skip must be positive")
        if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0
               for value in (self.timeout_seconds, self.startup_timeout_seconds)):
            raise ValueError("watchdog timeouts must be positive")
        if type(self.episode_timeout_tics) is not int or self.episode_timeout_tics < 1:
            raise ValueError("episode_timeout_tics must be a positive integer")
        if self.port is not None and (type(self.port) is not int or not 1024 <= self.port <= 65535):
            raise ValueError("port must lie between 1024 and 65535")
        if type(self.user_variable_count) is not int or not 0 <= self.user_variable_count <= 60:
            raise ValueError("user_variable_count must lie between zero and 60")
        if self.iwad_sha256 is not None and (not isinstance(self.iwad_sha256, str) or len(self.iwad_sha256) != 64 or any(c not in "0123456789abcdefABCDEF" for c in self.iwad_sha256)):
            raise ValueError("iwad_sha256 must be a SHA256 hexadecimal digest")
        if self.reset_strategy not in ("fresh_process", "native_map"):
            raise ValueError("reset_strategy must be fresh_process or native_map")


@dataclass(frozen=True)
class JointRawResult:
    episode_id: int
    decision_index: int
    dispatch_id: int
    participants: tuple[RawParticipantState, RawParticipantState]
    consumed_joint_decisions: int

    @property
    def tic(self) -> int:
        return self.participants[0].tic


@dataclass(frozen=True)
class NativeProcessIdentity:
    player_index: int
    pid: int
    create_time: float
    executable: str
    instance_id: str | None


def _json_safe(value: Any) -> Any:
    """Error evidence must remain writable even when the bad value is NaN."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else repr(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return repr(value)


class _PortLease:
    """Cross-process advisory lease plus a real UDP bind availability check.

    The UDP socket is closed before the engine starts. Holding it would prevent
    the engine binding; the file lock remains held for the engine's lifetime.
    OS locks are released on process death, so stale lock files are harmless.
    """

    def __init__(self, port: int, stream: Any):
        self.port, self.stream = port, stream

    @classmethod
    def acquire(cls, requested: int | None = None) -> _PortLease:
        root = Path(tempfile.gettempdir()) / "world-models-vizdoom-port-leases"
        root.mkdir(parents=True, exist_ok=True)
        last_error: Exception | None = None
        for _ in range(1 if requested is not None else 64):
            probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            stream = None
            locked = False
            try:
                if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                    probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
                probe.bind(("0.0.0.0", requested or 0))
                port = int(probe.getsockname()[1])
                stream = (root / f"{port}.lock").open("a+b")
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    if stream.read(1) == b"":
                        stream.write(b"0")
                        stream.flush()
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                locked = True
                return cls(port, stream)
            except OSError as exc:
                last_error = exc
                if stream is not None and not locked:
                    stream.close()
            finally:
                probe.close()
        raise OSError(f"no available ViZDoom UDP port (requested={requested}): {last_error}")

    def release(self) -> None:
        if self.stream is not None:
            # Closing the descriptor releases either platform's advisory lock.
            self.stream.close()
            self.stream = None


class TwoPlayerCoordinator:
    """Own two workers and publish only complete synchronized results.

    ``consumed_joint_decisions`` counts dispatched attempts conservatively. A
    failed attempt remains in the durable journal and is never task failure data.
    Recovery starts fresh processes/episode without resetting that counter.
    """

    def __init__(self, config: CoordinatorConfig, backend_factory: Any = ViZDoomBackend):
        if backend_factory is ViZDoomBackend:
            path, digest = resolve_iwad(config.iwad_path, config.iwad_sha256)
            config = replace(config, iwad_path=path, iwad_sha256=digest)
        self.config, self.backend_factory = config, backend_factory
        self.run_dir = Path(config.work_root).resolve() / f"env-{uuid.uuid4().hex[:12]}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._port_lease: _PortLease | None = _PortLease.acquire(config.port)
        self.port = self._port_lease.port
        self.episode_id = -1
        self.decision_index = 0
        self.dispatch_id = 0
        self.consumed_joint_decisions = 0
        self.last_tics: list[int | None] = [None, None]
        self._processes: list[Any] = []
        self._connections: list[Any] = []
        self._active = False
        self._broken = False
        self._last_result: JointRawResult | None = None
        self._action_barrier: Any | None = None
        self.worker_work_dirs: tuple[str, ...] = ()
        self._native_processes: list[NativeProcessIdentity] = []

    @property
    def native_processes(self) -> tuple[NativeProcessIdentity, ...]:
        """Only native children captured while their participant was healthy."""
        return tuple(self._native_processes)

    def _capture_native_processes(self, result: JointRawResult) -> None:
        if self.backend_factory is not ViZDoomBackend:
            return
        spec = importlib.util.find_spec("vizdoom")
        if spec is None or spec.origin is None:
            raise RuntimeError("cannot identify the installed native engine")
        expected = (Path(spec.origin).parent / ("vizdoom.exe" if os.name == "nt" else "vizdoom")).resolve()
        captured = []
        for player, worker in enumerate(self._processes):
            children = psutil.Process(worker.pid).children(recursive=True)
            matches = []
            for child in children:
                try:
                    if Path(child.exe()).resolve() == expected:
                        matches.append(child)
                except psutil.NoSuchProcess:
                    continue
            if len(matches) != 1:
                raise RuntimeError(f"participant {player} did not own exactly one installed native engine")
            child = matches[0]
            captured.append(NativeProcessIdentity(player, child.pid, child.create_time(), str(expected),
                                                   result.participants[player].instance_id))
            # Preserve partial ownership immediately if a later capture fails.
            self._native_processes = list(captured)
        self._journal("native_processes", processes=[asdict(item) for item in captured])

    def _cleanup_native_processes(self) -> None:
        """A dead Python worker cannot close its engine; use captured identities."""
        records, failed = [], []
        for identity in self._native_processes:
            try:
                process = psutil.Process(identity.pid)
                if (process.create_time() != identity.create_time
                        or Path(process.exe()).resolve() != Path(identity.executable).resolve()):
                    records.append({**asdict(identity), "cleanup": "identity_changed_no_signal_sent"})
                    continue
                process.terminate()
                try:
                    process.wait(timeout=2)
                except psutil.TimeoutExpired:
                    # Recheck the immutable identity before the stronger signal.
                    current = psutil.Process(identity.pid)
                    if (current.create_time() == identity.create_time
                            and Path(current.exe()).resolve() == Path(identity.executable).resolve()):
                        current.kill()
                        current.wait(timeout=3)
                    else:
                        records.append({**asdict(identity), "cleanup": "identity_changed_before_force_no_signal_sent"})
                        continue
                records.append({**asdict(identity), "cleanup": "terminated_owned_engine"})
            except psutil.NoSuchProcess:
                records.append({**asdict(identity), "cleanup": "already_exited"})
            except psutil.Error as exc:
                failed.append(identity)
                records.append({**asdict(identity), "cleanup": "failed", "error": repr(exc)})
        if records:
            self._journal("native_process_cleanup", processes=records)
        self._native_processes = failed
        if failed:
            raise RuntimeError("unable to stop every captured native engine")

    def _journal(self, kind: str, **fields: Any) -> None:
        record = {"kind": kind, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                  "episode_id": self.episode_id, "decision_index": self.decision_index,
                  "dispatch_id": self.dispatch_id, "consumed_joint_decisions": self.consumed_joint_decisions, **fields}
        with (self.run_dir / "dispatch_ledger.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(_json_safe(record), allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def _spawn(self) -> None:
        context = mp.get_context("spawn")
        self._native_processes = []
        self._action_barrier = context.Barrier(2)
        self.worker_work_dirs = tuple(str(self.run_dir / f"episode-{self.episode_id}" / f"player-{player}") for player in range(2))
        for player in range(2):
            parent, child = context.Pipe()
            worker_config = WorkerConfig(
                player_index=player, scenario_path=str(Path(self.config.scenario_path).resolve()),
                port=self.port, seed=self.config.seed,
                # A reset must not reload a prior engine's generated INI or
                # other mutable files. Both peers share only this generation's
                # parent, never a previous episode's working directory.
                work_dir=self.worker_work_dirs[player], map_name=self.config.map_name,
                # Native synchronous multiplayer can block if one engine is
                # permitted to run several tics ahead. Barrier each native tic;
                # the public action decision still repeats for frame_skip tics.
                frame_skip=1, episode_timeout_tics=self.config.episode_timeout_tics,
                user_variable_count=self.config.user_variable_count,
                iwad_path=self.config.iwad_path, iwad_sha256=self.config.iwad_sha256,
                action_barrier=self._action_barrier,
                action_barrier_timeout_seconds=self.config.timeout_seconds,
            )
            process = context.Process(target=participant_worker, args=(child, worker_config, self.backend_factory), name=f"doom-player-{player}")
            try:
                process.start()
            except BaseException:
                parent.close()
                raise
            finally:
                child.close()
            self._processes.append(process)
            self._connections.append(parent)

    def invalidate(self, message: str, details: Any = None) -> CoordinatorError:
        """Durably reject further interaction after a caller detects bad data."""
        self._broken = True
        record = {"message": message, "details": details, "episode_id": self.episode_id,
                  "decision_index": self.decision_index, "dispatch_id": self.dispatch_id,
                  "consumed_joint_decisions": self.consumed_joint_decisions,
                  "last_acknowledged_tics": self.last_tics,
                  "workers": [{"pid": process.pid, "alive": process.is_alive(), "exitcode": process.exitcode} for process in self._processes]}
        record = _json_safe(record)
        self._journal("infrastructure_error", error=record)
        with (self.run_dir / "last_error.json").open("w", encoding="utf-8") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False))
            stream.flush()
            os.fsync(stream.fileno())
        return CoordinatorError(message, record)

    def _error(self, message: str, details: Any = None) -> CoordinatorError:
        return self.invalidate(message, details)

    def _exchange(self, command: str, *, actions: tuple[int, int] | None = None, seed: int | None = None,
                  substep_index: int = 0, substep_count: int = 1) -> JointRawResult:
        self.dispatch_id += 1
        if command == "step" and substep_index == 0:
            self.consumed_joint_decisions += 1
        self._journal("dispatch", command=command, actions=actions, seed=seed, port=self.port,
                      substep_index=substep_index, substep_count=substep_count)
        # Both messages are sent before waiting for either blocking engine call.
        try:
            for player, connection in enumerate(self._connections):
                request = {"command": command, "episode_id": self.episode_id,
                           "decision_index": self.decision_index, "dispatch_id": self.dispatch_id,
                           "seed": seed, "substep_index": substep_index, "substep_count": substep_count}
                if actions is not None:
                    request["action"] = actions[player]
                connection.send(request)
        except (OSError, EOFError) as exc:
            raise self._error("participant dispatch failed", repr(exc)) from exc
        timeout = self.config.startup_timeout_seconds if command in ("start", "reset") else self.config.timeout_seconds
        deadline = time.monotonic() + timeout
        pending = {connection: player for player, connection in enumerate(self._connections)}
        states: dict[int, RawParticipantState] = {}
        while pending:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise self._error(f"watchdog timeout during {command}", {"pending_players": list(pending.values())})
            for connection in wait(list(pending), timeout=min(remaining, 0.2)):
                player = pending.pop(connection)
                try:
                    reply = connection.recv()
                except (EOFError, OSError) as exc:
                    raise self._error("participant connection closed", {"player": player, "error": repr(exc)}) from exc
                if reply.get("status") != "ok":
                    raise self._error("participant raised an exception", reply)
                expected = (player, self.episode_id, self.decision_index, self.dispatch_id, substep_index)
                actual = tuple(reply.get(key) for key in ("player_index", "episode_id", "decision_index", "dispatch_id", "substep_index"))
                if actual != expected:
                    raise self._error("stale or mismatched participant response", {"expected": expected, "actual": actual})
                state = reply.get("state")
                if not isinstance(state, RawParticipantState) or state.player_index != player:
                    raise self._error("invalid participant state payload")
                self.last_tics[player] = state.tic
                states[player] = state
            if any(not process.is_alive() for process in self._processes) and pending:
                # Give a queued error reply one short poll, then fail promptly.
                if not wait(list(pending), timeout=0.05):
                    raise self._error("participant exited before replying")
        if states[0].tic != states[1].tic:
            raise self._error("participant tic mismatch", {"tics": [states[i].tic for i in range(2)]})
        if command == "step" and self._last_result is not None:
            elapsed = states[0].tic - self._last_result.tic
            terminal = any(state.finished or state.dead for state in states.values())
            if elapsed != 1 and not (terminal and 0 < elapsed <= 1):
                raise self._error("unexpected simulator advancement", {"elapsed": elapsed, "expected": 1, "terminal": terminal})
        result = JointRawResult(self.episode_id, self.decision_index, self.dispatch_id,
                                (states[0], states[1]), self.consumed_joint_decisions)
        self._last_result = result
        self._journal("acknowledged", command=command, tics=[states[i].tic for i in range(2)],
                      selected_iwads=[states[i].iwad_provenance for i in range(2)] if command == "start" else None)
        return result

    def start(self) -> JointRawResult:
        if self._active:
            if self._broken:
                raise RuntimeError("coordinator is broken; call recover()")
            assert self._last_result is not None
            return self._last_result
        if self._broken:
            raise RuntimeError("coordinator is broken; call recover()")
        for attempt in range(3):
            try:
                return self._start_once()
            except CoordinatorError as exc:
                # Advisory leases exclude cooperating processes. An unrelated
                # program can still bind between preflight and engine startup.
                evidence = json.dumps(exc.record).lower()
                collision = any(text in evidence for text in ("address already in use", "could not bind", "cannot bind", "failed to bind", "wsaeaddrinuse", "10048"))
                if self.config.port is not None or not collision or attempt == 2:
                    raise
                self._journal("port_collision_retry", previous_port=self.port, attempt=attempt + 1)
                self._broken = False
        raise AssertionError("unreachable startup retry")

    def _start_once(self) -> JointRawResult:
        if self._port_lease is None:
            self._port_lease = _PortLease.acquire(self.config.port)
            self.port = self._port_lease.port
        self.episode_id += 1
        self.decision_index = 0
        self._last_result = None
        self.last_tics = [None, None]
        try:
            self._spawn()
            self._active = True
            result = self._exchange("start")
            self._capture_native_processes(result)
            return result
        except BaseException as exc:
            if not self._broken:
                error = self.invalidate("participant startup failed", repr(exc))
            else:
                error = exc
            self._stop_workers()
            raise error from (exc if error is not exc else None)

    def reset(self, seed: int | None = None) -> JointRawResult:
        if self._broken:
            raise RuntimeError("coordinator is broken; call recover()")
        if seed is not None:
            # Preserve the latest requested seed for both first start and recovery.
            self.config = replace(self.config, seed=seed)
        if not self._active:
            return self.start()
        if self.config.reset_strategy == "fresh_process":
            # P1.7 replay exposed retained native input/visual/RNG state after
            # multiplayer new_episode despite identical ready variables. A
            # fresh pair follows the exact same native startup path each time.
            # Physical reset cost is benchmarked separately, not hidden.
            self._journal("coordinated_reset", strategy="fresh_process",
                          next_episode_id=self.episode_id + 1, seed=self.config.seed)
            self._stop_workers()
            try:
                return self.start()
            except BaseException:
                self._broken = True
                raise
        self.episode_id += 1
        self.decision_index = 0
        self._last_result = None
        return self._exchange("reset", seed=seed)

    def step(self, actions: tuple[int, int]) -> JointRawResult:
        if not self._active or self._broken:
            raise RuntimeError("start/recover coordinator before stepping")
        if not isinstance(actions, (tuple, list)) or len(actions) != 2:
            raise ValueError("one action for each of two players is required")
        validated = tuple(validate_action(action) for action in actions)
        self.decision_index += 1
        result = None
        for substep in range(self.config.frame_skip):
            result = self._exchange("step", actions=validated, substep_index=substep,
                                    substep_count=self.config.frame_skip)
            if any(state.finished or state.dead for state in result.participants):
                break
        assert result is not None
        return result

    def recover(self) -> JointRawResult:
        self._stop_workers()
        self._journal("recovery", next_episode_id=self.episode_id + 1)
        self._broken = False
        try:
            return self.start()
        except BaseException:
            self._broken = True
            raise

    def _stop_workers(self) -> None:
        for connection in self._connections:
            try:
                connection.send({"command": "close"})
            except (OSError, EOFError):
                pass
        for process in self._processes:
            process.join(timeout=1)
            if process.is_alive():
                if os.name == "nt":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True, timeout=10, check=False,
                                   creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                else:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        process.terminate()
                process.join(timeout=2)
                if process.is_alive():
                    process.kill()
                    process.join(timeout=2)
        # Includes engines orphaned by an externally killed Python worker. The
        # PID and creation/executable identity checks prevent signaling a reused
        # PID or a similarly named process owned by another environment.
        cleanup_error = None
        try:
            self._cleanup_native_processes()
        except BaseException as exc:
            cleanup_error = exc
        for connection in self._connections:
            connection.close()
        self._connections.clear()
        self._processes.clear()
        self._active = False
        self._action_barrier = None
        if self._port_lease is not None:
            self._port_lease.release()
            self._port_lease = None
        if cleanup_error is not None:
            raise self.invalidate("native participant cleanup failed", repr(cleanup_error)) from cleanup_error

    def close(self) -> None:
        self._stop_workers()

    def __enter__(self) -> TwoPlayerCoordinator:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
