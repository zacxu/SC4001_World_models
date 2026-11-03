"""Process-level synchronization/watchdog tests using a deterministic backend."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import socket
import tempfile
import time
import unittest
from unittest.mock import MagicMock, patch

from world_models.env.coordinator import CoordinatorConfig, CoordinatorError, JointRawResult, NativeProcessIdentity, TwoPlayerCoordinator, _PortLease
from world_models.env.workers import RawParticipantState, ViZDoomBackend, WorkerConfig, resolve_iwad


class BarrierBackend:
    """Both participants must enter each operation before either can finish."""

    def __init__(self, config):
        self.config = config
        self.tic = 3
        self.operation = 0

    def _barrier(self, operation):
        root = Path(self.config.work_dir).parent
        (root / f"{operation}-{self.config.player_index}.ready").write_text("ready")
        peer = root / f"{operation}-{1-self.config.player_index}.ready"
        limit = time.monotonic() + 5
        while not peer.exists():
            if time.monotonic() >= limit:
                raise RuntimeError("other participant was not released concurrently")
            time.sleep(0.005)

    def _state(self):
        return RawParticipantState(self.config.player_index, self.tic, bytes([self.config.player_index]), {"USER1": 0.0})

    def start(self):
        self._barrier("start")
        return self._state()

    def reset(self, seed=None):
        self.operation += 1
        self._barrier(f"reset-{self.operation}")
        self.tic = 3
        return self._state()

    def step(self, action):
        self.operation += 1
        self._barrier(f"step-{self.operation}")
        self.tic += self.config.frame_skip
        return self._state()

    def close(self):
        pass


class FailingBackend(BarrierBackend):
    def step(self, action):
        if self.config.player_index == 1:
            raise RuntimeError("fixture worker exception")
        return super().step(action)


class SeedBackend(BarrierBackend):
    def _state(self):
        return RawParticipantState(self.config.player_index,self.tic,None,{"seed":float(self.config.seed)})


class GenerationBackend(BarrierBackend):
    def _state(self):
        return RawParticipantState(self.config.player_index, self.tic, None,
                                   instance_id=f"{os.getpid()}|{self.config.work_dir}")


class StallingBackend(BarrierBackend):
    def step(self, action):
        if self.config.player_index == 1:
            time.sleep(10)
        return super().step(action)


class DriftingBackend(BarrierBackend):
    def step(self, action):
        state = super().step(action)
        if self.config.player_index == 1:
            return RawParticipantState(1, self.tic + 1, state.screen_buffer)
        return state


class BindCollisionBackend(BarrierBackend):
    def start(self):
        marker = Path(self.config.work_dir).parent.parent / "collision-seen"
        if self.config.player_index == 0 and not marker.exists():
            marker.write_text("yes")
            raise RuntimeError("could not bind UDP socket: address already in use")
        return super().start()


def _probe_port_lease(port, result):
    try:
        lease = _PortLease.acquire(port)
    except OSError:
        result.put(False)
    else:
        lease.release()
        result.put(True)


class CoordinatorTests(unittest.TestCase):
    def make(self, directory, backend=BarrierBackend, **kwargs):
        return TwoPlayerCoordinator(CoordinatorConfig("fixture.wad", directory, startup_timeout_seconds=10, **kwargs), backend)

    def test_both_participants_are_released_before_wait_and_reset_is_coordinated(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder)
            try:
                initial = coordinator.start()
                self.assertEqual(initial.tic, 3)
                self.assertEqual(initial.decision_index, 0)
                first = coordinator.step((1, 2))
                self.assertEqual(first.tic, 7)
                self.assertEqual(first.decision_index, 1)
                self.assertEqual(first.consumed_joint_decisions, 1)
                reset = coordinator.reset(seed=5)
                self.assertEqual(reset.episode_id, 1)
                self.assertEqual(reset.decision_index, 0)
                self.assertEqual(reset.tic, 3)
                self.assertEqual(coordinator.step((0, 0)).consumed_joint_decisions, 2)
            finally:
                coordinator.close()

    def test_first_reset_seed_reaches_both_worker_configurations(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder,SeedBackend)
            try:
                first = coordinator.reset(seed=123)
                self.assertEqual([p.variables["seed"] for p in first.participants],[123.,123.])
                self.assertEqual(coordinator.config.seed,123)
            finally:
                coordinator.close()
        for seed in (True,-1,2**32):
            with self.subTest(seed=seed),self.assertRaises(ValueError):
                CoordinatorConfig("fixture.wad","unused",seed=seed)

    def test_fresh_reset_replaces_both_workers_and_generation_directories_without_losing_counts(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder, GenerationBackend)
            try:
                old = coordinator.start()
                coordinator.step((0, 0))
                fresh = coordinator.reset(seed=123)
                self.assertEqual(fresh.episode_id, old.episode_id + 1)
                self.assertEqual(fresh.decision_index, 0)
                self.assertEqual(fresh.consumed_joint_decisions, 1)
                self.assertTrue(all(left.instance_id != right.instance_id for left, right in zip(old.participants, fresh.participants)))
                for participant in old.participants:
                    self.assertIn("episode-0", participant.instance_id)
                for participant in fresh.participants:
                    self.assertIn("episode-1", participant.instance_id)
                self.assertTrue((coordinator.run_dir / "episode-0" / "start-0.ready").is_file())
                self.assertTrue((coordinator.run_dir / "episode-1" / "start-0.ready").is_file())
                rows = [json.loads(line) for line in (coordinator.run_dir / "dispatch_ledger.jsonl").read_text().splitlines()]
                resets = [row for row in rows if row["kind"] == "coordinated_reset"]
                self.assertEqual(len(resets), 1)
                self.assertEqual(resets[0]["strategy"], "fresh_process")
                self.assertEqual(resets[0]["consumed_joint_decisions"], 1)
                self.assertEqual(coordinator.step((0, 0)).consumed_joint_decisions, 2)
            finally:
                coordinator.close()

    def test_legacy_native_map_reset_is_explicit_and_retains_workers(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder, GenerationBackend, reset_strategy="native_map")
            try:
                old = coordinator.start()
                fresh = coordinator.reset()
                self.assertEqual([p.instance_id for p in old.participants], [p.instance_id for p in fresh.participants])
                self.assertEqual(fresh.episode_id, 1)
            finally:
                coordinator.close()
        with self.assertRaises(ValueError):
            CoordinatorConfig("fixture.wad", "unused", reset_strategy="unknown")

    def test_invalid_joint_actions_are_rejected_before_dispatch(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.make(folder) as coordinator:
                for actions in ((0,), (0, 12), (True, 0)):
                    with self.subTest(actions=actions), self.assertRaises(ValueError):
                        coordinator.step(actions)
                self.assertEqual(coordinator.consumed_joint_decisions, 0)

    def test_desynchronization_is_an_infrastructure_error_not_task_data(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.make(folder, DriftingBackend) as coordinator:
                with self.assertRaises(CoordinatorError) as caught:
                    coordinator.step((0, 0))
                self.assertIn("tic mismatch", str(caught.exception))
                self.assertEqual(coordinator.consumed_joint_decisions, 1)
                self.assertTrue((coordinator.run_dir / "last_error.json").is_file())
                with self.assertRaises(RuntimeError):
                    coordinator.step((0, 0))

    def test_worker_exception_is_recorded_with_identifiers(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.make(folder, FailingBackend) as coordinator:
                with self.assertRaises(CoordinatorError) as caught:
                    coordinator.step((0, 0))
                record = caught.exception.record
                self.assertEqual(record["episode_id"], 0)
                self.assertEqual(record["decision_index"], 1)
                self.assertEqual(record["consumed_joint_decisions"], 1)
                self.assertIn("workers", record)
                rows = [json.loads(line) for line in (coordinator.run_dir / "dispatch_ledger.jsonl").read_text().splitlines()]
                self.assertTrue(any(row["kind"] == "infrastructure_error" for row in rows))

    def test_watchdog_preserves_consumption_and_recovery_starts_fresh_episode(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder, StallingBackend, timeout_seconds=0.1)
            try:
                coordinator.start()
                with self.assertRaises(CoordinatorError) as caught:
                    coordinator.step((0, 0))
                self.assertIn("watchdog", str(caught.exception))
                self.assertEqual(coordinator.consumed_joint_decisions, 1)
                recovered = coordinator.recover()
                self.assertEqual(recovered.episode_id, 1)
                self.assertEqual(recovered.consumed_joint_decisions, 1)
                self.assertEqual(recovered.decision_index, 0)
            finally:
                coordinator.close()

    def test_instances_have_separate_workspaces_and_ports(self):
        with tempfile.TemporaryDirectory() as folder:
            left, right = self.make(folder), self.make(folder)
            try:
                self.assertNotEqual(left.run_dir, right.run_dir)
                self.assertNotEqual(left.port, right.port)
            finally:
                left.close()
                right.close()

    def test_public_invalidation_is_durable_with_nonfinite_evidence(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.make(folder) as coordinator:
                coordinator.step((0, 0))
                error = coordinator.invalidate("bad wrapper data", {"nan":float("nan"), "inf":float("inf")})
                record = json.loads((coordinator.run_dir / "last_error.json").read_text())
                self.assertEqual(record["details"], {"nan":"nan", "inf":"inf"})
                self.assertEqual(error.record["consumed_joint_decisions"], 1)
                for operation in (lambda: coordinator.step((0,0)), lambda: coordinator.reset(), lambda: coordinator.start()):
                    with self.assertRaises(RuntimeError):
                        operation()
                    self.assertEqual(coordinator.consumed_joint_decisions, 1)
                recovered = coordinator.recover()
                self.assertEqual(recovered.consumed_joint_decisions, 1)
                self.assertEqual(recovered.decision_index, 0)
                self.assertEqual(coordinator.step((0,0)).consumed_joint_decisions, 2)

    def test_occupied_fixed_port_is_rejected_and_leases_cross_process_boundaries(self):
        with tempfile.TemporaryDirectory() as folder:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as occupied:
                occupied.bind(("0.0.0.0", 0))
                with self.assertRaises(OSError):
                    self.make(folder, port=occupied.getsockname()[1])
            coordinator = self.make(folder)
            port = coordinator.port
            try:
                # The engine can bind this port while the project lease is held.
                with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as engine:
                    engine.bind(("0.0.0.0", port))
                with self.assertRaises(OSError):
                    self.make(folder, port=port)
                context = mp.get_context("spawn")
                result = context.Queue()
                child = context.Process(target=_probe_port_lease, args=(port, result))
                child.start()
                self.assertFalse(result.get(timeout=10))
                child.join(timeout=10)
                self.assertEqual(child.exitcode, 0)
                result.close()
            finally:
                coordinator.close()
            replacement = self.make(folder, port=port)
            replacement.close()

    def test_automatic_bind_collision_retries_and_fixed_port_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder, BindCollisionBackend)
            try:
                result = coordinator.start()
                self.assertEqual(result.episode_id, 1)
                self.assertEqual(coordinator.consumed_joint_decisions, 0)
                records = [json.loads(line) for line in (coordinator.run_dir / "dispatch_ledger.jsonl").read_text().splitlines()]
                self.assertEqual(sum(row["kind"] == "port_collision_retry" for row in records), 1)
            finally:
                coordinator.close()
        with tempfile.TemporaryDirectory() as folder:
            lease = _PortLease.acquire()
            port = lease.port
            lease.release()
            coordinator = self.make(folder, BindCollisionBackend, port=port)
            try:
                with self.assertRaises(CoordinatorError):
                    coordinator.start()
                self.assertEqual(coordinator.episode_id, 0)
                # Startup failure releases the cross-process reservation.
                replacement = _PortLease.acquire(port)
                replacement.release()
            finally:
                coordinator.close()

    def test_iwad_is_explicit_hashed_and_engine_readback_is_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            iwad = Path(folder) / "base.wad"
            iwad.write_bytes(b"test IWAD")
            path, digest = resolve_iwad(str(iwad))
            self.assertEqual(resolve_iwad(path, digest), (path, digest))
            config = WorkerConfig(0,"scenario.wad",12345,0,folder,iwad_path=path,iwad_sha256=digest,user_variable_count=0)
            game = MagicMock()
            game.get_doom_game_path.return_value = path
            vzd = MagicMock()
            vzd.DoomGame.return_value = game
            backend = ViZDoomBackend(config)
            with patch.dict("sys.modules", {"vizdoom": vzd}), patch("world_models.env.actions.button_enums", return_value=[]), patch.object(backend,"_snapshot",return_value="state"):
                self.assertEqual(backend.start(), "state")
            game.set_doom_game_path.assert_called_once_with(path)
            self.assertEqual(game.get_doom_game_path.call_count, 2)
            self.assertEqual(backend._iwad_provenance["sha256"], digest)
            other_iwad = Path(folder)/"other.wad"
            other_iwad.write_bytes(iwad.read_bytes())
            game.get_doom_game_path.side_effect = [path,str(other_iwad.resolve())]
            with patch.dict("sys.modules", {"vizdoom": vzd}), patch("world_models.env.actions.button_enums", return_value=[]):
                with self.assertRaisesRegex(RuntimeError,"different IWAD"):
                    ViZDoomBackend(config).start()
            game.get_doom_game_path.side_effect = None
            game.get_doom_game_path.return_value = str(Path(folder)/"wrong.wad")
            with patch.dict("sys.modules", {"vizdoom": vzd}), patch("world_models.env.actions.button_enums", return_value=[]):
                with self.assertRaisesRegex(RuntimeError, "retain"):
                    ViZDoomBackend(config).start()
            iwad.write_bytes(b"modified")
            with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                resolve_iwad(path,digest)
            with patch.dict("sys.modules", {"vizdoom": vzd}), patch("world_models.env.actions.button_enums", return_value=[]):
                with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
                    ViZDoomBackend(config).start()

    def test_native_actions_are_assigned_before_joint_barrier_and_advance(self):
        order = []
        barrier = MagicMock()
        barrier.wait.side_effect = lambda **kwargs: order.append(("barrier", kwargs["timeout"]))
        config = WorkerConfig(0, "scenario.wad", 12345, 0, "unused", frame_skip=1,
                              action_barrier=barrier, action_barrier_timeout_seconds=3.5)
        backend = ViZDoomBackend(config)
        backend.game = MagicMock()
        backend.game.set_action.side_effect = lambda action: order.append(("set", action))
        backend.game.advance_action.side_effect = lambda tics: order.append(("advance", tics))
        with patch.object(backend, "_snapshot", return_value="state"):
            self.assertEqual(backend.step(1), "state")
        self.assertEqual([item[0] for item in order], ["set", "barrier", "advance"])
        self.assertEqual(order[1], ("barrier", 3.5))
        self.assertEqual(order[2], ("advance", 1))
        backend.game.make_action.assert_not_called()

    def test_native_child_capture_uses_exact_executable_and_exposes_frozen_identities(self):
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder)
            expected = (Path(folder)/("vizdoom.exe" if os.name == "nt" else "vizdoom")).resolve()
            children = []
            for index in range(2):
                child = MagicMock()
                child.pid = 600 + index
                child.exe.return_value = str(expected)
                child.create_time.return_value = 10.0 + index
                unrelated = MagicMock()
                unrelated.exe.return_value = str(Path(folder)/"unrelated.exe")
                worker = MagicMock()
                worker.children.return_value = [child, unrelated]
                children.append(worker)
            raw = JointRawResult(0,0,0,tuple(RawParticipantState(index,1,None,instance_id=f"native-{index}") for index in range(2)),0)
            try:
                coordinator.backend_factory = ViZDoomBackend
                coordinator._processes = [SimpleNamespace(pid=500),SimpleNamespace(pid=501)]
                with patch("world_models.env.coordinator.importlib.util.find_spec", return_value=SimpleNamespace(origin=str(Path(folder)/"__init__.py"))), patch("world_models.env.coordinator.psutil.Process",side_effect=children):
                    coordinator._capture_native_processes(raw)
                self.assertEqual([item.pid for item in coordinator.native_processes],[600,601])
                self.assertEqual([item.instance_id for item in coordinator.native_processes],["native-0","native-1"])
                self.assertIsInstance(coordinator.native_processes,tuple)
                with self.assertRaises(AttributeError):
                    coordinator.native_processes[0].pid = 999
            finally:
                coordinator._processes = []
                coordinator._native_processes = []
                coordinator.close()

    def test_orphan_cleanup_rechecks_creation_and_executable_before_signaling(self):
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder)
            expected = str((Path(folder)/"vizdoom.exe").resolve())
            owned, reused, changed = MagicMock(), MagicMock(), MagicMock()
            owned.create_time.return_value = 1.0
            owned.exe.return_value = expected
            reused.create_time.return_value = 999.0
            reused.exe.return_value = expected
            changed.create_time.return_value = 3.0
            changed.exe.return_value = str(Path(folder)/"unrelated.exe")
            try:
                coordinator._native_processes = [NativeProcessIdentity(index,100+index,float(index+1),expected,None) for index in range(3)]
                with patch("world_models.env.coordinator.psutil.Process",side_effect=[owned,reused,changed]):
                    coordinator._cleanup_native_processes()
                owned.terminate.assert_called_once_with()
                owned.wait.assert_called_once_with(timeout=2)
                reused.terminate.assert_not_called()
                changed.terminate.assert_not_called()
                self.assertEqual(coordinator.native_processes,())
            finally:
                coordinator._native_processes = []
                coordinator.close()

    def test_force_cleanup_reopens_process_identity_and_preserves_failures_for_retry(self):
        import psutil
        with tempfile.TemporaryDirectory() as folder:
            coordinator = self.make(folder)
            expected = str((Path(folder)/"vizdoom.exe").resolve())
            old, replacement = MagicMock(), MagicMock()
            old.create_time.return_value = 1.0
            old.exe.return_value = expected
            old.wait.side_effect = psutil.TimeoutExpired(2)
            replacement.create_time.return_value = 2.0
            replacement.exe.return_value = expected
            identity = NativeProcessIdentity(0,100,1.0,expected,None)
            try:
                coordinator._native_processes = [identity]
                with patch("world_models.env.coordinator.psutil.Process",side_effect=[old,replacement]):
                    coordinator._cleanup_native_processes()
                old.terminate.assert_called_once_with()
                old.kill.assert_not_called()
                replacement.kill.assert_not_called()
                coordinator._native_processes = [identity]
                with patch("world_models.env.coordinator.psutil.Process",side_effect=psutil.AccessDenied(100)):
                    with self.assertRaisesRegex(RuntimeError,"unable to stop"):
                        coordinator._cleanup_native_processes()
                self.assertEqual(coordinator.native_processes,(identity,))
            finally:
                coordinator._native_processes = []
                coordinator.close()


if __name__ == "__main__":
    unittest.main()
