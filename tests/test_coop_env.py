"""Task-wrapper contracts tested independently of native ViZDoom availability."""

from copy import deepcopy
from dataclasses import replace
import unittest

import numpy as np

from world_models.contracts import BoundaryReason, PREVIOUS_ACTION_START
from world_models.env.coop import EnvironmentIntegrityError, LifecycleState, TwoPlayerCoopEnv
from world_models.env.coordinator import CoordinatorError, JointRawResult
from world_models.env.workers import RawParticipantState


class FakeCoordinator:
    def __init__(self):
        self.index = 0
        self.episode = 0
        self.count = 0
        self.dispatch = 0
        self.flags = {"USER1": 0., "USER2": 0., "USER3": 0., "USER4": 0., "USER5": 0., "USER6": 0.,
                      "USER7": 0., "USER8": 0., "USER9": 0., "USER10": 2., "USER11": 4.}
        self.dead = False
        self.missing = False
        self.error = False
        self.last_reset_seed = None
        self.transform = None
        self.broken = False
        self.errors = []

    def result(self):
        participants = tuple(RawParticipantState(
            player_index=i, tic=1+4*self.index,
            screen_buffer=None if self.missing else np.full((3,120,160), i*100, dtype=np.uint8),
            variables=dict(self.flags), dead=self.dead,
        ) for i in range(2))
        result = JointRawResult(self.episode,self.index,self.dispatch,participants,self.count)
        return self.transform(result) if self.transform else result

    def start(self):
        return self.result()

    def reset(self, seed=None):
        self.last_reset_seed = seed
        self.index = 0
        self.episode += 1
        self.dispatch += 1
        self.flags.update(USER1=0.,USER2=0.,USER3=0.,USER4=0.,USER5=0.,USER6=0.,USER7=0.,USER8=0.,USER9=0.)
        self.dead = self.missing = False
        return self.result()

    def step(self, actions):
        if self.broken:
            raise RuntimeError("recover first")
        self.count += 1
        self.dispatch += 4
        if self.error:
            raise RuntimeError("worker unavailable")
        self.index += 1
        self.flags["USER9"] = float(max(0,self.index-1))
        return self.result()

    def recover(self):
        self.error = False
        self.broken = False
        return self.reset()

    def invalidate(self, message, details=None):
        self.broken = True
        record = {"message": message, "details": details, "consumed_joint_decisions": self.count}
        self.errors.append(record)
        return CoordinatorError(message, record)

    def close(self):
        pass


class CoopEnvironmentTests(unittest.TestCase):
    def make_env(self):
        coordinator = FakeCoordinator()
        manifest = {"task_family":"relay", "frame_skip":4, "deadline":400, "wad_path":"unused.wad",
                    "user_variables":{"stage":1,"success":2,"failure":3,"milestone_count":4,"registered_players":10}}
        return TwoPlayerCoopEnv("relay",scenario_manifest=manifest,coordinator=coordinator), coordinator

    def test_first_reset_seed_is_forwarded(self):
        env, coordinator = self.make_env()
        env.reset(seed=123)
        self.assertEqual(coordinator.last_reset_seed,123)

    def test_provided_layout_seed_must_match(self):
        manifest = {"task_family":"relay","frame_skip":4,"deadline":400,"seed":1,"wad_path":"unused.wad","user_variables":{}}
        with self.assertRaisesRegex(ValueError,"layout seed"):
            TwoPlayerCoopEnv("relay",seed=0,scenario_manifest=manifest,coordinator=FakeCoordinator())

    def test_cover_dwell_mismatch_is_rejected(self):
        manifest = {"task_family":"cover","frame_skip":4,"deadline":600,"dwell_decisions":19,"wad_path":"unused.wad","user_variables":{}}
        with self.assertRaisesRegex(ValueError,"dwell"):
            TwoPlayerCoopEnv("cover",scenario_manifest=manifest,coordinator=FakeCoordinator())

    def test_reset_flushes_stale_globals_and_preserves_local_images(self):
        env, coordinator = self.make_env()
        reset = env.reset()
        self.assertEqual(env.initialization_joint_decisions,1)
        local = reset.local_input(1,PREVIOUS_ACTION_START,(0.,)*256)
        self.assertEqual(local.local_rgb.data, bytes([100])*36864)
        self.assertFalse(hasattr(local,"training_only_info"))
        reset.training_only_info["scenario_manifest"]["user_variables"]["stage"] = 999
        self.assertEqual(env.scenario["user_variables"]["stage"],1)
        reset.training_only_info["task_flags"]["stage"] = 99
        self.assertEqual(env._previous_flags["stage"], 0)

    def test_only_first_milestone_change_pays_bonus(self):
        env, coordinator = self.make_env()
        env.reset()
        coordinator.flags["USER4"] = 1.
        first = env.step(0,0)
        second = env.step(0,0)
        self.assertAlmostEqual(first.team_reward,.099)
        self.assertAlmostEqual(second.team_reward,-.001)
        self.assertEqual(first.rewards,(first.team_reward,first.team_reward))
        self.assertEqual(first.metadata.joint_transition_count,1)

    def test_success_death_and_missing_final_images_are_preserved(self):
        env, coordinator = self.make_env()
        env.reset()
        coordinator.flags["USER2"] = 1.
        coordinator.dead = coordinator.missing = True
        result = env.step(0,0)
        self.assertEqual(result.boundary.reason,BoundaryReason.DEATH)
        self.assertEqual(result.observations,(None,None))
        self.assertEqual(result.image_reconstruction_mask,(0.,0.))
        self.assertEqual(result.training_only_info["image_reconstruction_valid"],[False,False])
        self.assertAlmostEqual(result.team_reward,-1.001)
        with self.assertRaises(RuntimeError):
            env.step(0,0)
        fresh=env.reset()
        self.assertEqual(fresh.image_reconstruction_mask,(1.,1.))
        self.assertEqual(fresh.training_only_info["image_reconstruction_valid"],[True,True])
        self.assertEqual(result.observations,(None,None))
        self.assertEqual(result.image_reconstruction_mask,(0.,0.))

    def test_deadline_failure_and_success_at_deadline(self):
        for success in (False,True):
            env, coordinator = self.make_env()
            env.reset()
            from world_models.contracts import EpisodeClock
            env.clock = EpisodeClock(399,400)
            coordinator.flags["USER2"] = float(success)
            coordinator.flags["USER3"] = float(not success)
            result = env.step(0,0)
            expected = BoundaryReason.SUCCESS if success else BoundaryReason.DEADLINE
            self.assertEqual(result.boundary.reason,expected)
            self.assertEqual(result.boundary.bootstrap_mask,0.)

    def test_collection_cutoff_is_not_a_failure_reward(self):
        env, _ = self.make_env()
        env.reset()
        result = env.step(0,0,collection_cutoff=True)
        self.assertTrue(result.truncated)
        self.assertFalse(result.terminated)
        self.assertEqual(result.boundary.bootstrap_mask,1.)
        self.assertEqual(result.image_reconstruction_mask,(1.,1.))
        self.assertEqual(result.training_only_info["image_reconstruction_valid"],[True,True])
        self.assertTrue(all(frame is not None for frame in result.observations))
        self.assertAlmostEqual(result.team_reward,-.001)

    def test_one_missing_terminal_image_has_an_independent_loss_mask(self):
        env,coordinator=self.make_env()
        env.reset()
        coordinator.flags["USER2"]=1.
        coordinator.transform=lambda raw:replace(raw,participants=(replace(raw.participants[0],screen_buffer=None),raw.participants[1]))
        result=env.step(0,0)
        self.assertIsNone(result.observations[0])
        self.assertIsNotNone(result.observations[1])
        self.assertEqual(result.image_reconstruction_mask,(0.,1.))
        self.assertEqual(result.training_only_info["image_reconstruction_valid"],[False,True])
        self.assertEqual(result.boundary.bootstrap_mask,0.)

    def test_infrastructure_error_does_not_generate_a_transition_or_reward(self):
        env, coordinator = self.make_env()
        env.reset()
        coordinator.error = True
        with self.assertRaisesRegex(RuntimeError,"worker unavailable"):
            env.step(0,0)
        self.assertEqual(env.clock.decision_index,0)

    def test_milestone_regression_is_rejected(self):
        env, coordinator = self.make_env()
        env.reset()
        coordinator.flags["USER4"] = 1.
        env.step(0,0)
        coordinator.flags["USER4"] = 0.
        with self.assertRaisesRegex(RuntimeError,"regressed"):
            env.step(0,0)

    def test_integrity_faults_are_atomic_durable_and_require_fresh_recovery(self):
        def participants(raw, **changes):
            return replace(raw, participants=tuple(replace(p, **changes) for p in raw.participants))

        def disagree(raw):
            other = replace(raw.participants[1], variables={**raw.participants[1].variables, "USER2": 1.})
            return replace(raw, participants=(raw.participants[0], other))

        faults = {
            "task state disagreement": disagree,
            "nonintegral flag": lambda raw: participants(raw, variables={**raw.participants[0].variables, "USER2": .5}),
            "NaN variable": lambda raw: participants(raw, variables={**raw.participants[0].variables, "USER2": float("nan")}),
            "malformed image": lambda raw: participants(raw, screen_buffer=np.zeros((3, 4, 4), dtype=np.uint8)),
            "unexpected engine completion": lambda raw: participants(raw, finished=True),
            "malformed engine reward": lambda raw: participants(raw, engine_reward=float("nan")),
            "wrong episode": lambda raw: replace(raw, episode_id=raw.episode_id+1),
            "wrong dispatch": lambda raw: replace(raw, dispatch_id=raw.dispatch_id+1),
            "consumption mismatch": lambda raw: replace(raw, consumed_joint_decisions=raw.consumed_joint_decisions+1),
        }
        for name, fault in faults.items():
            with self.subTest(name=name):
                env, coordinator = self.make_env()
                env.reset()
                before = env._previous_raw
                coordinator.transform = fault
                with self.assertRaises(EnvironmentIntegrityError):
                    env.step(0, 0)
                self.assertEqual(env.state, LifecycleState.NEEDS_RECOVERY)
                self.assertEqual(env.clock.decision_index, 0)
                self.assertIs(env._previous_raw, before)
                self.assertEqual(len(coordinator.errors), 1)
                consumed = coordinator.count
                for retry in (lambda: env.step(0, 0), lambda: env.reset()):
                    with self.assertRaises(RuntimeError):
                        retry()
                    self.assertEqual(coordinator.count, consumed)
                coordinator.transform = None
                reset = env.recover()
                self.assertEqual(env.state, LifecycleState.READY)
                self.assertEqual(env.clock.decision_index, 0)
                self.assertNotEqual(reset.training_only_info["episode_id"], before.episode_id)
                self.assertTrue(reset.training_only_info["reset_actor_memories"])
                self.assertEqual(coordinator.count, consumed+1)
                self.assertEqual(env.step(0, 0).training_only_info["consumed_joint_decisions"], consumed+2)

    def test_counter_regression_cannot_consume_another_decision(self):
        env, coordinator = self.make_env()
        env.reset()
        coordinator.flags["USER4"] = 1.
        env.step(0, 0)
        coordinator.flags["USER4"] = 0.
        with self.assertRaises(EnvironmentIntegrityError):
            env.step(0, 0)
        coordinator.flags["USER4"] = 1.
        count = coordinator.count
        with self.assertRaises(RuntimeError):
            env.step(0, 0)
        self.assertEqual(coordinator.count, count)
        self.assertEqual(env.clock.decision_index, 1)

    def test_startup_and_recovery_image_failures_leave_environment_blocked(self):
        env, coordinator = self.make_env()
        coordinator.transform = lambda raw: replace(raw, participants=tuple(
            replace(p, screen_buffer=np.zeros((3, 4, 4), dtype=np.uint8)) for p in raw.participants)) if raw.decision_index > 0 else raw
        with self.assertRaises(EnvironmentIntegrityError):
            env.reset()
        self.assertEqual(env.state, LifecycleState.NEEDS_RECOVERY)
        self.assertIsNone(env._previous_raw)
        self.assertEqual(coordinator.count, 1)
        with self.assertRaises(EnvironmentIntegrityError):
            env.recover()
        self.assertEqual(env.state, LifecycleState.NEEDS_RECOVERY)
        self.assertEqual(coordinator.count, 2)
        self.assertEqual(env.initialization_joint_decisions, 2)

    def test_bad_initial_rgb_is_rejected_before_warmup_dispatch(self):
        env, coordinator = self.make_env()
        coordinator.transform = lambda raw: replace(raw,participants=tuple(
            replace(p,screen_buffer=np.zeros((3,4,4),dtype=np.uint8)) for p in raw.participants))
        with self.assertRaises(EnvironmentIntegrityError):
            env.reset()
        self.assertEqual(coordinator.count,0)
        self.assertEqual(env.initialization_joint_decisions,0)
        self.assertEqual(env.state,LifecycleState.NEEDS_RECOVERY)

    def test_invalid_caller_input_does_not_dispatch_or_break_ready_state(self):
        env, coordinator = self.make_env()
        env.reset()
        count = coordinator.count
        for args, kwargs in (((True,0),{}), ((0,12),{}), ((0,0),{"collection_cutoff":1})):
            with self.assertRaises(ValueError):
                env.step(*args, **kwargs)
            self.assertEqual(coordinator.count, count)
            self.assertEqual(env.state, LifecycleState.READY)
        with self.assertRaises(ValueError):
            env.reset(seed=-1)
        self.assertEqual(coordinator.count, count)

    def test_info_conversion_failure_is_atomic_and_ended_state_is_explicit(self):
        env, coordinator = self.make_env()
        env.reset()
        env.scenario["bad_metadata"] = float("nan")
        with self.assertRaises(EnvironmentIntegrityError):
            env.step(0, 0)
        self.assertEqual(env.clock.decision_index, 0)
        self.assertEqual(env.state, LifecycleState.NEEDS_RECOVERY)
        del env.scenario["bad_metadata"]
        env.recover()
        coordinator.flags["USER2"] = 1.
        env.step(0, 0)
        self.assertEqual(env.state, LifecycleState.EPISODE_ENDED)

    def test_declared_task_clock_registration_and_start_tic_are_validated(self):
        def flags_changed(raw, key, value):
            return replace(raw, participants=tuple(replace(p, variables={**p.variables,key:value}) for p in raw.participants))

        for key, value, pattern in (("USER9",0.,"elapsed"),("USER9",2.,"elapsed"),
                                    ("USER10",1.,"registered"),("USER11",8.,"start tic")):
            with self.subTest(key=key,value=value):
                env, coordinator = self.make_env()
                env.scenario["user_variables"].update(elapsed_decisions=9,start_tic=11)
                env.reset()
                coordinator.transform = lambda raw: flags_changed(raw,key,value)
                with self.assertRaisesRegex(EnvironmentIntegrityError,pattern):
                    env.step(0,0)
                self.assertEqual(env.clock.decision_index,0)
                self.assertEqual(env.state,LifecycleState.NEEDS_RECOVERY)
        env, coordinator = self.make_env()
        env.scenario["user_variables"]["elapsed_decisions"] = 9
        env.reset()
        coordinator.flags["USER2"] = 1.
        coordinator.transform = lambda raw: flags_changed(raw,"USER9",0.)
        self.assertTrue(env.step(0,0).terminated)  # ACS can freeze during the repeated action.

    def test_reset_requires_zero_task_counters_and_fresh_episode_metadata(self):
        for kind in ("stage", "elapsed", "progress", "decision"):
            with self.subTest(kind=kind):
                env, coordinator = self.make_env()
                env.scenario["user_variables"].update(elapsed_decisions=9,progress_1=5)
                def fault(raw):
                    if kind == "decision":
                        return replace(raw,decision_index=1)
                    key = {"stage":"USER1","elapsed":"USER9","progress":"USER5"}[kind]
                    return replace(raw,participants=tuple(replace(p,variables={**p.variables,key:1.}) for p in raw.participants))
                coordinator.transform = fault
                with self.assertRaises(EnvironmentIntegrityError):
                    env.reset()
                self.assertEqual(env.state,LifecycleState.NEEDS_RECOVERY)
        env, coordinator = self.make_env()
        env.reset()
        previous_episode = coordinator.episode
        coordinator.transform = lambda raw: replace(raw,episode_id=previous_episode)
        with self.assertRaisesRegex(EnvironmentIntegrityError,"old identifiers"):
            env.reset()

    def test_close_is_final_and_cannot_be_used_to_bypass_recovery(self):
        env, coordinator = self.make_env()
        env.reset()
        env.close()
        count = coordinator.count
        for operation in (lambda:env.step(0,0),lambda:env.reset(),lambda:env.recover()):
            with self.assertRaisesRegex(RuntimeError,"closed"):
                operation()
            self.assertEqual(coordinator.count,count)
        self.assertEqual(env.state,LifecycleState.CLOSED)

    def test_failed_reset_keeps_previous_committed_episode_until_recovery(self):
        env, coordinator = self.make_env()
        env.reset()
        env.step(0,0)
        previous = env._previous_raw
        coordinator.transform = lambda raw: replace(raw,participants=tuple(
            replace(p,screen_buffer=np.zeros((3,4,4),dtype=np.uint8)) for p in raw.participants)) if raw.decision_index > 0 else raw
        with self.assertRaises(EnvironmentIntegrityError):
            env.reset()
        self.assertIs(env._previous_raw,previous)
        self.assertEqual(env.clock.decision_index,1)
        count = coordinator.count
        with self.assertRaises(RuntimeError):
            env.step(0,0)
        self.assertEqual(coordinator.count,count)
        coordinator.transform = None
        reset = env.recover()
        self.assertEqual(env.clock.decision_index,0)
        self.assertGreater(reset.training_only_info["episode_id"],previous.episode_id)
        self.assertEqual(coordinator.count,count+1)


if __name__ == "__main__":
    unittest.main()
