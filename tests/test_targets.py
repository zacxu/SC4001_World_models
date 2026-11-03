"""Privileged target alignment, masking, event retention and reward isolation."""

from copy import deepcopy
from dataclasses import replace
import json
import math
from pathlib import Path
import unittest

import numpy as np

from world_models.contracts import BoundaryReason, EpisodeBoundary, RewardRule
from world_models.env.coop import EnvironmentIntegrityError, LifecycleState, TwoPlayerCoopEnv
from world_models.env.coordinator import CoordinatorError, JointRawResult
from world_models.env.scenario_builder import USER_VARIABLES, _parameters
from world_models.env.targets import (
    EVENT_FIELDS, NATIVE_COUNTERS, USER_FIELDS, authored_objects, build_privileged_targets,
    decode_user_field, event_snapshot, native_counter_baseline, registry_manifest, reward_components,
)
from world_models.env.workers import RawParticipantState


def fixture(family="relay"):
    scenario,_,_ = _parameters(family,0)
    scenario.update(seed=0,frame_skip=4,wad_path="unused.wad",user_variables=dict(USER_FIELDS),
                    privileged_registry_version=registry_manifest()["schema_version"],
                    privileged_registry_sha256=registry_manifest()["sha256"],
                    user_variable_encodings={name:"acs_fixed_16_16" for name in ("enemy_1_x","enemy_1_y","enemy_2_x","enemy_2_y")})
    scenario["authored_objects"] = authored_objects(scenario)
    flags = {name:0 for name in USER_FIELDS}
    flags.update(registered_players=2,start_tic=4,player_0_alive=1,player_1_alive=1)
    if family == "cover":
        flags.update(enemy_1_x=704.5,enemy_1_y=192.25,enemy_1_alive=1,enemy_1_health=20,enemy_health=20)
    native = {"POSITION_X":1.5,"POSITION_Y":2.25,"POSITION_Z":0.,"ANGLE":90.,"VELOCITY_X":1.25,"VELOCITY_Y":-2.5,
              "HEALTH":100.,"ARMOR":100.,"SELECTED_WEAPON_AMMO":200.,**{name:0. for name in NATIVE_COUNTERS}}
    wire = {f"USER{index}":float(flags[name]*65536 if name in scenario["user_variable_encodings"] else flags[name]) for name,index in USER_FIELDS.items()}
    participants = tuple(RawParticipantState(i,5,np.zeros((3,120,160),dtype=np.uint8),{**native,**wire}) for i in (0,1))
    return scenario,flags,JointRawResult(0,1,5,participants,1)


class ModernCoordinator:
    def __init__(self):
        self.scenario,self.flags,self.base = fixture()
        self.episode=0; self.index=0; self.dispatch=0; self.count=0
        self.broken=False; self.transform=None; self.errors=[]

    def result(self):
        wire = {f"USER{index}":float(self.flags[name]) for name,index in USER_FIELDS.items()}
        raw = JointRawResult(self.episode,self.index,self.dispatch,tuple(replace(p,tic=1+4*self.index,variables={**p.variables,**wire}) for p in self.base.participants),self.count)
        return self.transform(raw) if self.transform else raw

    def reset(self,seed=None):
        self.episode+=1; self.index=0; self.dispatch+=1
        for name in EVENT_FIELDS.values(): self.flags[name]=0
        self.flags.update(stage=0,success=0,failure=0,milestone_count=0,elapsed_decisions=0,latch_active=0,plate_active=0)
        return self.result()

    def step(self,actions):
        if self.broken: raise RuntimeError("recover first")
        self.count+=1; self.index+=1; self.dispatch+=4
        self.flags["elapsed_decisions"]=self.index-1
        return self.result()

    def invalidate(self,message,details=None):
        self.broken=True
        record={"message":message,"details":details,"consumed_joint_decisions":self.count}
        self.errors.append(record)
        return CoordinatorError(message,record)

    def recover(self):
        self.broken=False
        return self.reset()

    def close(self): pass


class PrivilegedRegistryTests(unittest.TestCase):
    def test_exported_registry_and_builder_mapping_are_exactly_versioned(self):
        stored=json.loads(Path("configs/privileged_registry_v1.json").read_text())
        self.assertEqual(stored,registry_manifest())
        self.assertEqual(USER_FIELDS,USER_VARIABLES)
        self.assertEqual(sorted(USER_FIELDS.values()),list(range(1,41)))
        self.assertEqual(stored["reward_reservation"]["acs_global"],0)
        self.assertEqual(stored["reward_reservation"]["value"],0.)

    def test_fixed_point_wire_conversion_matches_engine_utility(self):
        import vizdoom
        scenario,_,_=fixture("cover")
        for value in (0,98304,-98304,46137344):
            self.assertEqual(decode_user_field("enemy_1_x",float(value),scenario),vizdoom.doom_fixed_to_float(value))
        self.assertEqual(decode_user_field("stage",1.,scenario),1)
        for value in (.5,float("nan")):
            with self.assertRaises(ValueError): decode_user_field("enemy_1_x",value,scenario)

    def test_stable_authored_ids_survive_role_and_layout_randomization(self):
        for family in ("relay","cover"):
            identities=[]
            for seed in range(8):
                scenario,_,_=_parameters(family,seed)
                objects=authored_objects(scenario)
                identities.append({obj["object_id"]:obj.get("engine_tid",obj.get("sector_tag")) for obj in objects})
            self.assertTrue(all(value==identities[0] for value in identities))
            self.assertEqual(identities[0]["player_0"],100)
            self.assertEqual(identities[0]["player_1"],101)
            if family == "cover": self.assertEqual([identities[0]["enemy_1"],identities[0]["enemy_2"]],[200,201])

    def test_heading_normalization_masks_and_geometry_extent(self):
        scenario,flags,raw=fixture("cover")
        scenario["zones"]["extraction"][2]=1536
        scenario["authored_objects"]=authored_objects(scenario)
        targets=build_privileged_targets(raw,flags,scenario)
        player=targets["players"][0]
        self.assertAlmostEqual(player["raw"]["heading_sin"],1.)
        self.assertAlmostEqual(player["raw"]["heading_cos"],0.)
        self.assertAlmostEqual(player["normalized"]["position_x"],2*1.5/1536-1)
        self.assertEqual(player["normalized"]["velocity_y"],-2.5/32)
        self.assertFalse(targets["task"]["valid"]["plate_active"])
        self.assertIsNone(targets["task"]["normalized"]["plate_active"])
        enemy2=next(obj for obj in targets["objects"] if obj["object_id"]=="enemy_2")
        self.assertFalse(enemy2["state"]["valid"]["position_x"])
        self.assertIsNone(enemy2["state"]["normalized"]["position_x"])

    def test_terminal_rgb_absence_does_not_erase_valid_state_targets(self):
        scenario,flags,raw=fixture()
        raw=replace(raw,participants=tuple(replace(p,screen_buffer=None,dead=True,variables={**p.variables,"HEALTH":-8.,"SELECTED_WEAPON_AMMO":-1.}) for p in raw.participants))
        player=build_privileged_targets(raw,flags,scenario)["players"][0]
        self.assertEqual(player["raw"]["alive"],0)
        self.assertEqual(player["normalized"]["health"],0)
        self.assertTrue(player["valid"]["position_x"])
        self.assertFalse(player["valid"]["ammo"])
        self.assertIsNone(player["normalized"]["ammo"])

    def test_missing_declared_state_or_mismatched_registry_is_rejected(self):
        scenario,flags,raw=fixture()
        bad=replace(raw,participants=tuple(replace(p,variables={k:v for k,v in p.variables.items() if k!="POSITION_X"}) for p in raw.participants))
        with self.assertRaisesRegex(ValueError,"missing declared"): build_privileged_targets(bad,flags,scenario)
        scenario["privileged_registry_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"registry version/hash"): build_privileged_targets(raw,flags,scenario)

    def test_wrong_wire_encoding_and_ambiguous_object_ids_are_rejected(self):
        scenario,flags,raw=fixture()
        scenario["user_variable_encodings"]={}
        with self.assertRaisesRegex(ValueError,"wire encodings"): build_privileged_targets(raw,flags,scenario)
        scenario,flags,raw=fixture()
        scenario["authored_objects"][0]["engine_tid"]=101
        with self.assertRaisesRegex(ValueError,"object IDs"): build_privileged_targets(raw,flags,scenario)


class EventTests(unittest.TestCase):
    def test_transient_plate_and_gate_edges_survive_same_final_state(self):
        scenario,flags,raw=fixture()
        before=event_snapshot(raw,flags,scenario)
        flags.update(plate_entry_count=1,plate_exit_count=1,gate_open_count=1,gate_close_count=1,plate_active=0,gate_1_open=0)
        after=event_snapshot(raw,flags,scenario,before["cumulative"])
        self.assertEqual([after["deltas"][key] for key in ("plate_entries","plate_exits","gate_opens","gate_closes")],[1,1,1,1])
        self.assertEqual(flags["plate_active"],0)

    def test_multiple_events_and_exact_shared_reward_decomposition(self):
        scenario,flags,raw=fixture("cover")
        before=event_snapshot(raw,flags,scenario)
        flags.update(milestone_count=2,console_1_completion_count=1,console_2_completion_count=1,stage=2)
        events=event_snapshot(raw,flags,scenario,before["cumulative"])
        bonus=.05*(events["deltas"]["console_1_completions"]+events["deltas"]["console_2_completions"])
        rule=RewardRule()
        boundary=EpisodeBoundary()
        components=reward_components(rule,boundary,bonus,rule.reward(boundary,bonus))
        self.assertAlmostEqual(components["total"],.099)
        self.assertEqual(components["shared_rewards"],[components["total"]]*2)

    def test_native_counters_are_episode_relative_and_death_cannot_be_lost(self):
        scenario,flags,raw=fixture()
        raw=replace(raw,participants=tuple(replace(p,variables={**p.variables,"DEATHCOUNT":5.}) for p in raw.participants))
        baseline=native_counter_baseline(raw)
        before=event_snapshot(raw,flags,scenario,native_baseline=baseline)
        self.assertEqual(before["cumulative"]["player_0_deaths"],0)
        after_raw=replace(raw,participants=(replace(raw.participants[0],dead=True,variables={**raw.participants[0].variables,"DEATHCOUNT":6.}),raw.participants[1]))
        after=event_snapshot(after_raw,flags,scenario,before["cumulative"],native_baseline=baseline,boundary=EpisodeBoundary(BoundaryReason.DEATH))
        self.assertEqual(after["deltas"]["player_0_deaths"],1)
        self.assertEqual(after["deltas"]["player_0_deathcount"],1)
        self.assertEqual(after["boundary_events"]["death"],1)
        # Even if the engine's monitor counter has not updated on this final
        # sub-tic, the irreversible authoritative dead flag retains the event.
        delayed=replace(raw,participants=(replace(raw.participants[0],dead=True),raw.participants[1]))
        self.assertEqual(event_snapshot(delayed,flags,scenario,before["cumulative"],native_baseline=baseline)["deltas"]["player_0_deaths"],1)

    def test_counter_regression_and_milestone_disagreement_rejected(self):
        scenario,flags,raw=fixture()
        flags["plate_entry_count"]=1
        before=event_snapshot(raw,flags,scenario)
        flags["plate_entry_count"]=0
        with self.assertRaisesRegex(ValueError,"regressed"): event_snapshot(raw,flags,scenario,before["cumulative"])
        flags["milestone_count"]=1
        with self.assertRaisesRegex(ValueError,"milestone counter"): event_snapshot(raw,flags,scenario)

    def test_wrapper_event_failures_are_atomic_and_retry_consumes_nothing(self):
        coordinator=ModernCoordinator()
        env=TwoPlayerCoopEnv("relay",scenario_manifest=coordinator.scenario,coordinator=coordinator)
        env.reset()
        coordinator.flags["plate_entry_count"]=1
        self.assertEqual(env.step(0,0).training_only_info["events"]["deltas"]["plate_entries"],1)
        previous=deepcopy(env._previous_events)
        coordinator.flags["plate_entry_count"]=0
        with self.assertRaisesRegex(EnvironmentIntegrityError,"regressed"): env.step(0,0)
        self.assertEqual(env.clock.decision_index,1)
        self.assertEqual(env._previous_events,previous)
        self.assertEqual(env.state,LifecycleState.NEEDS_RECOVERY)
        consumed=coordinator.count
        with self.assertRaises(RuntimeError): env.step(0,0)
        self.assertEqual(coordinator.count,consumed)

    def test_reserved_engine_reward_cannot_double_count_shared_reward(self):
        coordinator=ModernCoordinator()
        env=TwoPlayerCoopEnv("relay",scenario_manifest=coordinator.scenario,coordinator=coordinator)
        reset=env.reset()
        self.assertFalse(hasattr(reset.local_input(0,12,(0.,)*256),"privileged_targets"))
        coordinator.transform=lambda raw:replace(raw,participants=tuple(replace(p,engine_reward=.1) for p in raw.participants))
        with self.assertRaisesRegex(EnvironmentIntegrityError,"reserved engine reward"): env.step(0,0)
        self.assertEqual(env.clock.decision_index,0)


if __name__ == "__main__": unittest.main()
