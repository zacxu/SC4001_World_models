"""P3.3 world/agent stream separation and exact transactional resume."""
from copy import deepcopy
from pathlib import Path
import unittest

import torch

from world_models.infrastructure.config import resolve_config
from world_models.infrastructure.reproducibility import derive_seed
from world_models.policies import build_actor_critic
from world_models.rollouts.streams import RolloutStreams


ROOT = Path(__file__).resolve().parents[1]


def draw(streams):
    return {name: torch.rand(8, generator=generator)
            for name, generator in streams._generators().items()}


def assert_states_equal(test, left, right):
    test.assertEqual(left["identity"], right["identity"])
    test.assertEqual(left["schema_version"], right["schema_version"])
    for component in ("world", "actors"):
        test.assertEqual(set(left[component]), set(right[component]))
        for name in left[component]:
            test.assertTrue(torch.equal(left[component][name], right[component][name]), name)


class RolloutStreamTests(unittest.TestCase):
    def config(self, task="relay", variant="B"):
        return resolve_config(task, variant, "pilot", ROOT / "configs").settings

    def test_seed_reproducibility_and_all_named_streams_are_distinct(self):
        first, second = RolloutStreams.from_seed(33), RolloutStreams.from_seed(33)
        a, b = draw(first), draw(second)
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]))
            for other in a:
                if name != other:
                    self.assertFalse(torch.equal(a[name], a[other]))
        different = draw(RolloutStreams.from_seed(34))
        self.assertTrue(all(not torch.equal(a[name], different[name]) for name in a))
        self.assertEqual(first.identity["schema_version"], "p33-rollout-streams-v1")
        self.assertEqual(first.identity["device"], "cpu")

    def test_agent_and_posterior_draws_cannot_advance_other_components(self):
        changed, reference = RolloutStreams.from_seed(9), RolloutStreams.from_seed(9)
        torch.rand(29, generator=changed.actors[0])
        torch.rand(43, generator=changed.world.posterior)
        a, b = draw(changed), draw(reference)
        self.assertFalse(torch.equal(a["actor_0"], b["actor_0"]))
        self.assertFalse(torch.equal(a["posterior"], b["posterior"]))
        for name in ("actor_1", "prior", "outcome"):
            self.assertTrue(torch.equal(a[name], b[name]))

    def test_creation_sampling_serialization_and_restore_preserve_global_cpu_rng(self):
        before = torch.random.get_rng_state().clone()
        streams = RolloutStreams.from_seed(5)
        experiment = RolloutStreams.from_experiment(self.config(), master_seed=0)
        state = streams.state_dict()
        draw(streams)
        streams.load_state_dict(state)
        draw(experiment)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_state_resume_exactly_replays_all_components(self):
        streams = RolloutStreams.from_seed(83)
        draw(streams)
        state = streams.state_dict()
        expected = draw(streams)
        resumed = RolloutStreams.from_seed(83)
        resumed.load_state_dict(state)
        actual = draw(resumed)
        for name in expected:
            self.assertTrue(torch.equal(actual[name], expected[name]), name)

    def test_state_exports_and_restore_inputs_have_no_storage_aliases(self):
        streams = RolloutStreams.from_seed(11)
        pristine, state = streams.state_dict(), streams.state_dict()
        state["identity"]["seeds"]["world"] = 0
        state["world"]["prior"].zero_()
        state["actors"]["1"].zero_()
        assert_states_equal(self, streams.state_dict(), pristine)
        restored = deepcopy(pristine)
        streams.load_state_dict(restored)
        restored["world"]["outcome"].zero_()
        restored["actors"]["0"].zero_()
        assert_states_equal(self, streams.state_dict(), pristine)

    def test_malformed_last_state_cannot_partly_restore_any_component(self):
        streams = RolloutStreams.from_seed(8)
        saved = streams.state_dict()
        draw(streams)
        live = streams.state_dict()
        saved["actors"]["1"] = torch.zeros(3, dtype=torch.uint8)
        with self.assertRaisesRegex(ValueError, "malformed actor_1"):
            streams.load_state_dict(saved)
        assert_states_equal(self, streams.state_dict(), live)

    def test_strict_state_schema_types_and_identity_reject_without_mutation(self):
        streams = RolloutStreams.from_seed(32)
        state = streams.state_dict()
        bad_states = []
        for key in state:
            value = deepcopy(state)
            del value[key]
            bad_states.append(value)
        for section, key in (("world", "prior"), ("actors", "1")):
            for replacement in ([], torch.zeros(8), torch.zeros((2, 4), dtype=torch.uint8)):
                value = deepcopy(state)
                value[section][key] = replacement
                bad_states.append(value)
            value = deepcopy(state)
            value[section]["unexpected"] = torch.zeros(8, dtype=torch.uint8)
            bad_states.append(value)
        bad_states.extend([RolloutStreams.from_seed(33).state_dict()])
        value = deepcopy(state)
        value["identity"]["device"] = "cuda:0"
        bad_states.append(value)
        for bad in bad_states:
            with self.subTest(bad_identity=bad.get("identity")):
                with self.assertRaises(ValueError):
                    streams.load_state_dict(bad)
                assert_states_equal(self, streams.state_dict(), state)

    def test_valid_byte_state_from_wrong_seed_is_rejected_transactionally(self):
        streams = RolloutStreams.from_seed(19)
        state = streams.state_dict()
        bad = deepcopy(state)
        bad["actors"]["1"] = RolloutStreams.from_seed(20).actors[1].get_state()
        with self.assertRaisesRegex(ValueError, "different component seed"):
            streams.load_state_dict(bad)
        assert_states_equal(self, streams.state_dict(), state)

    def test_constructor_rejects_aliasing_seed_disagreement_and_identity_changes(self):
        streams = RolloutStreams.from_seed(24)
        for actors in ((streams.actors[0], streams.actors[0]),
                       (streams.world.prior, streams.actors[1])):
            with self.assertRaisesRegex(ValueError, "alias"):
                RolloutStreams(streams.world, actors, streams.identity)
        with self.assertRaisesRegex(ValueError, "seed"):
            RolloutStreams(streams.world,
                           (torch.Generator().manual_seed(1), streams.actors[1]), streams.identity)
        streams.identity["seeds"]["actor_0"] += 1
        with self.assertRaisesRegex(ValueError, "identity"):
            streams.validate()

    def test_validate_requested_device_and_reject_unsupported_generator_device(self):
        streams = RolloutStreams.from_seed(21)
        streams.validate(device="cpu")
        streams.validate(device=torch.device("cpu:0"))
        with self.assertRaisesRegex(ValueError, "device"):
            streams.validate(device="cuda:0")
        with self.assertRaisesRegex(ValueError, "CPU or CUDA"):
            RolloutStreams.from_seed(21, device="meta")

    def test_experiment_namespaces_are_independent_with_paired_initial_weights(self):
        before = torch.random.get_rng_state().clone()
        left_cfg, right_cfg = self.config(variant="B"), self.config(variant="D")
        left = RolloutStreams.from_experiment(left_cfg, master_seed=1)
        right = RolloutStreams.from_experiment(right_cfg, master_seed=1)
        self.assertEqual(left.identity["seeds"]["world"], derive_seed("relay", 1, "imagination", "B"))
        self.assertEqual(left.identity["seeds"]["actor_0"], derive_seed("relay", 1, "local_action_sampling", "B", 0))
        left_draw, right_draw = draw(left), draw(right)
        self.assertTrue(all(not torch.equal(left_draw[name], right_draw[name]) for name in left_draw))
        networks = [build_actor_critic(cfg, master_seed=1, variant=cfg["variant"]["name"])
                    for cfg in (left_cfg, right_cfg)]
        self.assertTrue(all(torch.equal(value, networks[1].state_dict()[name])
                            for name, value in networks[0].state_dict().items()))
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_experiment_seed_task_and_config_guards(self):
        reference = RolloutStreams.from_experiment(self.config(), master_seed=0).identity
        for task, seed in (("cover", 0), ("relay", 1)):
            changed = RolloutStreams.from_experiment(self.config(task), master_seed=seed).identity
            self.assertNotEqual(reference["seeds"], changed["seeds"])
        for seed in (-1, True, 2**63, 1.5):
            with self.assertRaises(ValueError):
                RolloutStreams.from_seed(seed)
            with self.assertRaises(ValueError):
                RolloutStreams.from_experiment(self.config(), master_seed=seed)
        bad = self.config()
        bad["environment"]["actor_global_input"] = True
        with self.assertRaises(ValueError):
            RolloutStreams.from_experiment(bad, master_seed=0)


if __name__ == "__main__":
    unittest.main()
