"""Numerical and recurrent fitting checks using tiny synthetic replay only.

These tests use synthetic fixtures for forwards, backpropagation, and one
numerical optimizer step. They do not perform a research fitting run.
"""

from copy import deepcopy
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import torch

from tests.test_model_objective_accumulation import _fixture, _slice_batch
from tests.test_model_sampling import identifier, make_store
from world_models.models.core import WorldState
from world_models.models.heads import AuxiliaryTarget
from world_models.models.world_model import SamplingStreams, SharedWorldModel
from world_models.training.fitting import (
    FitSettings, LossSupport, MetricAccumulator, accumulation_loss, compare_burn_in,
    evaluate_windows, forward_windows, initial_window_state, loss_support, microbatches,
    optimizer_update,
)
from world_models.training.normalization import fit_normalization
from world_models.training.objective import WorldModelObjective, merge_objective_results
from world_models.training.sampling import ReplaySampler


class FittingMathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_accumulation_matches_global_objective_and_gradients_with_disjoint_support(self):
        states, transitions, objective = _fixture()
        # Each health field is observed in a different microbatch. Means of
        # per-microbatch auxiliary means would assign incorrect field weights.
        for name, mask in (("player_0/health", [True, False, False, False]),
                           ("player_1/health", [False, True, True, True])):
            states.targets[name] = AuxiliaryTarget(torch.zeros((1, 4)), torch.tensor([mask]))
        states.auxiliary["player_1/health"] = torch.tensor([[0., 2., 3., 4.]], requires_grad=True)
        event_name = next(iter(transitions.auxiliary))
        transitions.auxiliary_targets[event_name] = AuxiliaryTarget(
            torch.ones((1, 4)), torch.tensor([[True, False, False, False]]))
        states.observed[0, 3] = False
        states.image_valid[0, 3, 1] = False
        transitions.posterior_valid[0, 2] = False
        left = (_slice_batch(states, 0, 1), _slice_batch(transitions, 0, 1))
        right = (_slice_batch(states, 1, 4), _slice_batch(transitions, 1, 4))
        production = [objective(*left), objective(*right)]
        support = LossSupport.from_results(production)
        reference = objective(states, transitions)
        merged = merge_objective_results(production)
        contributions = [accumulation_loss(result, support) for result in production]
        torch.testing.assert_close(sum(contributions), merged.total, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(sum(contributions), reference.total, rtol=1e-6, atol=1e-6)
        self.assertEqual(support.fields["player_0/health"], 1)
        self.assertEqual(support.fields["player_1/health"], 3)
        self.assertEqual(support.fields[event_name], 1)
        self.assertEqual(support.components["rgb"], 3*2*3*96*128)
        self.assertEqual(support.components["reward"], 3)
        self.assertEqual(support.components["outcome"], 3)
        self.assertEqual(support.components["dynamics_kl"], 3)
        for contribution in contributions:
            contribution.backward()
        reference.total.backward()
        paths = [
            (states.rgb, left[0].rgb, right[0].rgb),
            (transitions.bonus, left[1].bonus, right[1].bonus),
            (transitions.outcome_logits, left[1].outcome_logits, right[1].outcome_logits),
            (transitions.prior_logits, left[1].prior_logits, right[1].prior_logits),
            (transitions.posterior_logits, left[1].posterior_logits, right[1].posterior_logits),
            (states.auxiliary["player_0/health"], left[0].auxiliary["player_0/health"], right[0].auxiliary["player_0/health"]),
            (states.auxiliary["player_1/health"], left[0].auxiliary["player_1/health"], right[0].auxiliary["player_1/health"]),
            (transitions.auxiliary[event_name], left[1].auxiliary[event_name], right[1].auxiliary[event_name]),
        ]
        for full, first, remaining in paths:
            self.assertTrue(torch.isfinite(full.grad).all())
            torch.testing.assert_close(full.grad, torch.cat((first.grad, remaining.grad), dim=1), rtol=1e-6, atol=1e-6)
        self.assertGreater(float(transitions.prior_logits.grad.abs().sum()), 0.)
        self.assertGreater(float(transitions.posterior_logits.grad.abs().sum()), 0.)
        self.assertGreater(float(states.auxiliary["player_0/health"].grad.abs().sum()), 0.)
        self.assertGreater(float(states.auxiliary["player_1/health"].grad.abs().sum()), 0.)

        accumulator = MetricAccumulator()
        for result in production:
            accumulator.add(result)
        metrics = accumulator.metrics()
        self.assertAlmostEqual(metrics["total"], float(reference.total.detach()), places=5)
        for name, term in reference.components.items():
            self.assertEqual(metrics["components"][name]["count"], int(term.count))
            self.assertAlmostEqual(metrics["components"][name]["mean"], float(term.mean.detach()), places=5)

    def test_grouping_preserves_every_unpadded_window_and_its_arrays(self):
        windows = [SimpleNamespace(burn_in=burn, loss_steps=steps, states=object())
                   for burn, steps in ((0, 1), (2, 3), (0, 1), (1, 1), (2, 3), (0, 1))]
        batches = list(microbatches(windows, 2))
        self.assertEqual(sorted(id(window) for batch in batches for window in batch), sorted(map(id, windows)))
        self.assertEqual(sorted(map(len, batches)), [1, 1, 2, 2])
        for batch in batches:
            self.assertEqual(len({(window.burn_in, window.loss_steps) for window in batch}), 1)
            self.assertTrue(all(any(window is original for original in windows) for window in batch))

    def test_burn_comparison_keeps_anchors_and_loss_targets_fixed(self):
        windows = [SimpleNamespace(dataset_index=0, episode_id="validation-a", anchor=80, loss_steps=48),
                   SimpleNamespace(dataset_index=0, episode_id="validation-b", anchor=96, loss_steps=25)]
        bank = SimpleNamespace(window=Mock(side_effect=lambda index, episode, anchor, **kwargs:
                              SimpleNamespace(dataset_index=index, episode_id=episode, anchor=anchor, **kwargs)))
        observations = []
        def evaluation(model, selected, objective, **kwargs):
            observations.append((selected, kwargs))
            return {"total": float(selected[0].burn_in)}
        with patch("world_models.training.fitting.evaluate_windows", side_effect=evaluation):
            result = compare_burn_in(object(), bank, windows, object(), seed=45)
        self.assertEqual(set(result["context_results"]), {"16", "32", "64", "full_prefix"})
        self.assertEqual(result["optimizer_updates"], 0)
        self.assertEqual(bank.window.call_count, 8)
        for run, context in zip(observations, (16, 32, 64, "full_prefix")):
            selected, kwargs = run
            self.assertEqual([(w.episode_id, w.anchor, w.loss_steps) for w in selected],
                             [(w.episode_id, w.anchor, w.loss_steps) for w in windows])
            self.assertEqual(kwargs["seed"], 45)
            self.assertTrue(all(w.full_prefix == (context == "full_prefix") for w in selected))
            self.assertTrue(all(w.burn_in == (0 if context == "full_prefix" else context) for w in selected))


class FittingUnrollTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.fit = identifier("fitting-unroll", "fit")
        cls.dataset = make_store(Path(cls.temporary.name)/"replay", [(cls.fit, 7, True)])
        cls.sampler = ReplaySampler([cls.dataset], seed=10, burn_in=1, loss_steps=2)
        cls.normalization = fit_normalization(cls.dataset)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(48)
            self.model = SharedWorldModel()
        self.objective = WorldModelObjective(self.model.task, self.normalization)

    def test_label_support_equals_real_results_with_unequal_context_and_terminal_masks(self):
        windows = [self.sampler.window(0, self.fit, 0, burn_in=0, loss_steps=1),
                   self.sampler.window(0, self.fit, 3, burn_in=2, loss_steps=3),
                   self.sampler.window(0, self.fit, 6, burn_in=3, loss_steps=2)]
        with torch.no_grad():
            results = [forward_windows(self.model, [window], SamplingStreams.from_seed(19), self.objective)
                       for window in windows]
        actual = loss_support(windows, self.model.task, windows[0].event_fields)
        expected = LossSupport.from_results(results)
        self.assertEqual(actual.components, expected.components)
        self.assertEqual(actual.fields, expected.fields)
        self.assertEqual(actual.components["rgb"], 7*2*3*96*128)
        self.assertEqual(actual.components["dynamics_kl"], 4)
        self.assertEqual(actual.components["reward"], 5)
        self.assertEqual(actual.components["outcome"], 5)
        # Terminal privileged labels remain valid even without both RGB views.
        self.assertEqual(actual.fields["player_0/health"], 8)

    def test_initial_context_uses_absolute_clock_and_exactly_one_posterior_draw(self):
        window = self.sampler.window(0, self.fit, 4, burn_in=1, loss_steps=1)
        images = torch.tensor(window.states["rgb"][0:1])
        decisions = torch.tensor([window.start], dtype=torch.long)
        streams, reference_streams = SamplingStreams.from_seed(24), SamplingStreams.from_seed(24)
        before = streams.state_dict()
        with torch.no_grad(), patch.object(self.model.core, "reset", side_effect=AssertionError("extra reset draw")):
            actual = initial_window_state(self.model, images, decisions, streams)
            memory = self.model.core.reset_memory.unsqueeze(0)
            stochastic = torch.zeros((1, 32, 32))
            stochastic[..., 0] = 1
            reference = self.model.core.observe(WorldState(memory, stochastic, torch.zeros_like(stochastic), decisions),
                images.float()/255, generator=reference_streams.posterior)
        self.assertEqual(actual.decision_index.tolist(), [3])
        for name in ("memory", "stochastic", "logits", "decision_index"):
            torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=0, atol=0)
        self.assertTrue(torch.equal(streams.posterior.get_state(), reference_streams.posterior.get_state()))
        self.assertTrue(torch.equal(streams.prior.get_state(), before["prior"]))
        self.assertTrue(torch.equal(streams.outcome.get_state(), before["outcome"]))

    def test_burn_in_has_no_graph_and_reset_examples_train_reset_memory(self):
        window = self.sampler.window(0, self.fit, 2, burn_in=1, loss_steps=1)
        grad_modes, reconstruction_states = [], []
        observe, reconstruct = self.model.core.observe, self.model.reconstruct
        def tracked_observe(*args, **kwargs):
            grad_modes.append(torch.is_grad_enabled())
            return observe(*args, **kwargs)
        def tracked_reconstruct(state):
            reconstruction_states.append(state.memory.requires_grad)
            return reconstruct(state)
        with patch.object(self.model.core, "observe", side_effect=tracked_observe), \
             patch.object(self.model, "reconstruct", side_effect=tracked_reconstruct):
            result = forward_windows(self.model, [window], SamplingStreams.from_seed(32), self.objective)
            result.total.backward()
        self.assertEqual(grad_modes, [False, False, True])
        self.assertEqual(reconstruction_states[0], False)
        self.assertIsNone(self.model.core.reset_memory.grad)
        self.assertGreater(float(self.model.decoder.agent_queries.grad.abs().sum()), 0.)
        self.model.zero_grad(set_to_none=True)
        reset = self.sampler.window(0, self.fit, 0, burn_in=0, loss_steps=1)
        forward_windows(self.model, [reset], SamplingStreams.from_seed(32), self.objective).total.backward()
        self.assertIsNotNone(self.model.core.reset_memory.grad)
        self.assertTrue(torch.isfinite(self.model.core.reset_memory.grad).all())
        self.assertGreater(float(self.model.core.reset_memory.grad.abs().sum()), 0.)

    def test_cpu_float32_and_bfloat16_forward_backward_are_finite(self):
        window = self.sampler.window(0, self.fit, 1, burn_in=0, loss_steps=1)
        for precision in ("float32", "bfloat16"):
            with self.subTest(precision=precision):
                self.model.zero_grad(set_to_none=True)
                result = forward_windows(self.model, [window], SamplingStreams.from_seed(57), self.objective,
                                         precision=precision)
                self.assertEqual(result.total.dtype, torch.float32)
                self.assertTrue(torch.isfinite(result.total))
                result.total.backward()
                gradients = [parameter.grad for parameter in self.model.parameters() if parameter.grad is not None]
                self.assertTrue(gradients)
                self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))
                self.assertGreater(sum(float(gradient.abs().sum()) for gradient in gradients), 0.)

    def test_synthetic_optimizer_step_accumulates_clips_and_keeps_fp32_moments(self):
        # Equal shapes can still have unequal valid-label support: the second
        # window ends with a missing required terminal RGB view.
        windows = [self.sampler.window(0, self.fit, anchor, burn_in=1, loss_steps=1)
                   for anchor in (1, 6)]
        settings = FitSettings(effective_batch=2, microbatch=1, burn_in=1, loss_steps=1,
                               precision="float32", gradient_clip=.05)
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=settings.learning_rate,
                                      weight_decay=settings.weight_decay)
        before = self.model.decoder.agent_queries.detach().clone()
        original_clip = torch.nn.utils.clip_grad_norm_
        clipped_norms = []
        def tracked_clip(parameters, *args, **kwargs):
            parameters = list(parameters)
            norm = original_clip(parameters, *args, **kwargs)
            clipped_norms.append(torch.linalg.vector_norm(torch.stack([
                parameter.grad.detach().float().norm() for parameter in parameters if parameter.grad is not None])))
            return norm
        with patch("world_models.training.fitting.torch.nn.utils.clip_grad_norm_", side_effect=tracked_clip) as clip:
            metrics = optimizer_update(self.model, optimizer, windows, SamplingStreams.from_seed(86),
                                       self.objective, settings)
        self.assertEqual(clip.call_count, 1)
        self.assertTrue(torch.isfinite(torch.tensor(metrics["gradient_norm_before_clip"])))
        self.assertGreater(metrics["gradient_norm_before_clip"], settings.gradient_clip)
        self.assertLessEqual(float(clipped_norms[0]), settings.gradient_clip*(1+1e-5))
        self.assertFalse(torch.equal(before, self.model.decoder.agent_queries.detach()))
        self.assertTrue(optimizer.state)
        for state in optimizer.state.values():
            self.assertEqual(float(state["step"]), 1.)
            for name in ("exp_avg", "exp_avg_sq"):
                self.assertEqual(state[name].dtype, torch.float32)
                self.assertTrue(torch.isfinite(state[name]).all())
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.assertEqual((metrics["sequences"], metrics["loss_transitions"], metrics["burn_in_transitions"]), (2, 2, 2))
        self.assertEqual(metrics["components"]["rgb"]["count"], 3*2*3*96*128)
        self.assertEqual(metrics["components"]["dynamics_kl"]["count"], 1)
        self.assertEqual(metrics["components"]["outcome"]["count"], 2)

    def test_evaluation_preserves_weights_mode_and_independent_training_rng(self):
        window = self.sampler.window(0, self.fit, 2, burn_in=1, loss_steps=1)
        fit_streams = SamplingStreams.from_seed(67)
        fit_state = fit_streams.state_dict()
        sampler_state = deepcopy(self.sampler.state_dict())
        global_rng = torch.random.get_rng_state().clone()
        parameters = {name: value.detach().clone() for name, value in self.model.state_dict().items()}
        first = evaluate_windows(self.model, [window], self.objective, seed=81)
        self.assertTrue(self.model.training)
        self.model.eval()
        second = evaluate_windows(self.model, [window], self.objective, seed=81)
        self.assertFalse(self.model.training)
        self.assertEqual(first, second)
        self.assertEqual(self.sampler.state_dict(), sampler_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), global_rng))
        self.assertTrue(all(torch.equal(fit_streams.state_dict()[key], value) for key, value in fit_state.items()))
        self.assertTrue(all(torch.equal(value, self.model.state_dict()[name]) for name, value in parameters.items()))
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))
        self.model.train()
        with patch("world_models.training.fitting.forward_windows", side_effect=RuntimeError("diagnostic failure")):
            with self.assertRaisesRegex(RuntimeError, "diagnostic failure"):
                evaluate_windows(self.model, [window], self.objective, seed=81)
        self.assertTrue(self.model.training)


if __name__ == "__main__":
    unittest.main()
