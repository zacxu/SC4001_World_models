"""P3.2 cache-version invariants using tiny, untrained tensor fixtures."""
from dataclasses import dataclass, replace
import unittest

import torch
from torch import nn

from world_models.imagination.versions import module_version


@dataclass(frozen=True)
class Configuration:
    width: int = 2
    mixture: float = .01


@dataclass(frozen=True)
class Task:
    deadline: int = 400


class Fixture(nn.Module):
    def __init__(self, *, dtype=torch.float32):
        super().__init__()
        self.config = Configuration()
        self.task = Task()
        self.weight = nn.Parameter(torch.tensor([[1., 2.], [3., 4.]], dtype=dtype))
        self.register_buffer("counter", torch.tensor(3, dtype=torch.long))
        self.register_buffer("scalar", torch.tensor(.5, dtype=dtype))


class OtherFixture(Fixture):
    pass


class ModuleVersionTests(unittest.TestCase):
    def test_equal_values_in_separate_storage_have_equal_version(self):
        first, second = Fixture(), Fixture()
        self.assertNotEqual(first.weight.data_ptr(), second.weight.data_ptr())
        self.assertEqual(module_version(first), module_version(second))
        self.assertEqual(len(module_version(first)), 64)

    def test_data_write_is_detected_even_when_tensor_version_counter_is_unchanged(self):
        module = Fixture()
        before, counter = module_version(module), module.weight._version
        module.weight.data.add_(1.)
        self.assertEqual(module.weight._version, counter)
        self.assertNotEqual(module_version(module), before)

    def test_normal_parameter_update_and_state_load_invalidate_then_restore(self):
        module = Fixture()
        original = {name: value.clone() for name, value in module.state_dict().items()}
        before = module_version(module)
        with torch.no_grad():
            module.weight.mul_(2.)
        self.assertNotEqual(module_version(module), before)
        module.load_state_dict(original)
        self.assertEqual(module_version(module), before)

    def test_persistent_buffers_are_versioned(self):
        module = Fixture()
        before = module_version(module)
        module.counter.add_(1)
        self.assertNotEqual(module_version(module), before)

    def test_nonpersistent_buffers_are_versioned_even_when_absent_from_state_dict(self):
        module = Fixture()
        module.register_buffer("inference_offset", torch.tensor(.25), persistent=False)
        self.assertNotIn("inference_offset", module.state_dict())
        before = module_version(module)
        module.inference_offset.add_(.5)
        self.assertNotEqual(module_version(module), before)

    def test_nested_config_task_and_class_are_versioned(self):
        module = Fixture()
        module.child = Fixture()
        original = module_version(module)
        module.child.config = replace(module.child.config, mixture=.02)
        self.assertNotEqual(module_version(module), original)
        module.child = Fixture()
        self.assertEqual(module_version(module), original)
        module.child.task = replace(module.child.task, deadline=300)
        self.assertNotEqual(module_version(module), original)
        module.child = OtherFixture()
        self.assertNotEqual(module_version(module), original)

    def test_config_task_and_qualified_module_class_are_versioned(self):
        original = module_version(Fixture())
        configured = Fixture()
        configured.config = replace(configured.config, mixture=.02)
        self.assertNotEqual(module_version(configured), original)
        task = Fixture()
        task.task = replace(task.task, deadline=300)
        self.assertNotEqual(module_version(task), original)
        self.assertNotEqual(module_version(OtherFixture()), original)

    def test_dtype_shape_and_name_are_versioned(self):
        original = module_version(Fixture())
        self.assertNotEqual(module_version(Fixture(dtype=torch.float64)), original)
        shape = Fixture()
        shape.weight = nn.Parameter(shape.weight.detach().reshape(4))
        self.assertNotEqual(module_version(shape), original)
        renamed = Fixture()
        renamed.other_weight = renamed.weight
        del renamed.weight
        self.assertNotEqual(module_version(renamed), original)

    def test_bfloat16_and_scalar_tensor_bytes_are_supported_without_casting(self):
        first, second = Fixture(dtype=torch.bfloat16), Fixture(dtype=torch.bfloat16)
        self.assertEqual(module_version(first), module_version(second))
        second.scalar.add_(torch.tensor(.125, dtype=torch.bfloat16))
        self.assertNotEqual(module_version(first), module_version(second))
        self.assertEqual(first.scalar.dtype, torch.bfloat16)
        self.assertEqual(first.scalar.shape, torch.Size([]))

    def test_noncontiguous_equal_values_have_equal_version(self):
        first, second = Fixture(), Fixture()
        second.weight = nn.Parameter(first.weight.detach().t().contiguous().t())
        self.assertFalse(second.weight.is_contiguous())
        self.assertEqual(module_version(first), module_version(second))

    def test_hashing_preserves_rng_modes_gradients_parameters_and_graph(self):
        module = Fixture()
        module.train()
        module.weight.grad = torch.ones_like(module.weight)
        saved_rng = torch.get_rng_state().clone()
        saved_parameters = {name: value.clone() for name, value in module.state_dict().items()}
        saved_gradient = module.weight.grad.clone()
        graph = module.weight.square().sum()
        first = module_version(module)
        self.assertTrue(module.training)
        self.assertTrue(module.weight.requires_grad)
        self.assertTrue(torch.equal(torch.get_rng_state(), saved_rng))
        self.assertTrue(torch.equal(module.weight.grad, saved_gradient))
        for name, value in module.state_dict().items():
            self.assertTrue(torch.equal(value, saved_parameters[name]), name)
        graph.backward()
        torch.testing.assert_close(module.weight.grad, saved_gradient + 2 * module.weight.detach())
        module.eval().requires_grad_(False)
        self.assertEqual(module_version(module), first)

    def test_complex_scalar_and_empty_integer_buffers_are_supported(self):
        first, second = nn.Module(), nn.Module()
        for module in (first, second):
            module.register_buffer("complex", torch.tensor(1 + 2j))
            module.register_buffer("empty", torch.empty((0, 2), dtype=torch.int64))
            module.register_buffer("boolean", torch.tensor(True))
        self.assertEqual(module_version(first), module_version(second))
        second.complex.add_(1j)
        self.assertNotEqual(module_version(first), module_version(second))

    def test_state_registration_order_does_not_change_identity(self):
        first, second = nn.Module(), nn.Module()
        for module, names in ((first, ("one", "two")), (second, ("two", "one"))):
            for name in names:
                module.register_buffer(name, torch.tensor(1 if name == "one" else 2))
        self.assertEqual(module_version(first), module_version(second))

    def test_nonfinite_parameters_or_metadata_are_rejected(self):
        for value in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(value=value):
                module = Fixture()
                module.weight.data[0, 0] = value
                with self.assertRaisesRegex(ValueError, "nonfinite"):
                    module_version(module)
        module = Fixture()
        module.config = replace(module.config, mixture=float("nan"))
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            module_version(module)

    def test_invalid_module_metadata_and_nonmaterialized_state_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "torch.nn.Module"):
            module_version(object())
        module = Fixture()
        module.config = {"width": 2}
        with self.assertRaisesRegex(ValueError, "dataclass"):
            module_version(module)
        module = nn.Linear(2, 2, device="meta")
        with self.assertRaisesRegex(ValueError, "dense materialized"):
            module_version(module)


if __name__ == "__main__":
    unittest.main()
