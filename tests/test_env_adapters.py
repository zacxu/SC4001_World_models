"""Meaningful action/image boundary checks; no engine or torch import needed."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from world_models.contracts import ACTION_BUTTONS, ACTION_COUNT, PREVIOUS_ACTION_START, RGBFrame
from world_models.env.actions import BUTTON_NAMES, encode_action
from world_models.env.observations import adapt_rgb, frame_array, normalized_rgb, save_rgb


class ActionAdapterTests(unittest.TestCase):
    def test_every_action_has_the_declared_semantics(self):
        for action in range(ACTION_COUNT):
            with self.subTest(action=action):
                encoded = encode_action(action)
                self.assertEqual(len(encoded), 8)
                self.assertEqual({name for name, pressed in zip(BUTTON_NAMES, encoded) if pressed}, set(ACTION_BUTTONS[action]))
                self.assertTrue(all(type(value) is int and value in (0, 1) for value in encoded))

    def test_reset_symbol_and_invalid_action_values_cannot_execute(self):
        for action in (PREVIOUS_ACTION_START, -1, 99, True, 1.0, "1"):
            with self.subTest(action=action), self.assertRaises(ValueError):
                encode_action(action)

    def test_engine_button_order_can_be_permuted_without_changing_action(self):
        reverse = tuple(reversed(BUTTON_NAMES))
        self.assertEqual(encode_action(9, reverse), [int(name in {"MOVE_FORWARD", "ATTACK"}) for name in reverse])

    def test_missing_duplicate_and_unknown_buttons_are_rejected(self):
        for buttons in (BUTTON_NAMES[:-1], BUTTON_NAMES[:-1] + ("ATTACK",), BUTTON_NAMES[:-1] + ("JUMP",), "ATTACK"):
            with self.subTest(buttons=buttons), self.assertRaises(ValueError):
                encode_action(0, buttons)


class ObservationAdapterTests(unittest.TestCase):
    def test_shape_channel_order_and_constants_are_preserved(self):
        source = np.zeros((3, 120, 160), dtype=np.uint8)
        source[0] = 255
        source[1] = 80
        source[2] = 17
        result = adapt_rgb(source)
        self.assertIsInstance(result, RGBFrame)
        array = frame_array(result)
        self.assertEqual(array.shape, (3, 96, 128))
        self.assertEqual(array.dtype, np.uint8)
        for channel, expected in enumerate((255, 80, 17)):
            self.assertTrue(np.all(array[channel] == expected))

    def test_hwc_and_chw_declarations_produce_identical_frames(self):
        source = np.arange(3 * 120 * 160, dtype=np.uint8).reshape((120, 160, 3))
        self.assertEqual(adapt_rgb(source, layout="HWC"), adapt_rgb(source.transpose(2, 0, 1), layout="CHW"))

    def test_renderer_buffer_mutation_cannot_change_stored_frame(self):
        source = np.full((3, 120, 160), 91, dtype=np.uint8)
        result = adapt_rgb(source)
        source[:] = 0
        self.assertTrue(np.all(frame_array(result) == 91))
        with self.assertRaises(ValueError):
            frame_array(result)[0, 0, 0] = 0

    def test_normalization_is_float32_in_unit_interval(self):
        source = np.zeros((3, 120, 160), dtype=np.uint8)
        source[0] = 255
        result = normalized_rgb(adapt_rgb(source))
        self.assertEqual(result.dtype, np.float32)
        self.assertEqual(float(result.min()), 0.0)
        self.assertEqual(float(result.max()), 1.0)
        self.assertTrue(np.all(result[0] == 1.0))

    def test_missing_or_malformed_frames_are_not_fabricated(self):
        invalid = (None, b"pixels", np.zeros((3, 96, 128), dtype=np.uint8),
                   np.zeros((3, 120, 160), dtype=np.float32), np.zeros((120, 160), dtype=np.uint8))
        for source in invalid:
            with self.subTest(source_type=type(source)), self.assertRaises(ValueError):
                adapt_rgb(source)
        with self.assertRaises(ValueError):
            adapt_rgb(np.zeros((3, 120, 160), dtype=np.uint8), layout="BGR")

    def test_resize_matches_the_documented_bilinear_reference(self):
        source = np.zeros((120, 160, 3), dtype=np.uint8)
        source[:, :80, 0] = 255
        source[:60, :, 1] = 255
        expected = np.asarray(Image.fromarray(source).resize((128, 96), Image.Resampling.BILINEAR)).transpose(2, 0, 1)
        self.assertTrue(np.array_equal(frame_array(adapt_rgb(source, layout="HWC")), expected))

    def test_saved_image_is_exactly_the_actor_visible_frame(self):
        source = np.full((3, 120, 160), 42, dtype=np.uint8)
        frame = adapt_rgb(source)
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "actor.png"
            save_rgb(frame, destination)
            image = np.asarray(Image.open(destination))
        self.assertEqual(image.shape, (96, 128, 3))
        self.assertTrue(np.array_equal(image.transpose(2, 0, 1), frame_array(frame)))


if __name__ == "__main__":
    unittest.main()
