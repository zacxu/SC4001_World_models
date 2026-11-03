"""Local RGB conversion at a single, explicit real/imagined actor boundary."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from world_models.contracts import LOCAL_RGB_SHAPE, RGBFrame


RENDER_HEIGHT = 120
RENDER_WIDTH = 160
RESIZE_METHOD = "Pillow.Resampling.BILINEAR"


def adapt_rgb(screen_buffer: np.ndarray, *, layout: str = "CHW") -> RGBFrame:
    """Copy, bilinearly resize, and return immutable uint8 CHW local RGB.

    ViZDoom CRCGCB supplies CHW; RGB24 supplies HWC. The caller must declare the
    chosen layout. Missing terminal frames must remain missing in the environment
    result instead of being passed here or replaced with fabricated pixels.
    """
    if not isinstance(screen_buffer, np.ndarray):
        raise ValueError("screen_buffer must be an actual RGB numpy array")
    if screen_buffer.dtype != np.uint8:
        raise ValueError("screen_buffer must have uint8 dtype")
    if layout == "CHW":
        if screen_buffer.shape != (3, RENDER_HEIGHT, RENDER_WIDTH):
            raise ValueError("CHW render buffer must have shape [3,120,160]")
        hwc = np.array(screen_buffer.transpose(1, 2, 0), copy=True, order="C")
    elif layout == "HWC":
        if screen_buffer.shape != (RENDER_HEIGHT, RENDER_WIDTH, 3):
            raise ValueError("HWC render buffer must have shape [120,160,3]")
        hwc = np.array(screen_buffer, copy=True, order="C")
    else:
        raise ValueError("layout must be CHW or HWC")
    resized = Image.fromarray(hwc).resize(
        (LOCAL_RGB_SHAPE[2], LOCAL_RGB_SHAPE[1]), resample=Image.Resampling.BILINEAR
    )
    chw = np.asarray(resized, dtype=np.uint8).transpose(2, 0, 1)
    return RGBFrame(chw.tobytes(order="C"))


def frame_array(frame: RGBFrame) -> np.ndarray:
    """Return a read-only [3,96,128] view backed by immutable frame bytes."""
    if not isinstance(frame, RGBFrame):
        raise ValueError("frame must be an RGBFrame")
    return np.frombuffer(frame.data, dtype=np.uint8).reshape(LOCAL_RGB_SHAPE)


def normalized_rgb(frame: RGBFrame) -> np.ndarray:
    """Return a detached float32 CHW array in [0,1], with no global features."""
    return frame_array(frame).astype(np.float32) / np.float32(255.0)


def save_rgb(frame: RGBFrame, destination: Path | str) -> None:
    """Save the precise actor-visible RGB frame for interface inspection."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(frame_array(frame).transpose(1, 2, 0)).save(path)

