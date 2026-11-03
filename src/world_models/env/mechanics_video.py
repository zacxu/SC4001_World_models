"""Review videos from genuine local actor frames, outside collection timing."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

from world_models.env.observations import save_rgb


class MechanicsVideo:
    def __init__(self, directory: str | Path, task_family: str, frame_skip: int, *,
                 frame_provenance: str = "Caller-supplied frames; see accompanying trajectory for native provenance"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.task_family, self.frame_skip = task_family, frame_skip
        self.frame_provenance = frame_provenance
        self.records = []

    def append(self, result, actions=None):
        index = len(self.records)
        paths = []
        hashes = []
        for player, frame in enumerate(result.observations):
            if frame is None:
                raise ValueError("mechanics video must not fabricate a missing actor frame")
            path = self.directory / f"frame{index:04d}_player{player}.png"
            save_rgb(frame, path)
            paths.append(str(path.resolve()))
            hashes.append(hashlib.sha256(frame.data).hexdigest())
        info = result.training_only_info
        self.records.append({"frame": index, "episode_id": info["episode_id"],
                             "decision": info["decision_index"], "actions": actions,
                             "image_paths": paths, "rgb_payload_sha256": hashes,
                             "task_flags": info["task_flags"], "events": info.get("events"),
                             "team_reward": getattr(result, "team_reward", None),
                             "boundary": getattr(getattr(result, "boundary", None), "reason", None)})

    def finish(self) -> dict:
        path = self.directory / f"{self.task_family}_mechanics.mp4"
        writer = imageio_ffmpeg.write_frames(str(path), (512, 224), fps=35/self.frame_skip,
                                             codec="libx264", pix_fmt_out="yuv420p", quality=8,
                                             macro_block_size=16, output_params=["-threads", "1", "-movflags", "+faststart"],
                                             ffmpeg_log_level="error")
        writer.send(None)
        try:
            for row in self.records:
                image = Image.new("RGB", (256, 112), "#101010")
                draw = ImageDraw.Draw(image)
                draw.text((2, 0), f"{self.task_family} decision {row['decision']} | P0 / P1", fill="white")
                for player, source in enumerate(row["image_paths"]):
                    with Image.open(source) as frame:
                        image.paste(frame, (128*player, 16))
                # Integer nearest-neighbor enlargement; source PNGs stay exact.
                writer.send(image.resize((512, 224), Image.Resampling.NEAREST).tobytes())
        finally:
            writer.close()
        probe = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
        try:
            metadata = next(probe)
            decoded = sum(1 for _ in probe)
        finally:
            probe.close()
        if decoded != len(self.records):
            raise RuntimeError("encoded mechanics video lost frames")
        manifest = {"task_family": self.task_family, "path": str(path.resolve()),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "fps": 35/self.frame_skip, "decoded_frames": decoded, "dimensions": metadata["size"],
                    "frame_provenance":self.frame_provenance,
                    "source": "128x96 actor PNGs, side by side; caption lies outside actor pixels. Lossy H264 review export; source PNGs are authoritative.",
                    "records": self.records}
        (self.directory/"video_manifest.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
        return {key:value for key,value in manifest.items() if key != "records"}
