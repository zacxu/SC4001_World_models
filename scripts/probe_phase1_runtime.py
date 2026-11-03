"""Actual headless ViZDoom single-player/runtime and local-RGB smoke probe.

This checks engine startup and the actor image interface. It does not establish
multiplayer synchronization or custom cooperative task correctness.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time

import vizdoom as vzd

from world_models.env.actions import BUTTON_NAMES, button_enums, encode_action
from world_models.env.observations import adapt_rgb, frame_array, save_rgb


ROOT = Path(__file__).resolve().parents[1]


def file_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def probe(output_dir: Path) -> tuple[dict, bool]:
    output_dir.mkdir(parents=True, exist_ok=True)
    package_dir = Path(vzd.__file__).resolve().parent
    scenario_dir = Path(vzd.scenarios_path)
    assets = [package_dir / "freedoom1.wad", package_dir / "freedoom2.wad",
              package_dir / "vizdoom.pk3", package_dir / "vizdoom.exe",
              scenario_dir / "basic.cfg", scenario_dir / "basic.wad"]
    assets.extend(package_dir.glob("vizdoom*.pyd"))
    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "real_headless_singleplayer_engine_and_local_image_smoke_only",
        "status": "failed", "python": sys.version, "platform": platform.platform(),
        "packages": {name: importlib.metadata.version(name) for name in ("vizdoom", "numpy", "Pillow")},
        "configuration": {"mode": "PLAYER", "headless": True, "screen_format": "CRCGCB",
                          "render_shape": [3,120,160], "actor_shape": [3,96,128], "action_repeat": 4,
                          "seed": 1234, "button_names": BUTTON_NAMES, "base_assets": "bundled freedoom2.wad"},
        "files": [file_record(path) for path in assets if path.is_file()],
        "provenance": {
            "vizdoom_original_code": {"license": "MIT for original ViZDoom code; bundled engine components have separate varying licenses",
                "source": "https://github.com/Farama-Foundation/ViZDoom", "version": vzd.__version__},
            "license_statement": "https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/main/README.md",
            "freedoom_license": "https://raw.githubusercontent.com/freedoom/freedoom/master/COPYING.adoc",
            "freedoom_note": "Preserve Freedoom copyright/conditions/disclaimer and credits when redistributing base assets; do not label the entire wheel MIT.",
        },
        "steps": [],
    }
    game = vzd.DoomGame()
    started = time.monotonic()
    try:
        game.load_config(str(scenario_dir / "basic.cfg"))
        game.set_doom_game_path(str(package_dir / "freedoom2.wad"))
        game.set_doom_scenario_path(str(scenario_dir / "basic.wad"))
        game.set_window_visible(False)
        game.set_mode(vzd.Mode.PLAYER)
        game.set_screen_resolution(vzd.ScreenResolution.RES_160X120)
        game.set_screen_format(vzd.ScreenFormat.CRCGCB)
        game.set_available_buttons(button_enums())
        game.set_render_hud(True)
        game.set_render_minimal_hud(False)
        game.set_render_messages(False)
        game.set_automap_buffer_enabled(False)
        game.set_depth_buffer_enabled(False)
        game.set_labels_buffer_enabled(False)
        game.set_objects_info_enabled(False)
        game.set_sectors_info_enabled(False)
        game.set_sound_enabled(False)
        game.set_console_enabled(False)
        game.set_seed(1234)
        game.init()
        initial = game.get_state()
        if initial is None:
            raise RuntimeError("engine initialized without a valid initial image")
        actor_image = adapt_rgb(initial.screen_buffer)
        save_rgb(actor_image, output_dir / "singleplayer.png")
        report["initial_image"] = {"raw_shape": list(initial.screen_buffer.shape),
                                   "raw_dtype": str(initial.screen_buffer.dtype),
                                   "actor_shape": list(frame_array(actor_image).shape),
                                   "actor_dtype": str(frame_array(actor_image).dtype),
                                   "tic": initial.tic, "png": file_record(output_dir / "singleplayer.png")}
        for action in (0,1,2,3,4,5,6,8,7,9,10,11):
            before = game.get_state()
            if before is None:
                break
            before_tic = before.tic
            reward = game.make_action(encode_action(action), 4)
            after = game.get_state()
            record = {"action": action, "buttons": encode_action(action), "start_tic": before_tic,
                      "engine_reward": reward, "finished": game.is_episode_finished(),
                      "end_tic": after.tic if after is not None else game.get_episode_time()}
            if after is not None:
                if after.tic - before_tic != 4:
                    raise RuntimeError("nonterminal step did not advance exactly four tics")
                save_rgb(adapt_rgb(after.screen_buffer), output_dir / "singleplayer_after.png")
            report["steps"].append(record)
            if game.is_episode_finished():
                break
        if not report["steps"]:
            raise RuntimeError("no engine transition was completed")
        report["status"] = "passed"
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
    finally:
        game.close()
        report["elapsed_seconds"] = time.monotonic() - started
    return report, report["status"] == "passed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "phase1_steps1_4")
    args = parser.parse_args()
    report, passed = probe(args.output_dir.resolve())
    path = args.output_dir / "singleplayer_runtime_probe.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(path), "error": report.get("error")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
