"""Record the actual tested simulator stack and asset provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
from pathlib import Path
import platform
import sys

from world_models.env.actions import BUTTON_NAMES
from world_models.env.observations import RESIZE_METHOD
from world_models.env.scenario_builder import compiler_path
from world_models.env.workers import resolve_iwad


def file_record(path: Path, root: Path | None = None) -> dict:
    return {"path": str(path.relative_to(root)) if root else str(path.resolve()),
            "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def runtime_manifest(root: Path) -> dict:
    import vizdoom
    import imageio_ffmpeg
    package = Path(vizdoom.__file__).resolve().parent
    assets = [file_record(path, package) for path in sorted(package.iterdir())
              if path.is_file() and path.suffix.lower() in (".wad", ".pk3", ".exe", ".pyd", ".dll", ".so")]
    pins = {}
    for line in (root / "requirements-phase1.lock").read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        name, expected = line.split("==", 1)
        actual = importlib.metadata.version(name)
        pins[name] = {"expected": expected, "observed": actual, "matches": actual == expected}
    compiler = compiler_path()
    selected_iwad, selected_sha256 = resolve_iwad()
    return {"recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "scope": "observed_development_simulator_stack; not a CUDA/PyTorch training-host validation",
            "python": {"version": platform.python_version(), "executable": sys.executable},
            "platform": platform.platform(), "packages": pins, "assets": assets,
            "selected_iwad": {"path": str(selected_iwad), "sha256": selected_sha256,
                              "selection": "explicit bundled freedoom2.wad; each worker verifies engine path and hash"},
            "video_encoder": {**file_record(Path(imageio_ffmpeg.get_ffmpeg_exe())),
                              "version": imageio_ffmpeg.get_ffmpeg_version(),
                              "package": "imageio-ffmpeg==0.6.0", "purpose": "mechanics review videos"},
            "compiler": {"version": "1.60", **file_record(compiler),
                         "source": "https://github.com/ZDoom/acc/releases/tag/1.60"},
            "renderer": {"mode": "PLAYER", "off_screen": True, "sound": False,
                         "format": "CRCGCB", "raw_shape": [3,120,160], "actor_shape": [3,96,128],
                         "resize": RESIZE_METHOD, "buttons": list(BUTTON_NAMES),
                         "autoaim": False, "friendly_fire": False, "automatic_respawn": False,
                         "client_prediction": False,
                         "native_barrier": "both set_action then shared barrier then advance_action(1); four tics per logical decision",
                         "coordinated_reset": "fresh participant processes and isolated episode working directories"},
            "license_provenance": {
                "original_vizdoom_code": "MIT; bundled engine components retain their separate licenses",
                "vizdoom_source": "https://github.com/Farama-Foundation/ViZDoom/tree/1.3.2",
                "vizdoom_license_statement": "https://github.com/Farama-Foundation/ViZDoom#license",
                "freedoom": "https://github.com/freedoom/freedoom/blob/master/COPYING.adoc",
                "acc": "https://github.com/ZDoom/acc/releases/tag/1.60",
                "generated_pwads": "Original project geometry and scripts; no copied IWAD sprites/textures in generated PWADs.",
                "redistribution": "Preserve upstream Freedoom credits/license when distributing the IWAD; do not label the entire engine bundle MIT."}}
