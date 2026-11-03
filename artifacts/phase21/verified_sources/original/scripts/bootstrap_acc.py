"""Install the pinned official Windows ACS compiler inside this workspace."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import urllib.request
import zipfile


ROOT = Path(__file__).resolve().parents[1]
URL = "https://github.com/ZDoom/acc/releases/download/1.60/acc-1.60-win32.zip"
ARCHIVE_SHA256 = "d70c633fcf07da705562ce65104379e1c9ecaeaa159981acc5275840e606a15f"
EXECUTABLE_SHA256 = "a318f4b45818222dc273a68fab37e393d6881e603816c7317da40d2819fa3ce5"


def bootstrap() -> dict:
    if os.name != "nt":
        raise RuntimeError("This bootstrap pins the tested Windows compiler; on Linux install official ACC 1.60 and set WORLD_MODELS_ACC.")
    tools = ROOT / ".tools"
    tools.mkdir(exist_ok=True)
    archive = tools / "acc-1.60-win32.zip"
    if not archive.exists():
        with urllib.request.urlopen(URL, timeout=60) as response:
            content = response.read()
        if hashlib.sha256(content).hexdigest() != ARCHIVE_SHA256:
            raise RuntimeError("official compiler archive checksum mismatch")
        archive.write_bytes(content)
    if hashlib.sha256(archive.read_bytes()).hexdigest() != ARCHIVE_SHA256:
        raise RuntimeError("cached compiler archive checksum mismatch")
    destination = (tools / "acc-1.60").resolve()
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if not target.is_relative_to(destination):
                raise RuntimeError("unsafe compiler archive member")
        source.extractall(destination)
    if hashlib.sha256((destination / "acc.exe").read_bytes()).hexdigest() != EXECUTABLE_SHA256:
        raise RuntimeError("extracted compiler executable checksum mismatch")
    return {"version": "1.60", "source": URL, "archive_sha256": ARCHIVE_SHA256,
            "executable_sha256": EXECUTABLE_SHA256, "path": str(destination / "acc.exe")}


if __name__ == "__main__":
    print(json.dumps(bootstrap(), indent=2))
