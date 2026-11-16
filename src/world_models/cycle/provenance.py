"""Source identities for the additive Phase 3 training controller."""
from pathlib import Path
import hashlib


ROOT = Path(__file__).resolve().parents[3]


def source_identity():
    from scripts.phase34 import source_identity as foundation

    result = foundation()
    # Real collection adds scenario/coordinator/ledger dependencies beyond the
    # earlier model-only receipts. Bind their implementation as well.
    paths = [*sorted((ROOT / "src/world_models").rglob("*.py")), ROOT / "scripts/phase35.py",
             ROOT / "scripts/phase24_freeze.py"]
    result.update({path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in paths})
    return result
