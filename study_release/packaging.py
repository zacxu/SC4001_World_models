"""Offline, allowlisted evidence packaging; never train, simulate or publish."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PureWindowsPath
import re
import shutil


ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "p62-minimal-release-v1"
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_FORBIDDEN = {".git", ".aws", ".ssh", ".envs", ".venv", "venv", "__pycache__", "cache", ".cache"}
_TEXT = {".json", ".jsonl", ".sha256", ".csv", ".yaml", ".yml", ".txt", ".md", ".py", ".acs", ".lock", ".toml", ".svg", ".png", ".cfg", ".ini"}
_BINARY = {".pt", ".pth", ".npz", ".npy", ".bin", ".wad", ".pk3", ".zip", ".tar", ".gz", ".exe", ".dll", ".so", ".o", ".lmp"}


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(data)
    return digest.hexdigest()


def _safe(value):
    path = Path(value).absolute()
    if ".." in path.parts:
        raise ValueError("release paths cannot traverse parents")
    for item in (path, *path.parents):
        if item.is_symlink() or getattr(item, "is_junction", lambda: False)():
            raise ValueError("release paths cannot contain links or junctions")
    if path != path.resolve():
        raise ValueError("release paths must be normalized")
    return path


def _relative(value):
    if type(value) is not str or not value or "\\" in value:
        raise ValueError("package references require POSIX relative paths")
    path = Path(value)
    if path.is_absolute() or ":" in value or any(x in (".", "..") for x in value.split("/")):
        raise ValueError("package path escapes its release directory")
    return path


def _forbidden(path, root):
    relative = path.relative_to(root)
    if any(part in _FORBIDDEN for part in relative.parts):
        raise ValueError("environments, caches and secrets are outside the release allowlist")
    if any(part.startswith(".env") or part.lower() in {"credentials", "id_rsa", "id_ed25519"}
           or part.lower().endswith((".pem", ".key")) for part in relative.parts):
        raise ValueError("secret-looking paths cannot be packaged")


def _json(path, record):
    path.write_text(json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def _ref(path):
    return {"path": str(path), "sha256": _sha(path)}


def _refs(value):
    if type(value) is dict:
        if type(value.get("path")) is str and _DIGEST.fullmatch(str(value.get("sha256", ""))):
            yield value
        for member in value.values():
            yield from _refs(member)
    elif type(value) is list:
        for member in value:
            yield from _refs(member)


def _readme():
    return """# Minimal reproducible exploratory release

This is a saved-evidence release of one frozen Relay B/D cohort, not a claim of
successful transfer, statistical superiority or a complete Phase 6 paper.

Regenerate the endpoint tables and two figures without a simulator, model,
GPU, replay restoration or training:

```bash
cd source
python -m pip install -r study_release/requirements.txt
python -B -m study_release.saved regenerate --release-dir .. --output-dir ../../regenerated
```

Use a fresh output directory. The command first verifies this package's byte
inventory and rebuilds tables from the included episode records and frozen bank.
The original JSON reports retain their original server paths; the explicit
reproduction_inputs.json mapping makes this saved-record command relocatable.
Unmeasured resource costs remain unknown. Raw per-seed values are preserved;
there are no confidence intervals, p-values or online learning-curve claims.

Source, locks and project-authored ACS/scenario source are under source/.
Four local-only actor exports are included. Their public actor input remains
local RGB, previous own action and local recurrent state; privileged teacher
inputs do not become deployment inputs. Actor files use Torch serialization;
only load checkpoints you trust. The saved-record reproduction command does
not deserialize actor or training checkpoints.

archive_locations.json records original paths, exact SHA-256 and byte sizes
for omitted model/replay/training binaries and external engine/compiler assets.
Those paths are archive locations on the original research server, not public
download links. This package does not promise independent retraining without
that archive, compatible original dependencies and separately licensed tools.
No commercial Doom assets, external compiler, environment or cache is copied.
Original task source is included with existing attribution/provenance; an open
source redistribution licence is not asserted where the project has none.
Freedoom/base engine assets must be obtained with their upstream licences.

command_provenance.json distinguishes saved command evidence from recommended
reproduction commands. Missing historical shell command text is not invented.
See evidence/ for frozen registrations, P2.4 selection/provenance, final native
attempt records, ledgers, deviations, P5 reports and fixed P6.1 diagnosis.
Earlier relay_week_v1 development history is retained separately and is never
pooled into the equal-budget endpoint comparison.

SHA256SUMS and release_manifest.json cover all packaged files. No uploading,
publishing, new native episodes, policy updates or model updates is performed.
"""


def build_release(context, diagnosis, output_dir, *, root=ROOT, extra_sources=(), archive_binary_limit=32 * 1024 * 1024):
    """Copy authenticated saved evidence and inventory omitted large binaries.

    The caller authenticates the completed P5 context before and after calling.
    This function independently rechecks every supplied byte reference and all
    frozen source hashes, then validates the complete resulting package.
    """
    root, output = _safe(root), _safe(output_dir)
    if type(archive_binary_limit) is not int or archive_binary_limit < 1:
        raise ValueError("binary size limit must be a positive integer")
    if output.exists():
        raise FileExistsError("release packaging requires a fresh directory")
    if root.is_relative_to(output) or any(output.is_relative_to(root / d) for d in
            ("src", "scripts", "tests", "configs", "scenarios", "study_reporting", "study_release")):
        raise ValueError("release output cannot overwrite project sources")
    cohort = context["cohort"]
    identity = cohort["cohort_identity"]
    if diagnosis.get("cohort_identity") != identity or diagnosis.get("status") != "complete":
        raise ValueError("release requires a complete diagnosis of this cohort")
    runs = context["runs"]
    if (len(runs) != 4 or {(r["variant"], r["master_seed"]) for r in runs} !=
            {(v, s) for v in ("B", "D") for s in (0, 1)}):
        raise ValueError("release requires the four registered local actors")
    protected = [Path(r["run_dir"]) for r in runs] + [Path(p).parent for p in context["evaluation_paths"]]
    if any(output.is_relative_to(_safe(p)) or _safe(p).is_relative_to(output) for p in protected):
        raise ValueError("release output cannot overlap training or test evidence")
    source_rows, evidence_rows, archived, queue, seen = {}, {}, {}, [], set()

    def add(path, digest=None, *, category="evidence", must_copy=False):
        path = _safe(path)
        if not path.is_relative_to(root):
            raise ValueError("release inputs must stay inside the project root")
        _forbidden(path, root)
        if not path.is_file() or digest is not None and _sha(path) != digest:
            raise ValueError(f"missing or changed release input: {path}")
        if path.is_relative_to(output) or output.is_relative_to(path):
            raise ValueError("release output overlaps an input")
        if path in seen:
            return
        seen.add(path)
        digest, size = _sha(path), path.stat().st_size
        row = {"original_path": str(path), "sha256": digest, "bytes": size, "category": category}
        external_tool = ".tools" in path.relative_to(root).parts
        binary = path.suffix.lower() in _BINARY
        omit = external_tool or path.suffix.lower() in {".wad", ".pk3"} or (not must_copy and (binary or size > archive_binary_limit))
        if omit:
            row["reason"] = "external or separately licensed asset" if external_tool or path.suffix.lower() in {".wad", ".pk3"} else "large replay/model/training binary retained in original server archive"
            archived[str(path)] = row
        elif category == "source":
            row["path"] = "source/" + path.relative_to(root).as_posix()
            source_rows[str(path)] = row
        else:
            if path.suffix.lower() not in _TEXT and not must_copy:
                raise ValueError(f"evidence extension is outside the release allowlist: {path.name}")
            row["path"] = "evidence/" + path.relative_to(root).as_posix()
            evidence_rows[str(path)] = row
        # Parse metadata references, never binary payloads. A direct reference is
        # required; missing historical nested paths are reported, not fabricated.
        if path.suffix == ".json" and not omit:
            queue.append(path)
        receipt = path.with_suffix(".sha256")
        if path.suffix != ".sha256" and receipt.is_file() and receipt != path:
            add(receipt, category=category, must_copy=True)

    for names in (cohort["source_sha256"], context["source_identity"]):
        if not names:
            raise ValueError("frozen source identities are required")
        for name, digest in sorted(names.items()):
            relative = _relative(name)
            add(root / relative, digest, category="source", must_copy=True)
    for path in sorted((root / "study_release").rglob("*.py")):
        add(path, category="source", must_copy=True)
    add(root / "study_release" / "requirements.txt", category="source", must_copy=True)
    for value in extra_sources:
        add(value if Path(value).is_absolute() else root / value, category="source", must_copy=True)
    # These operational helpers are outside the original *.py source glob.
    # Capture their current bytes explicitly without claiming they were frozen
    # prospectively with the already completed scientific cohort.
    helper_names = ("scripts/gpu_server_env.sh", "scripts/aspire2a_cuda.sh",
        "scripts/phase24_aspire2a.pbs", "linux_gpu_server_setup.md",
        "linux_gpu_server_setup_through_p23.md", "research_implementation_proposal.md")
    captured_helpers = []
    for name in helper_names:
        path = root / name
        if path.is_file():
            add(path, category="source", must_copy=True)
            captured_helpers.append({"path": "source/" + name, "sha256": _sha(path)})
    docs = [root / "README.md", root / "detailed_research_implementation_plan.md"]
    docs += sorted((root / "docs").glob("*.md")) + sorted((root / "scenarios").rglob("README.md"))
    for path in docs:
        if path.is_file():
            add(path, category="source", must_copy=True)
    diagnosis_artifacts = context.get("diagnosis_artifacts", [])
    if (len(diagnosis_artifacts) != 3 or not context.get("diagnosis_path")
            or len({r["path"] for r in diagnosis_artifacts}) != 3
            or {Path(r["path"]).suffix for r in diagnosis_artifacts} != {".json", ".md", ".csv"}
            or str(_safe(context["diagnosis_path"])) not in {str(_safe(r["path"])) for r in diagnosis_artifacts}):
        raise ValueError("release requires the diagnosis JSON, prose and selected-trace CSV artifacts")
    if json.loads(_safe(context["diagnosis_path"]).read_text(encoding="utf-8")) != diagnosis:
        raise ValueError("diagnosis publication differs from the requested release diagnosis")
    mandatory = context.get("release_inputs", []) + context["references"] + cohort["inputs"]["bound_files"] + diagnosis_artifacts
    for row in mandatory:
        add(row["path"], row["sha256"])
    for run in runs:
        if run.get("status") != "complete" or run.get("training_complete") is not True:
            raise ValueError("cannot package incomplete policy training as complete")
        for row in run["files"]:
            add(row["path"], row["sha256"])
        actor = _safe(run["actor_export"])
        for name in ("manifest.json", "manifest.sha256", "state.pt"):
            path = actor / name
            expected = next((r["sha256"] for r in run["files"] if r["path"] == str(path)), None)
            if expected is None:
                raise ValueError("local actor export is not bound by its final audit")
            # A first archived state.pt must be promoted to a required copy.
            if str(path) in archived:
                seen.remove(path); del archived[str(path)]
            add(path, expected, must_copy=True)
    if len(context["evaluation_paths"]) != 4 or len(context["evaluation_reports"]) != 4:
        raise ValueError("release requires four complete native evaluation reports")
    for path, report in zip(context["evaluation_paths"], context["evaluation_reports"]):
        if (report.get("status") != "complete" or report.get("native_backend") is not True
                or report.get("native_engine") is not True):
            raise ValueError("release requires completed native final-bank records")
        add(path)
        directory = _safe(path).parent
        for name in ("evaluation_manifest.json", "bank.json"):
            add(directory / name)
        for row in report.get("attempts", []):
            add(directory / _relative(row["path"]), row["sha256"])
        # The final evaluator's complete attempt histories and authored scenario
        # metadata are portable evidence. Never infer a generic relative path's
        # anchor from its appearance inside an unrelated provenance document.
        for artifact in sorted(directory.rglob("*")):
            if artifact.is_file() and artifact.suffix.lower() in (_TEXT - {".lock"}) | _BINARY:
                add(artifact)
    nested_missing = []
    for path in queue:
        record = json.loads(path.read_text(encoding="utf-8"))
        for row in _refs(record):
            candidate = Path(row["path"])
            if not candidate.is_absolute():
                # A known, already copied native artifact may have its schema's
                # local reference corroborated. This never adds generic paths.
                try:
                    anchored = _safe(path.parent / _relative(row["path"]))
                except ValueError:
                    anchored = None
                if anchored in seen and anchored.is_file() and _sha(anchored) == row["sha256"]:
                    continue
                nested_missing.append({"original_path": row["path"], "sha256": row["sha256"],
                    "status": "external_historical_path_not_copied" if PureWindowsPath(row["path"]).is_absolute()
                        else "relative_reference_retained_with_original_anchor_semantics"})
                continue
            candidate = _safe(candidate)
            if not candidate.is_relative_to(root):
                # System/compiler provenance is retained in its original JSON;
                # no file outside the project is copied or claimed archived.
                nested_missing.append({"original_path": str(candidate), "sha256": row["sha256"], "status": "external_reference_not_copied"})
            elif not candidate.is_file():
                nested_missing.append({"original_path": str(candidate), "sha256": row["sha256"], "status": "historical_nested_reference_unavailable"})
            else:
                add(candidate, row["sha256"])
    # Snapshot the complete source/reference closure before any output write.
    rows = sorted([*source_rows.values(), *evidence_rows.values()], key=lambda r: r["path"])
    output.mkdir(parents=True)
    for row in rows:
        target = output / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(row["original_path"], target)
        if target.stat().st_size != row["bytes"] or _sha(target) != row["sha256"]:
            raise ValueError("a release input changed while being copied")
    mapping = {row["original_path"]: {k: row[k] for k in ("path", "sha256", "bytes")} for row in rows}

    def mapped(path):
        try:
            return dict(mapping[str(_safe(path))])
        except KeyError as error:
            raise ValueError(f"required saved-record reproduction input was not copied: {path}") from error

    inputs = {"schema_version": "p62-reproduction-inputs-v1", "cohort_identity": identity,
        "protocol_sha256": cohort["protocol_sha256"], "analysis": mapped(context["analysis"]["report_path"]),
        "evaluations": [mapped(p) for p in context["evaluation_paths"]],
        "bank": mapped(_safe(context["evaluation_paths"][0]).parent / "bank.json"), "bank_key": None,
        "resources": mapped(context["deliverables"]["resources"]["path"]),
        "secondary": mapped(context["deliverables"]["secondary"]["path"]),
        "figure_data": [mapped(r["path"]) for r in context["figure_evidence"]["data"]],
        "tables": [mapped(_safe(context["analysis"]["report_path"]).parent / name) for name in
            ("arms.csv", "paired_differences.csv", "per_layout.csv", "episodes.csv")]
            + [mapped(r["path"]) for r in context["deliverables"]["tables"]]}
    _json(output / "reproduction_inputs.json", inputs)
    _json(output / "archive_locations.json", {"schema_version": "p62-original-server-archive-v1",
        "copied_binary_archive": False, "public_download_available": False,
        "files": sorted(archived.values(), key=lambda r: r["original_path"]),
        "archive_paths_have_been_hashed_not_uploaded": True,
        "unresolved_nested_provenance": sorted({json.dumps(r, sort_keys=True): r for r in nested_missing}.values(), key=lambda r: r["original_path"]),
        "retention": "Keep these exact original-server files and receipts; no upload or deletion was performed."})
    _json(output / "command_provenance.json", {"schema_version": "p62-command-provenance-v1",
        "historical_shell_commands_invented": False,
        "historical_shell_commands_recorded": context.get("historical_shell_commands_recorded", False),
        "saved_command_evidence": context.get("commands_provenance", []),
        "reproduction_command": "cd source && python -B -m study_release.saved regenerate --release-dir .. --output-dir ../../regenerated",
        "reproduction_command_was_historical_training_command": False})
    (output / "README.md").write_text(_readme(), encoding="utf-8", newline="\n")
    # Verify original bytes again so concurrent checkpoint/file changes fail.
    for row in [*rows, *archived.values()]:
        if _sha(row["original_path"]) != row["sha256"]:
            raise ValueError("original evidence changed during release packaging")
    files = [{"path": p.relative_to(output).as_posix(), "sha256": _sha(p), "bytes": p.stat().st_size,
              "category": "source" if p.is_relative_to(output / "source") else "evidence"}
             for p in sorted(output.rglob("*"), key=lambda item: item.relative_to(output).as_posix()) if p.is_file()]
    manifest = {"schema_version": SCHEMA, "status": "complete", "cohort_identity": identity,
        "protocol_sha256": cohort["protocol_sha256"], "files": files,
        "original_to_release": mapping, "frozen_source_sha256": cohort["source_sha256"],
        "reporting_source_sha256": context["source_identity"], "archive_binary_count": len(archived),
        "operational_helper_capture": {"scope": "current working-tree bytes at packaging; no retrospective cohort freeze claimed",
            "files": captured_helpers},
        "historical_nested_references_unavailable": sum(r["status"] == "historical_nested_reference_unavailable" for r in nested_missing),
        "nested_provenance_references_retained_without_reanchoring": len(nested_missing),
        "local_actor_exports": [{"variant": r["variant"], "master_seed": r["master_seed"],
            "manifest": mapped(Path(r["actor_export"]) / "manifest.json"),
            "state": mapped(Path(r["actor_export"]) / "state.pt")} for r in runs],
        "new_environment_interactions": 0, "new_training_updates": 0,
        "packaging_executes_reproduction": False, "published": False,
        "phase6_complete": False, "phase63_paper_acceptance": False,
        "independent_retraining_available_without_original_archive": False}
    _json(output / "release_manifest.json", manifest)
    receipt = output / "release_manifest.sha256"
    receipt.write_text(f"{_sha(output / 'release_manifest.json')}  release_manifest.json\n", encoding="ascii")
    checksum_paths = sorted((p for p in output.rglob("*") if p.is_file()),
        key=lambda item: item.relative_to(output).as_posix())
    (output / "SHA256SUMS").write_text("".join(f"{_sha(p)}  {p.relative_to(output).as_posix()}\n" for p in checksum_paths), encoding="ascii")
    return inspect_release(output)


def inspect_release(release_dir):
    """Verify every packaged byte without original paths or scientific imports."""
    root = _safe(release_dir)
    manifest_path = root / "release_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA or manifest.get("status") != "complete":
        raise ValueError("invalid minimal-release manifest")
    receipt = root / "release_manifest.sha256"
    if receipt.read_text(encoding="ascii") != f"{_sha(manifest_path)}  release_manifest.json\n":
        raise ValueError("release manifest receipt changed")
    rows, seen = manifest.get("files"), set()
    if type(rows) is not list or not rows:
        raise ValueError("release manifest needs its complete file inventory")
    for row in rows:
        relative = _relative(row["path"])
        path = _safe(root / relative)
        if (row["path"] in seen or not path.is_relative_to(root) or not path.is_file()
                or type(row.get("bytes")) is not int or path.stat().st_size != row["bytes"]
                or _sha(path) != row["sha256"]):
            raise ValueError("release file inventory is missing, duplicated or changed")
        seen.add(row["path"])
    expected = seen | {"release_manifest.json", "release_manifest.sha256", "SHA256SUMS"}
    actual = set()
    for p in root.rglob("*"):
        _safe(p)
        if p.is_file():
            actual.add(p.relative_to(root).as_posix())
    if actual != expected:
        raise ValueError("release has unlisted or missing files")
    checksum_paths = sorted((root / p for p in expected if p != "SHA256SUMS"),
        key=lambda item: item.relative_to(root).as_posix())
    checksums = "".join(f"{_sha(p)}  {p.relative_to(root).as_posix()}\n" for p in checksum_paths)
    if (root / "SHA256SUMS").read_text(encoding="ascii") != checksums:
        raise ValueError("release checksum listing changed")
    return manifest
