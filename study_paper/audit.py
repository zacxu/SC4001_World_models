"""Read saved evidence on Windows without rewriting historical server paths."""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import sys
import tempfile
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from study_release.evidence import canonical_hash, file_sha256, read_report
from study_release.packaging import inspect_release
from .tables import blocks, replace_block

COMPLETION = ROOT / "artifacts/phase6_runs/completion_20261113_104421"
REPRODUCTION = ROOT / "artifacts/phase6_runs/reproduction_20261113_104832"
MANUSCRIPT = ROOT / "paper/research_paper_draft.md"
CLAIMS = ROOT / "paper/claim_evidence.json"
REVIEW = ROOT / "paper/manuscript_review.json"
GENERATED = ROOT / "paper/generated"
ORIGINAL_ROOT = "/home/zixuan/projects/WM1/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def relative_file(root, name):
    require(isinstance(name, str) and bool(name), "empty evidence path")
    require(not PureWindowsPath(name).drive and not PurePosixPath(name).is_absolute()
        and "\\" not in name and all(p not in ("", ".", "..") for p in name.split("/")),
        "evidence path must be a confined relative POSIX path")
    root = Path(root).resolve()
    path = root / name
    require(not any(p.is_symlink() or getattr(p, "is_junction", lambda: False)()
        for p in (path, *path.parents)), "evidence paths cannot contain links")
    require(path.resolve().is_relative_to(root) and path.is_file(), f"missing evidence file: {name}")
    return path


def relocated(name):
    if name.startswith(ORIGINAL_ROOT):
        return relative_file(ROOT, name[len(ORIGINAL_ROOT):])
    path = Path(name)
    if path.is_absolute() and path.resolve().is_relative_to(ROOT):
        return relative_file(ROOT, path.resolve().relative_to(ROOT).as_posix())
    return relative_file(ROOT, name)


def check_ref(ref, pins):
    path = relocated(ref["path"])
    require(file_sha256(path) == ref["sha256"], f"changed evidence: {path}")
    require("bytes" not in ref or path.stat().st_size == ref["bytes"], f"changed size: {path}")
    pins[path] = ref["sha256"]
    return path


def pin_report(path, pins):
    report = read_report(path)
    for p in (path, path.with_suffix(".sha256")):
        pins[p] = file_sha256(p)
    return report


def json_pointer(value, pointer):
    require(type(pointer) is str and (not pointer or pointer.startswith("/")), "invalid JSON pointer")
    if not pointer:
        return value
    for token in pointer[1:].split("/"):
        require(re.search(r"~(?![01])", token) is None, "invalid JSON pointer escape")
        token = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list):
            require(re.fullmatch(r"0|[1-9][0-9]*", token) is not None, "invalid array pointer")
            value = value[int(token)]
        else:
            value = value[token]
    return value


def strict_equal(left, right):
    return type(left) is type(right) and canonical_hash(left) == canonical_hash(right)


def validate_stage_reports(completion, reproduction, manifest):
    identity = manifest["cohort_identity"]
    require(completion.get("schema_version") == "p6162-completion-v1"
        and reproduction.get("schema_version") == "p62-saved-reproduction-v1", "wrong stage schema")
    require(all(r.get("status") == "complete" and r.get("cohort_identity") == identity
        for r in (completion, reproduction)), "stage status or cohort mismatch")
    require(all(completion.get(k) is True for k in ("P6.1_complete", "P6.2_complete", "release_byte_integrity_checked")),
        "P6.1-P6.2 incomplete")
    require(all(reproduction.get(k) is True for k in ("numerical_reproduction", "six_tables_byte_identical", "two_figures_rendered")),
        "saved reproduction incomplete")
    require(all(r.get(k) is False for r in (completion, reproduction)
        for k in ("research_training_executed", "research_evaluation_executed")), "unexpected scientific rerun")


def verify_sources(manifest, completion):
    from study_release.evidence import prior_identities, source_identity
    for name, actual in prior_identities().items():
        require(actual == manifest[name], f"frozen source set changed: {name}")
    require(source_identity() == completion["source_sha256"], "P6 release sources changed")


def load_evidence():
    pins = {}
    completion = pin_report(COMPLETION / "phase61_62_completion.json", pins)
    repro = pin_report(REPRODUCTION / "reproduction_report.json", pins)
    release = COMPLETION / "release"
    print("Checking sealed release inventory...", flush=True)
    manifest = inspect_release(release)
    validate_stage_reports(completion, repro, manifest)
    for report in (completion, repro):
        require(check_ref(report["release_manifest"], pins) == release / "release_manifest.json",
            "completion/reproduction points to another release")
    verify_sources(manifest, completion)
    for row in manifest["files"]:
        pins[release / row["path"]] = row["sha256"]
    for name in ("release_manifest.json", "release_manifest.sha256", "SHA256SUMS"):
        pins[release / name] = file_sha256(release / name)
    p5path = check_ref(completion["phase5_acceptance"], pins)
    p5 = pin_report(p5path, pins)
    require(p5.get("status") == "passed" and p5.get("phase5_acceptance") is True
        and p5.get("cohort_identity") == manifest["cohort_identity"], "Phase 5 acceptance mismatch")
    check_ref(completion["processing_manifest"], pins)
    for row in completion["diagnosis"]:
        check_ref(row, pins)
    # The older three manuscript tables predate the compact P6 package. Bind
    # their original local reports/receipts too, without adding to that package.
    from paper import update_development_tables as development
    historical = [development.P21, *development.P25.rglob("report.json")]
    historical += [development.OLD_P36 / "trained/evaluation.json",
        development.FINAL_P36 / "baseline/evaluation.json", development.FINAL_P36 / "trained/evaluation.json"]
    for path in historical:
        report = pin_report(path, pins)
        if path.name == "evaluation.json":
            for name, ref in report["files"].items():
                dep = relative_file(path.parent, name)
                require(file_sha256(dep) == ref["sha256"], "historical native evidence changed")
                pins[dep] = ref["sha256"]
    mapping = json.loads((release / "reproduction_inputs.json").read_text(encoding="utf-8"))
    require(mapping["cohort_identity"] == manifest["cohort_identity"]
        and mapping["protocol_sha256"] == manifest["protocol_sha256"], "reproduction mapping mismatch")
    index = {row["path"]: row for row in manifest["files"]}
    def input_file(row):
        require(row["path"] in index and row["sha256"] == index[row["path"]]["sha256"], "unindexed reproduction input")
        return relative_file(release, row["path"])
    def data(key):
        return json.loads(input_file(mapping[key]).read_text(encoding="utf-8"))
    evaluations = [json.loads(input_file(row).read_text(encoding="utf-8")) for row in mapping["evaluations"]]
    bank, secondary, resources = data("bank"), data("secondary"), data("resources")
    if mapping.get("bank_key"):
        bank = bank[mapping["bank_key"]]
    from world_models_endpoint.analysis import aggregate_records
    from study_release.saved import table_bytes
    from study_reporting.phase5 import write_tables
    from study_reporting.figures import build_figure_data
    metrics = aggregate_records(evaluations, bank=bank, cohort_identity=manifest["cohort_identity"],
        protocol_sha256=manifest["protocol_sha256"], require_native=True)
    require(canonical_hash(metrics) == canonical_hash(data("analysis")["analysis"]), "episode reaggregation differs")
    require(all(r["cohort_identity"] == manifest["cohort_identity"] for r in (secondary, resources)), "report cohort mismatch")
    outputs = table_bytes(metrics)
    with tempfile.TemporaryDirectory(prefix="wm_p63_") as tmp:
        directory = Path(tmp)
        write_tables(directory, secondary, resources)
        for name in ("resources.csv", "training_diagnostics.csv"):
            outputs[name] = (directory / name).read_bytes()
    require(set(outputs) == {Path(r["path"]).name for r in mapping["tables"]}, "six-table inventory mismatch")
    for row in mapping["tables"]:
        require(outputs[Path(row["path"]).name] == input_file(row).read_bytes(), "regenerated CSV differs from archive")
    require(len(repro["tables"]) == 6 and {Path(r["path"]).name for r in repro["tables"]} == set(outputs),
        "reproduction table set mismatch")
    for row in repro["tables"]:
        path = check_ref(row, pins)
        require(path.is_relative_to(REPRODUCTION) and outputs[path.name] == path.read_bytes(), "reproduced CSV differs")
    figures = build_figure_data({"cohort": {"cohort_identity": manifest["cohort_identity"],
        "protocol_sha256": manifest["protocol_sha256"]}, "analysis": {"status": "complete", "analysis": metrics,
        "cohort_identity": manifest["cohort_identity"]}, "references": [], "source_identity": manifest["reporting_source_sha256"]}, resources)
    require(len(mapping["figure_data"]) == 2, "numeric figure set incomplete")
    for row, numbers in zip(mapping["figure_data"], figures):
        path = input_file(row)
        require(canonical_hash(numbers) == canonical_hash(json.loads(path.read_text())), "figure numbers differ")
        copied = REPRODUCTION / "figures" / path.name
        recorded = pin_report(copied, pins)
        require(canonical_hash(recorded) == canonical_hash(numbers), "reproduced figure numbers differ")
        outputs[path.name] = path.read_bytes()
    required_images = {f"{name}.{ext}" for name in ("endpoint_results", "paired_effects_compute") for ext in ("svg", "png")}
    require(len(repro["figures"]) == 4 and {Path(row["path"]).name for row in repro["figures"]} == required_images,
        "rendered figure set incomplete")
    for row in repro["figures"]:
        path = check_ref(row, pins)
        require(path.is_relative_to(REPRODUCTION), "figure outside reproduction")
        receipt = path.with_suffix(path.suffix + ".sha256")
        require(receipt.read_text(encoding="ascii").strip() == f"{row['sha256']}  {path.name}", "figure receipt changed")
        pins[receipt] = file_sha256(receipt)
        outputs[path.name] = path.read_bytes()
    print("Reaggregated 80 saved episodes; six CSVs and two numeric figures match.", flush=True)
    return {"manifest": manifest, "completion": completion, "reproduction": repro, "release": release,
        "pins": pins, "metrics": metrics, "secondary": secondary, "resources": resources,
        "evaluations": evaluations, "bank": bank, "outputs": outputs,
        "blocks": blocks(metrics, secondary, resources, evaluations)}


def audit_claims(claims, text, release, manifest):
    require(claims.get("schema_version") == "p63-claim-evidence-v1"
        and claims.get("cohort_identity") == manifest["cohort_identity"], "claim schema/cohort mismatch")
    index = {r["path"]: r for r in manifest["files"]}
    rows, seen, assertions = claims.get("claims"), set(), 0
    require(type(rows) is list and bool(rows), "empty claims")
    for row in rows:
        require(type(row.get("id")) is str and row["id"] not in seen, "duplicate/invalid claim ID")
        seen.add(row["id"])
        require(row.get("manuscript_heading") in text and bool(row.get("claim"))
            and bool(row.get("scope")) and bool(row.get("limitation")), "unlocated or unbounded claim")
        require(type(row.get("evidence")) is list and bool(row["evidence"]), "claim has no evidence")
        for ref in row["evidence"]:
            path = relative_file(release, ref["package_path"])
            require(ref["package_path"] in index and ref["sha256"] == index[ref["package_path"]]["sha256"]
                and file_sha256(path) == ref["sha256"], "claim evidence is not sealed")
            if "pointer" in ref:
                actual = json_pointer(json.loads(path.read_text(encoding="utf-8")), ref["pointer"])
                require(strict_equal(actual, ref["expected"]), f"claim assertion differs: {row['id']}")
                assertions += 1
            else:
                require(bool(ref.get("source_note")), "source evidence needs a review note")
    return {"claims": len(rows), "exact_json_assertions": assertions}


def claims_markdown(claims):
    lines = ["# P6.3 claim-to-evidence checklist", "", f"Cohort: `{claims['cohort_identity']}`.", "",
        "Each item was reviewed against the stated scope. Exact assertions are in [claim_evidence.json](claim_evidence.json).",
        "Automated assertions check saved values and hashes; they do not independently prove the prose interpretation.", ""]
    base = "../artifacts/phase6_runs/completion_20261113_104421/release/"
    for row in claims["claims"]:
        lines += [f"## {row['id']}", "", f"**Claim:** {row['claim']}", "",
            f"**Manuscript:** {row['manuscript_heading']}", "", f"**Scope:** {row['scope']}", "",
            f"**Boundary:** {row['limitation']}", "", "**Evidence:**", ""]
        for ref in row["evidence"]:
            suffix = f"; JSON pointer `{ref['pointer']}`" if "pointer" in ref else f"; {ref['source_note']}"
            lines.append(f"- [{Path(ref['package_path']).name}]({base}{ref['package_path']}){suffix}")
        lines.append("")
    return "\n".join(lines)


def validate_review(review, manuscript_hash, claims_hash):
    require(review.get("schema_version") == "p63-substantive-review-v1"
        and review.get("status") == "complete", "substantive review missing")
    require(review.get("manuscript_sha256") == manuscript_hash
        and review.get("claims_sha256") == claims_hash, "substantive review is stale")
    required = {"method_and_protocol", "numerical_claims", "causal_limits", "literature", "scope_and_deviations", "reproducibility"}
    rows = review.get("checks", [])
    require({r.get("id") for r in rows} == required and len(rows) == len(required)
        and all(r.get("status") == "reviewed" and bool(r.get("finding")) for r in rows), "substantive review incomplete")
    require(review.get("independent_human_peer_review") is False, "do not imply human peer review")


def audit_links(text, parent):
    checked = []
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        if target.startswith(("https://", "http://", "#", "mailto:")):
            continue
        target = unquote(target.split("#", 1)[0])
        path = (parent / target).resolve()
        require(path.is_relative_to(ROOT) and path.is_file(), f"broken local manuscript link: {target}")
        checked.append(path)
    return checked


def verify_pins(pins):
    for path, digest in pins.items():
        require(path.is_file() and file_sha256(path) == digest, f"evidence changed during audit: {path}")
