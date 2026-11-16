"""Prepare/check the completed manuscript locally from saved evidence only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import sys

sys.dont_write_bytecode = True
from . import audit
from .audit import ROOT, MANUSCRIPT, CLAIMS, REVIEW, GENERATED, require
from .tables import replace_block
from study_release.evidence import file_sha256, write_report, read_report, timestamp


def source_identity():
    paths = list((ROOT / "study_paper").rglob("*.py")) + [ROOT / "study_paper/requirements.txt",
        ROOT / "paper/update_development_tables.py"]
    return {p.relative_to(ROOT).as_posix(): file_sha256(p) for p in sorted(paths)}


def prepare(context):
    GENERATED.mkdir(parents=True, exist_ok=True)
    for name, data in context["outputs"].items():
        (GENERATED / name).write_bytes(data)
    text = MANUSCRIPT.read_text(encoding="utf-8")
    for name, content in context["blocks"].items():
        text = replace_block(text, name, content)
    # These existing readers verify the separately scoped historical reports.
    from paper.update_development_tables import replay_table, model_table, development_table, replace_block as old_block
    for name, value in (("REPLAY", replay_table()), ("MODEL_SCREEN", model_table()), ("DEVELOPMENT", development_table())):
        text = old_block(text, name, value)
    MANUSCRIPT.write_text(text, encoding="utf-8", newline="\n")
    if CLAIMS.is_file():
        claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
        audit.audit_claims(claims, text, context["release"], context["manifest"])
        (ROOT / "paper/claim_evidence.md").write_text(audit.claims_markdown(claims), encoding="utf-8", newline="\n")
    return {"status": "prepared", "paper_tables_generated": 8,
        "saved_episode_records_reaggregated": 80, "figures_copied_from_authenticated_reproduction": 2,
        "research_training_executed": False, "research_evaluation_executed": False,
        "next_step": "Review the updated manuscript and bind manuscript_review.json before complete."}


def check_manuscript(context):
    text = MANUSCRIPT.read_text(encoding="utf-8")
    for name, content in context["blocks"].items():
        require(replace_block(text, name, content) == text, f"manuscript table differs: {name}")
    from paper.update_development_tables import replay_table, model_table, development_table, replace_block as old_block
    for name, value in (("REPLAY", replay_table()), ("MODEL_SCREEN", model_table()), ("DEVELOPMENT", development_table())):
        require(old_block(text, name, value) == text, f"development table differs: {name}")
    for name, data in context["outputs"].items():
        require((GENERATED / name).read_bytes() == data, f"generated paper asset differs: {name}")
    claims = json.loads(CLAIMS.read_text(encoding="utf-8"))
    counts = audit.audit_claims(claims, text, context["release"], context["manifest"])
    checklist = ROOT / "paper/claim_evidence.md"
    require(checklist.read_text(encoding="utf-8") == audit.claims_markdown(claims), "readable claim checklist differs")
    audit.validate_review(json.loads(REVIEW.read_text(encoding="utf-8")), file_sha256(MANUSCRIPT), file_sha256(CLAIMS))
    links = audit.audit_links(text, MANUSCRIPT.parent)
    links += audit.audit_links(checklist.read_text(encoding="utf-8"), checklist.parent)
    return {**counts, "local_links_checked": len(links), "generated_table_blocks": 8,
        "automated_semantic_review_claimed": False}, links


def complete(output):
    output = Path(output).absolute()
    require(output.parent == ROOT / "artifacts/phase6_runs" and output.name.startswith("paper_"),
        "use a fresh artifacts/phase6_runs/paper_NAME directory")
    require(not output.exists(), "completion output already exists; preserve earlier review")
    before = source_identity()
    context = audit.load_evidence()
    checked, links = check_manuscript(context)
    result = subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "study_paper/tests", "-v"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
    log = result.stdout + result.stderr
    count = re.search(r"Ran (\d+) tests?", log)
    require(result.returncode == 0 and count is not None and int(count[1]) > 0 and "skipped=" not in log,
        "local audit fixture tests failed:\n" + log)
    deliverables = [MANUSCRIPT, CLAIMS, REVIEW, ROOT / "paper/claim_evidence.md",
        ROOT / "docs/phase63_paper.md", ROOT / "detailed_research_implementation_plan.md"]
    deliverables += [GENERATED / name for name in sorted(context["outputs"])]
    for path in {*links, *deliverables}:
        context["pins"][path] = file_sha256(path)
    # The review can become stale if an editor saves while the tests run.
    checked_again, _ = check_manuscript(context)
    require(checked_again == checked and before == source_identity(), "paper or audit sources changed")
    audit.verify_sources(context["manifest"], context["completion"])
    print("Checking all bound bytes before publishing P6.3 acceptance...", flush=True)
    audit.verify_pins(context["pins"])
    output.mkdir(parents=True)
    (output / "tests.txt").write_text(log, encoding="utf-8", newline="\n")
    proof = {"schema_version": "p63-local-checks-v1", "status": "passed", "tests_run": int(count[1]),
        "failures": 0, "errors": 0, "skipped": 0, "source_sha256": before,
        "tests_sha256": file_sha256(output / "tests.txt"), "checks": checked,
        "scope": "synthetic rejection checks plus actual saved-evidence reaggregation and manuscript audit"}
    write_report(output / "implementation_checks.json", proof)
    snapshot = output / "manuscript_snapshot"
    snapshot.mkdir()
    for path in (MANUSCRIPT, CLAIMS, REVIEW, ROOT / "paper/claim_evidence.md"):
        (snapshot / path.name).write_bytes(path.read_bytes())
    (snapshot / "README.txt").write_text("Preserved P6.3 paper/review bytes. Relative links keep their original project paper/ anchor.\n",
        encoding="utf-8")
    dependencies = {path.relative_to(ROOT).as_posix(): digest for path, digest in sorted(context["pins"].items())}
    dependencies.update(before)
    for path in output.rglob("*"):
        if path.is_file():
            dependencies[path.relative_to(ROOT).as_posix()] = file_sha256(path)
    report = {"schema_version": "p63-phase6-acceptance-v1", "status": "passed", "created_at": timestamp(),
        "cohort_identity": context["manifest"]["cohort_identity"], "protocol_sha256": context["manifest"]["protocol_sha256"],
        "phase6_acceptance": True, "P6.1_complete": True, "P6.2_complete": True, "P6.3_complete": True,
        "P6.3_paper_review_complete": True, "acceptance_scope": "completed reduced study, manuscript and evidence; not learned competence",
        "saved_record_reproduction_verified": True, "current_saved_episodes_reaggregated": 80,
        "six_tables_byte_identical": True, "two_figures_numeric_data_verified": True,
        "new_graphics_rendering_executed": False, "checks": checked, "tests_run": int(count[1]),
        "manuscript": MANUSCRIPT.relative_to(ROOT).as_posix(), "manuscript_sha256": file_sha256(MANUSCRIPT),
        "substantive_review": REVIEW.relative_to(ROOT).as_posix(), "reviewer": "Codex-assisted evidence and manuscript review",
        "independent_human_peer_review": False, "successful_transfer_established": False,
        "B_versus_D_superiority_established": False, "P2.5_quality_acceptance": False,
        "independent_retraining_without_original_archive": False,
        "research_training_executed": False, "research_evaluation_executed": False, "native_environment_opened": False,
        "new_environment_interactions": 0, "new_model_updates": 0, "new_policy_optimizer_updates": 0,
        "reasons": [], "source_sha256": before, "bound_files_sha256": dependencies,
        "report_path": str(output / "phase6_acceptance.json")}
    # Small final concurrent-edit guard after writing only new output files.
    for path in deliverables:
        require(file_sha256(path) == context["pins"][path], f"deliverable changed during publication: {path}")
    write_report(output / "phase6_acceptance.json", report)
    return report


def inspect(path):
    report = read_report(Path(path))
    require(report.get("schema_version") == "p63-phase6-acceptance-v1" and report.get("status") == "passed"
        and report.get("phase6_acceptance") is True, "not a passed P6.3 report")
    for name, digest in report["bound_files_sha256"].items():
        require(file_sha256(audit.relative_file(ROOT, name)) == digest, f"accepted evidence/document changed: {name}")
    require(source_identity() == report["source_sha256"], "P6.3 source set changed")
    # Hashes of listed files alone cannot detect newly added release/source files.
    manifest = audit.inspect_release(audit.COMPLETION / "release")
    completion = read_report(audit.COMPLETION / "phase61_62_completion.json")
    audit.verify_sources(manifest, completion)
    require(report["cohort_identity"] == manifest["cohort_identity"]
        and report["protocol_sha256"] == manifest["protocol_sha256"], "accepted cohort/protocol changed")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    run = sub.add_parser("complete")
    run.add_argument("--output-dir", required=True)
    check = sub.add_parser("inspect")
    check.add_argument("--report", required=True)
    args = parser.parse_args(argv)
    try:
        result = prepare(audit.load_evidence()) if args.command == "prepare" else (
            complete(args.output_dir) if args.command == "complete" else inspect(args.report))
        print(json.dumps({k: result[k] for k in ("status", "phase6_acceptance", "P6.3_complete", "tests_run",
            "successful_transfer_established", "report_path", "next_step") if k in result}, indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError, IndexError, RuntimeError, ImportError) as error:
        print(f"P6.3 error ({type(error).__name__}): {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
