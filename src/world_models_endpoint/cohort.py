"""Prospective source freeze and authenticated final-actor handoffs.

Model/controller imports are confined to explicit preparation, training and CPU
audit functions. Final evaluation reads immutable audit receipts and actor bytes.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import re
import uuid

from world_models.data.ledger import atomic_json, file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.bank import validate_test_bank
from world_models_study.protocol import protocol_identity, training_seed_manifest
from . import checkpoints
from .evidence import ROOT, read_report, safe_path, source_identity, timestamp, validate_verification, write_report


SCHEMA = "p5-frozen-endpoint-cohort-v1"
RUN_SCHEMA = "p5-prospective-run-v1"


def _identity(document):
    return canonical_hash({k: v for k, v in document.items() if k != "cohort_identity"})


def _case(variant, seed):
    if variant not in ("B", "D") or type(seed) is not int or seed not in (0, 1):
        raise ValueError("the cohort requires B/D and master seeds 0/1")
    return f"relay/{variant}/seed_{seed}"


def _row(root, variant, seed, inputs):
    return {"case": _case(variant, seed), "variant": variant, "master_seed": seed, "seed": seed,
        "relative_run_dir": f"runs/{variant}/seed_{seed}",
        "run_dir": str(root / f"runs/{variant}/seed_{seed}"),
        "initial_binding_sha256": inputs["per_seed"][str(seed)]["initial_binding_sha256"],
        "training_seeds": training_seed_manifest(seed, variant)}


def _reference(path):
    path = safe_path(path)
    read_report(path)
    return {"path": str(path), "sha256": file_sha256(path)}


def _read_reference(row):
    path = safe_path(row["path"])
    report = read_report(path)
    if file_sha256(path) != row["sha256"]:
        raise ValueError(f"frozen report bytes changed: {path}")
    return report


def _input_files(inputs):
    checkpoints.validate_files(inputs["bound_files"])


def _preserved_sources(recorded, current):
    return bool(recorded) and all(current.get(name) == digest for name, digest in recorded.items())


def _historical_proof(path, *, schema, modules, combined):
    """Authenticate an unchanged earlier suite against its newly rerun cases."""
    r = read_report(path)
    expected = {name for name in combined["passed_tests"] if any(name.startswith(m + ".") for m in modules)}
    passed = r.get("passed_tests", [])
    log = safe_path(path).parent / "tests.txt"
    complete = "complete_" + schema.split("-")[0] + "_implementation"
    if (r.get("schema_version") != schema or r.get("status") != complete
            or r.get("test_modules") != list(modules)
            or set(passed) != expected or len(passed) != len(expected) or r.get("tests_run") != len(expected)
            or not _preserved_sources(r.get("source_sha256"), combined["source_sha256"])
            or r.get("source_unchanged_during_verification") is not True
            or any(type(r.get(k)) is not int or r[k] != 0 for k in ("errors", "failures", "skipped"))
            or r.get("research_training_executed") is not False
            or r.get("research_evaluation_executed") is not False
            or r.get("native_environment_evaluation_executed") is not False
            or r.get("gpu_runtime_measured") is not False
            or r.get("tests_log") != {"path": "tests.txt", "sha256": file_sha256(log)}):
        raise ValueError("historical candidate checks are incomplete or their original sources changed")
    entries = [entry if entry.endswith("." + method) else f"{entry}.{method}"
        for method, entry in re.findall(r"^(\S+) \(([^\n]+)\) \.\.\. ok$", log.read_text(encoding="utf-8"), re.MULTILINE)]
    if len(entries) != len(set(entries)) or set(entries) != expected:
        raise ValueError("historical candidate test log differs")
    return r


def _candidate(matrix, combined):
    from world_models_paired.cohort import plan as candidate_plan
    from world_models_paired.evidence import TEST_MODULES as paired_modules
    from world_models_study.evidence import TEST_MODULES as static_modules
    from world_models_study.protocol import load_protocol
    c = read_report(matrix)
    p = _read_reference(c["registration"])
    if (c.get("schema_version") != "p52-paired-matrix-v1" or c.get("status") != "training_matrix_registered"
            or c.get("candidate_only") is not True or c.get("cohort_frozen") is not False
            or canonical_hash(c.get("plan")) != canonical_hash(candidate_plan())
            or not _preserved_sources(c.get("source_sha256"), combined["source_sha256"])
            or not _preserved_sources(p.get("source_sha256"), combined["source_sha256"])
            or p.get("schema_version") != "p51-protocol-registration-v1" or p.get("status") != "protocol_registered"
            or p.get("protocol") != load_protocol() or p.get("protocol_sha256") != protocol_identity()
            or canonical_hash(p["inputs"]) != c["inputs_sha256"]):
        raise ValueError("candidate is not an unchanged scientific/input registration")
    _historical_proof(c["verification"]["path"], schema="p52-implementation-v1", modules=paired_modules, combined=combined)
    _historical_proof(p["verification"]["path"], schema="p51-implementation-v1", modules=static_modules, combined=combined)
    for parent in (c, p):
        proof = _read_reference(parent["verification"])
        if file_sha256(safe_path(parent["verification"]["path"]).parent / "tests.txt") != parent["verification"]["tests_log_sha256"]:
            raise ValueError("candidate test log binding changed")
    return c, p


def freeze_cohort(output_dir, *, matrix, verification_report, confirm_test_bank_unused=False):
    """Freeze one fresh four-run cohort before policies or test outcomes exist."""
    output, matrix, proof = map(safe_path, (output_dir, matrix, verification_report))
    if confirm_test_bank_unused is not True:
        raise ValueError("review prior test exposure and explicitly confirm the registered bank is unused")
    if output.exists():
        raise FileExistsError("prospective cohort requires a fresh directory")
    before = source_identity()
    combined = validate_verification(proof)
    candidate, parent = _candidate(matrix, combined)
    inputs = parent["inputs"]
    bank_path = safe_path(Path(candidate["registration"]["path"]).parent / "test_bank.json")
    bank = validate_test_bank(read_report(bank_path))
    protected = [ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios")]
    protected += [matrix.parent, proof.parent]
    protected += [safe_path(x["path"]).parent for x in inputs["bound_files"]]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in protected):
        raise ValueError("cohort output must be separate from sources and existing evidence/inputs")
    # Candidate/development runs never become this new cohort's fresh policies.
    for row in candidate["plan"]["run_matrix"]:
        if (matrix.parent / row["relative_run_dir"]).exists():
            raise ValueError("use an untrained candidate matrix; existing development runs remain separate")
    for claim in (ROOT / "artifacts").glob("**/endpoint_test_claim.json"):
        record = read_report(claim)
        if record.get("bank_sha256") == canonical_hash(bank):
            raise ValueError("this final test bank already has an execution claim; preserve its original cohort")
    doc = {"schema_version": SCHEMA, "status": "cohort_frozen", "created_at": timestamp(),
        "report_path": str(output / "cohort_manifest.json"), "source_sha256": before,
        "protocol_sha256": protocol_identity(), "parent_study_profile": "relay_four_day_v1",
        "study_profile": "relay_endpoint_protocol_v1", "candidate_matrix": _reference(matrix),
        "verification": _reference(proof), "registration": candidate["registration"],
        "inputs": deepcopy(inputs), "input_roots": parent["input_roots"], "bank": bank,
        "bank_sha256": canonical_hash(bank), "test_bank_sha256": canonical_hash(bank),
        "run_matrix": [_row(output, v, s, inputs) for s in (0, 1) for v in ("B", "D")],
        "capabilities": {"ledger": True, "final_test_evaluator": True, "paired_endpoint_reader": True},
        "test_bank_nonuse": {"operator_attested": True, "scope": "prior exposure reviewed before comparative training",
                             "automatically_proven": False},
        "cohort_frozen": True, "execution_ready": True, "candidate_only": False,
        "candidate_source_relation": "preserved original files plus additive endpoint implementation",
        "research_training_executed_at_freeze": False, "research_evaluation_executed_at_freeze": False,
        "automatic_training_launch": False, "cohort_acceptance": False,
        "historical_runs_promoted": False, "required_runs": 4, "required_episode_records": 80}
    doc["cohort_identity"] = _identity(doc)
    _input_files(inputs)
    validate_verification(proof)
    if before != source_identity() or _reference(matrix) != doc["candidate_matrix"]:
        raise ValueError("cohort sources or matrix changed during freeze")
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "cohort_manifest.json", doc)
    return doc


def inspect_cohort(path, *, check_inputs=False):
    """Authenticate metadata without importing training tests or model classes."""
    path = safe_path(path)
    r = read_report(path)
    if (r.get("schema_version") != SCHEMA or r.get("status") != "cohort_frozen"
            or r.get("report_path") != str(path) or r.get("cohort_identity") != _identity(r)
            or r.get("source_sha256") != source_identity() or r.get("protocol_sha256") != protocol_identity()
            or r.get("capabilities") != {"ledger": True, "final_test_evaluator": True, "paired_endpoint_reader": True}
            or any(r.get(k) is not v for k, v in {"cohort_frozen": True, "execution_ready": True,
                "candidate_only": False, "historical_runs_promoted": False, "cohort_acceptance": False,
                "research_training_executed_at_freeze": False, "research_evaluation_executed_at_freeze": False,
                "automatic_training_launch": False}.items())
            or r.get("required_runs") != 4 or r.get("required_episode_records") != 80
            or r.get("test_bank_nonuse", {}).get("operator_attested") is not True):
        raise ValueError("frozen cohort sources, protocol or prospective identity differ")
    validate_test_bank(r["bank"])
    if canonical_hash(r["bank"]) != r["bank_sha256"] or r["test_bank_sha256"] != r["bank_sha256"]:
        raise ValueError("frozen cohort bank changed")
    expected = [_row(path.parent, v, s, r["inputs"]) for s in (0, 1) for v in ("B", "D")]
    if canonical_hash(r["run_matrix"]) != canonical_hash(expected):
        raise ValueError("frozen cohort run slots or training RNG namespaces differ")
    candidate, parent = _read_reference(r["candidate_matrix"]), _read_reference(r["registration"])
    proof = _read_reference(r["verification"])
    validate_verification(r["verification"]["path"], discover=False)
    if (not _preserved_sources(candidate["source_sha256"], r["source_sha256"])
            or not _preserved_sources(parent["source_sha256"], r["source_sha256"])
            or canonical_hash(parent["inputs"]) != canonical_hash(r["inputs"])
            or candidate["inputs_sha256"] != canonical_hash(r["inputs"])
            or candidate["registration"] != r["registration"] or proof["source_sha256"] != r["source_sha256"]):
        raise ValueError("cohort parent evidence or initial inputs changed")
    if check_inputs:
        _input_files(r["inputs"])
    return r


def _slot(cohort, variant, seed):
    _case(variant, seed)
    return next(row for row in cohort["run_matrix"] if row["variant"] == variant and row["master_seed"] == seed)


def _binding(cohort, row):
    return {"schema_version": RUN_SCHEMA, "run_kind": "prospective_paired_cohort",
        "cohort_path": cohort["report_path"], "cohort_identity": cohort["cohort_identity"],
        "protocol_sha256": cohort["protocol_sha256"], "case": row["case"],
        "variant": row["variant"], "master_seed": row["master_seed"], "run_dir": row["run_dir"],
        "initial_binding_sha256": row["initial_binding_sha256"], "historical_adoption": False}


def _backend(variant):
    if variant == "B":
        from world_models_paired.runner import BRunner
        return BRunner
    from world_models_paired.d_runner import PairedDRunner
    return PairedDRunner


def _pairing(cohort, row, value):
    p = cohort["inputs"]["per_seed"][str(row["master_seed"])]["initial_binding"]
    expected = {"input_binding_sha256": canonical_hash(p),
        "world_model_checkpoint_sha256": p["selection"]["checkpoint_sha256"],
        "normalization_identity_sha256": p["normalization"]["identity_sha256"],
        "replay_manifest_sha256": p["provenance"]["replays"][0]["manifest_sha256"]}
    if (type(value) is not dict or set(value) != {*expected, "actor_cpu", "critic_cpu"}
            or any(value.get(k) != v for k, v in expected.items())
            or any(type(value.get(k)) is not str or len(value[k]) != 64 for k in ("actor_cpu", "critic_cpu"))):
        raise ValueError("prepared actor/critic or frozen model/data pairing differs")
    other = _slot(cohort, "D" if row["variant"] == "B" else "B", row["master_seed"])
    reference = Path(other["run_dir"]) / "endpoint_run.json"
    if reference.is_file() and read_report(reference).get("pairing") != value:
        raise ValueError("B/D initial actor, critic or frozen inputs are not paired")


def prepare_run(cohort, *, variant, seed, device="cuda:0"):
    from world_models.training.runner import run_lock
    c = inspect_cohort(cohort, check_inputs=True)
    row = _slot(c, variant, seed)
    directory, roots = safe_path(row["run_dir"]), c["input_roots"]
    context = _binding(c, row)
    expected = c["inputs"]["per_seed"][str(seed)]["initial_binding"]
    with run_lock(Path(c["report_path"]).parent):
        if (directory / "endpoint_run.sha256").exists():
            raise FileExistsError("run already prepared; resume it with run")
        backend = _backend(variant)
        if directory.exists():
            worker = (backend.resume(directory, device=device) if variant == "B" else
                backend.recover_prepared(directory, device=device, cohort_identity=c["cohort_identity"], input_binding=expected))
            durable = backend.inspect(directory)
            if (durable.get("local_cycles_completed") != 0 or durable.get("units_completed", 0) != 0
                    or variant == "B" and worker.manifest.get("cohort_binding") != context):
                raise ValueError("only this cohort's interrupted zero-work preparation can be recovered")
        elif variant == "B":
            worker = backend.prepare(directory, initial_selection=roots["initial_selection"], master_seed=seed,
                data_root=roots["data_root"], statistics_root=roots["statistics_root"], config_root=roots["config_root"],
                device=device, precision="bfloat16", cohort_binding=context)
        else:
            worker = backend.prepare(directory, seed=seed, initial_selection=roots["initial_selection"],
                replay_root=roots["data_root"], normalization_dir=roots["statistics_root"], config_root=roots["config_root"],
                device=device, precision="bfloat16", cohort_identity=c["cohort_identity"], input_binding=expected)
        pairing = worker.pairing()
        _pairing(c, row, pairing)
        manifest = directory / "run_manifest.json"
        read_report(manifest)
        _input_files(c["inputs"])
        if c["source_sha256"] != source_identity():
            raise ValueError("sources changed during prospective preparation")
        partial = directory / "endpoint_run.json"
        if partial.exists():
            saved = safe_path(directory / "uncommitted")
            saved.mkdir(exist_ok=True)
            partial.rename(saved / f"endpoint_run_{uuid.uuid4().hex}.json")
        write_report(directory / "endpoint_run.json", {**context, "pairing": pairing,
            "learner_manifest": _reference(manifest), "prepared_at": timestamp()})
    return audit_run(directory)


def _context(run_dir):
    directory = safe_path(run_dir)
    binding = read_report(directory / "endpoint_run.json")
    c = inspect_cohort(binding["cohort_path"])
    row = _slot(c, binding["variant"], binding["master_seed"])
    expected = _binding(c, row)
    if (directory != safe_path(row["run_dir"])
            or any(binding.get(k) != v for k, v in expected.items())):
        raise ValueError("run does not belong to this prospective cohort slot")
    manifest = _read_reference(binding["learner_manifest"])
    if safe_path(binding["learner_manifest"]["path"]) != directory / "run_manifest.json":
        raise ValueError("learner manifest path escapes its run")
    _pairing(c, row, binding["pairing"])
    if row["variant"] == "B" and manifest.get("cohort_binding") != expected:
        raise ValueError("B learner belongs to a different cohort")
    return directory, binding, c, manifest


def _audits(directory):
    rows = sorted((directory / "endpoint_audits").glob("audit_*.json"))
    return [p for p in rows if p.with_suffix(".sha256").is_file()]


def _preserve_uncommitted_audits(directory):
    """Recover an interrupted JSON/receipt publication under the audit lock."""
    root = safe_path(directory / "endpoint_audits")
    for receipt in root.glob("audit_*.sha256"):
        if not receipt.with_suffix(".json").is_file():
            raise ValueError("committed audit receipt is missing its JSON")
    for partial in root.glob("audit_*.json"):
        if partial.with_suffix(".sha256").exists():
            read_report(partial)  # A present, invalid receipt is never discarded.
            continue
        saved = safe_path(root / "uncommitted")
        saved.mkdir(exist_ok=True)
        safe_path(partial).rename(saved / f"{partial.stem}_{uuid.uuid4().hex}.json")


def audit_run(run_dir):
    """Read original durable CPU state, then publish a metadata-only handoff."""
    from world_models.training.runner import run_lock
    directory, binding, c, manifest = _context(run_dir)
    with run_lock(directory / "endpoint_audits"):
        _preserve_uncommitted_audits(directory)
        parent = checkpoints.latest(directory)
        durable = _backend(binding["variant"]).inspect(directory)
        rows = checkpoints.files(parent)
        initial_commits = sorted((directory / "checkpoints").glob("state_00000001_*"))
        initial_commits = [p for p in initial_commits if (p / "manifest.sha256").is_file()]
        if len(initial_commits) != 1:
            raise ValueError("run requires one immutable initial checkpoint")
        rows += checkpoints.files(initial_commits[0])
        rows += [_reference(directory / "endpoint_run.json"), _reference(directory / "run_manifest.json")]
        stage_evidence, stage_directories = {}, {}
        variant, seed = binding["variant"], binding["master_seed"]
        local = durable["local_cycles_completed"]
        counts = {"local_cycles_completed": local, "target_local_cycles": 20,
            "local_actor_optimizer_steps": local * 16, "local_critic_optimizer_steps": local * 16,
            "teacher_cycles_completed": durable.get("teacher_cycles_completed", 0),
            "target_teacher_cycles": 25 if variant == "D" else 0,
            "teacher_actor_optimizer_steps": durable.get("teacher_actor_optimizer_steps", 0),
            "teacher_critic_optimizer_steps": durable.get("teacher_critic_optimizer_steps", 0),
            "trajectories_completed": durable.get("trajectories_completed", 0),
            "distillation_optimizer_steps": durable.get("distillation_optimizer_steps", 0),
            "imagined_joint_steps_local": durable.get("imagined_joint_steps", 0) if variant == "B" else 0,
            "imagined_joint_steps_teacher": 0, "imagined_joint_steps_buffer": 0,
            "real_joint_transitions": 50000, "model_updates": 4500,
            "new_environment_interactions": 0, "new_model_updates": 0}
        if variant == "D":
            initial_student = safe_path(manifest["initial_student"]["path"])
            if initial_student != directory / "initial_student":
                raise ValueError("D initial student checkpoint escapes its run")
            rows += checkpoints.files(initial_student)
            rows += [_reference(directory / "d_binding.json"),
                     {"path": str(directory / "d_binding.sha256"),
                      "sha256": file_sha256(directory / "d_binding.sha256")}]
            # The small parent payload records the exact child commits it owns;
            # child state may have advanced one unit after an interrupted parent.
            from world_models.cycle.checkpoints import load_cycle_checkpoint
            saved, _ = load_cycle_checkpoint(parent)
            stage_evidence = saved["controller"]["stage_evidence"]
            stage_directories = manifest["paths"]
            for stage, evidence in stage_evidence.items():
                rows += checkpoints.files(evidence["checkpoint"])
                rows.append(_reference(Path(evidence["path"]) / "run_manifest.json"))
                for pub in evidence.get("publications", []):
                    # Entries also include payload files, not only JSON reports.
                    path = safe_path(pub["path"])
                    if file_sha256(path) != pub["sha256"]:
                        raise ValueError("child publication changed after its parent commit")
                    rows.append({"path": str(path), "sha256": pub["sha256"]})
            count = lambda stage: stage_evidence.get(stage, {}).get("counters", {}).get("imagined_joint_steps", 0)
            counts.update(imagined_joint_steps_local=count("local_cycle"), imagined_joint_steps_teacher=count("teacher"),
                          imagined_joint_steps_buffer=count("distillation") or count("buffer"))
        complete = durable["status"] == "complete"
        actor_export = None
        if complete:
            publication = directory / ("final_student.json" if variant == "B" else "completed_student.json")
            final = read_report(publication)
            actor_export = str(safe_path(directory / final["actor_export"])) if variant == "B" else final["actor_export"]["path"]
            rows += checkpoints.files(actor_export)
            rows.append(_reference(publication))
            if variant == "D":
                rows.append(_reference(directory / "accounting.json"))
        if parent != checkpoints.latest(directory):
            raise RuntimeError("training advanced during audit; retry status after this unit")
        files = list({row["path"]: row for row in rows}.values())
        checkpoints.validate_files(files)
        record = {"schema_version": "p5-durable-training-audit-v1", "case": binding["case"],
            "cohort_path": c["report_path"], "cohort_identity": c["cohort_identity"],
            "run_dir": str(directory), "variant": variant, "master_seed": seed, "run_seed": seed,
            "run_manifest": manifest, "run_identity": canonical_hash(manifest), "run_binding": binding,
            "initial_binding": c["inputs"]["per_seed"][str(seed)]["initial_binding"],
            "status": "complete" if complete else durable["status"], "phase": durable["phase"],
            "last_checkpoint": str(parent), "actor_export": actor_export, "counts": counts,
            "stage_evidence": stage_evidence, "stage_directories": stage_directories,
            "files": files, "training_complete": complete, "native_evaluation_executed": False}
        previous = _audits(directory)
        previous_doc = read_report(previous[-1]) if previous else None
        if previous_doc is not None:
            if canonical_hash(previous_doc["audit"]) == canonical_hash(record):
                return inspect_run(directory)
            if any(counts[k] < previous_doc["audit"]["counts"][k] for k in counts if type(counts[k]) is int):
                raise ValueError("durable training counters regressed")
        path = directory / "endpoint_audits" / f"audit_{len(previous)+1:08d}.json"
        write_report(path, {"schema_version": "p5-audit-publication-v1", "sequence": len(previous) + 1,
            "previous": None if not previous else _reference(previous[-1]), "created_at": timestamp(), "audit": record})
    return inspect_run(directory)


def inspect_run(run_dir):
    """Read audited bytes only; safe inside an actor-only evaluator process."""
    directory, binding, c, manifest = _context(run_dir)
    rows = _audits(directory)
    if not rows:
        raise ValueError("training audit is missing; run scripts.phase5 status first")
    previous = None
    for sequence, path in enumerate(rows, 1):
        pub = read_report(path)
        if (pub.get("schema_version") != "p5-audit-publication-v1" or pub.get("sequence") != sequence
                or pub.get("previous") != previous):
            raise ValueError("training audit publication chain differs")
        previous = _reference(path)
    audit = pub["audit"]
    if (audit["run_binding"] != binding or audit["run_identity"] != canonical_hash(manifest)
            or audit["cohort_identity"] != c["cohort_identity"]
            or safe_path(audit["last_checkpoint"]) != checkpoints.latest(directory)):
        raise ValueError("training audit is behind the checkpoint; run scripts.phase5 status to refresh")
    checkpoints.validate_files(audit["files"])
    return {**audit, **audit["counts"], "audit_path": str(rows[-1]),
            "audit_sha256": file_sha256(rows[-1]), "protocol_sha256": c["protocol_sha256"]}


def _all_prepared(c):
    for row in c["run_matrix"]:
        _context(row["run_dir"])


class TrainingRun:
    def __init__(self, worker, directory):
        self.worker, self.directory = worker, directory

    def request_stop(self):
        self.worker.request_stop()

    def run(self, *, max_units=None):
        directory, binding, c, _ = _context(self.directory)
        _all_prepared(c)
        inspect_cohort(c["report_path"], check_inputs=True)
        if self.worker.pairing() != binding["pairing"]:
            raise ValueError("restored initial pairing changed")
        try:
            if binding["variant"] == "B":
                self.worker.run(max_cycles=max_units)
            else:
                self.worker.run(max_units=max_units)
        except BaseException:
            try:
                audit_run(directory)
            except (OSError, ValueError, RuntimeError):
                pass
            raise
        return audit_run(directory)


def resume_run(run_dir, *, device="cuda:0"):
    directory, binding, c, _ = _context(run_dir)
    expected = c["inputs"]["per_seed"][str(binding["master_seed"])]["initial_binding"]
    kwargs = {} if binding["variant"] == "B" else {"cohort_identity": c["cohort_identity"], "input_binding": expected}
    worker = _backend(binding["variant"]).resume(directory, device=device, **kwargs)
    return TrainingRun(worker, directory)


def request_stop(run_dir):
    directory, _, _, _ = _context(run_dir)
    atomic_json(directory / "stop.request", {"requested_at": timestamp()})
    return {"status": "stop_requested", "run_dir": str(directory)}


def final_actor(run_dir):
    """Bind a final actor after all four arms finish, without loading a model."""
    directory, binding, c, _ = _context(run_dir)
    _input_files(c["inputs"])
    all_runs = [inspect_run(row["run_dir"]) for row in c["run_matrix"]]
    if any(r["status"] != "complete" or not r["training_complete"] for r in all_runs):
        raise ValueError("all four registered training runs must finish before final test evaluation")
    run = next(r for r in all_runs if r["run_dir"] == str(directory))
    if (run["local_cycles_completed"] != 20 or run["local_actor_optimizer_steps"] != 320
            or run["local_critic_optimizer_steps"] != 320 or run["new_model_updates"] != 0
            or run["new_environment_interactions"] != 0
            or binding["variant"] == "D" and (run["teacher_cycles_completed"], run["trajectories_completed"],
                                               run["distillation_optimizer_steps"]) != (25, 128, 100)):
        raise ValueError("final actor has incomplete or unequal registered stage budgets")
    export = safe_path(run["actor_export"])
    meta = checkpoints.authenticate(export, actor_only=True)["metadata"]
    if (meta.get("variant") != binding["variant"] or meta.get("run_seed") != binding["master_seed"]
            or meta.get("local_ppo_cycles") != 20 or meta.get("actor_optimizer_steps") != 320
            or meta.get("critic_optimizer_steps") != 320 or meta.get("model_updates") != 4500
            or meta.get("real_joint_transitions") != 50000 or meta.get("evaluation_executed") is not False):
        raise ValueError("final actor export metadata differs from its completed run")
    proof = {"schema_version": "p5-final-actor-binding-v1", "cohort_identity": c["cohort_identity"],
        "protocol_sha256": c["protocol_sha256"], "case": binding["case"],
        "actor_export_manifest": _reference(export / "manifest.json"), "audit": _reference(run["audit_path"]),
        "pairing": binding["pairing"], "all_four_training_complete": True, "local_cycles": 20}
    return {"actor_export": str(export), "variant": binding["variant"], "master_seed": binding["master_seed"],
        "cohort_path": c["report_path"], "cohort_identity": c["cohort_identity"],
        "protocol_sha256": c["protocol_sha256"], "bank": c["bank"], "bank_sha256": c["bank_sha256"],
        "binding": proof, "source_sha256": c["source_sha256"], "run_dir": str(directory),
        "case": binding["case"], "initial_binding_sha256": binding["initial_binding_sha256"]}
