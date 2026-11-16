"""Four immutable candidate slots, shared inputs and explicit training boundaries.

P5.2 supplies training machinery. P5.4/P5.5 must exist before a prospective
comparative cohort can be frozen. Explicit development runs never acquire that
status retrospectively, even if all their budgets complete.
"""
from __future__ import annotations

from pathlib import Path
import re
import uuid

from world_models.data.ledger import atomic_json, file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models_study.evidence import ROOT, read_report, safe_path, source_identity, timestamp, write_report
from world_models_study.protocol import load_protocol, protocol_identity, training_seed_manifest
from world_models_study.registration import inspect_registration
from .evidence import validate_verification


PENDING = ["P5.4 endpoint evaluator", "P5.5 paired endpoint reader",
           "prospective cohort eligibility and final source freeze"]
MARKERS = {"candidate_only": True, "training_tools_ready": True, "execution_ready": False,
           "cohort_frozen": False, "fresh_cohort_eligible": False, "cohort_acceptance": False,
           "research_training_executed_at_registration": False,
           "research_evaluation_executed": False, "automatic_training_launch": False}


def _case(seed, variant):
    if type(seed) is not int or seed not in (0, 1) or variant not in ("B", "D"):
        raise ValueError("the fixed matrix permits Relay seeds 0/1 and B/D only")
    return f"relay/{variant}/seed_{seed}"


def plan():
    p = load_protocol()
    rows = []
    for seed in (0, 1):
        for variant in ("B", "D"):
            rows.append({"case": _case(seed, variant), "variant": variant, "seed": seed,
                "relative_run_dir": f"runs/{variant}/seed_{seed}", "initial_transitions": 50000,
                "initial_model_updates": 4500, "new_environment_interactions": 0, "new_model_updates": 0,
                "local_cycles": 20, "local_actor_optimizer_steps": 320, "local_critic_optimizer_steps": 320,
                "teacher_cycles": 25 if variant == "D" else 0,
                "teacher_actor_optimizer_steps": 400 if variant == "D" else 0,
                "teacher_critic_optimizer_steps": 400 if variant == "D" else 0,
                "distillation_trajectories": 128 if variant == "D" else 0,
                "distillation_optimizer_steps": 100 if variant == "D" else 0,
                "training_rng": training_seed_manifest(seed, variant)})
    return {"schema_version": "p52-paired-matrix-plan-v1", "protocol_sha256": protocol_identity(),
        "study_profile": p["study_profile"], "run_matrix": rows,
        "physical_reused_initial_joint_transitions": 100000,
        "sum_per_run_attributed_initial_joint_transitions": 200000,
        "physical_selected_model_updates": 9000, "sum_per_run_attributed_model_updates": 18000,
        "pending": list(PENDING), **MARKERS}


def _guard_output(output, parent, verification):
    if output.exists():
        raise FileExistsError("the paired matrix needs a fresh directory")
    protected = [ROOT / name for name in ("src", "scripts", "tests", "configs", "scenarios")]
    protected += [parent.parent, verification.parent]
    protected += [safe_path(row["path"]).parent for row in read_report(parent)["inputs"]["bound_files"]]
    if any(output.is_relative_to(path) or path.is_relative_to(output) for path in protected):
        raise ValueError("paired output must be separate from inputs, sources and verification")


def _recheck_files(candidate):
    rows = candidate["inputs"].get("bound_files")
    if type(rows) is not list or not rows:
        raise ValueError("paired registration has no bound input files")
    seen = set()
    for row in rows:
        path = safe_path(row["path"])
        if (str(path) != row["path"] or str(path) in seen or not path.is_file()
                or file_sha256(path) != row["sha256"]):
            raise ValueError(f"paired immutable input changed: {path}")
        seen.add(str(path))


def register_cohort(output_dir, *, registration, verification_report):
    """Bind candidate slots only; does not prepare a network or execute a learner."""
    output, parent, proof = map(safe_path, (output_dir, registration, verification_report))
    _guard_output(output, parent, proof)
    before = source_identity()
    validate_verification(proof)
    candidate = inspect_registration(parent)
    document = {"schema_version": "p52-paired-matrix-v1", "status": "training_matrix_registered",
        "created_at": timestamp(), "protocol_sha256": protocol_identity(), "plan": plan(),
        "source_sha256": before, "registration": {"path": str(parent), "sha256": file_sha256(parent)},
        "verification": {"path": str(proof), "sha256": file_sha256(proof),
                         "tests_log_sha256": file_sha256(proof.parent / "tests.txt")},
        "inputs_sha256": canonical_hash(candidate["inputs"]),
        "test_bank_sha256": candidate["test_bank_sha256"],
        "pending": list(PENDING), "report_path": str(output / "cohort_manifest.json"), **MARKERS}
    validate_verification(proof)
    _recheck_files(candidate)
    if (before != source_identity() or file_sha256(parent) != document["registration"]["sha256"]
            or file_sha256(proof) != document["verification"]["sha256"]
            or file_sha256(proof.parent / "tests.txt") != document["verification"]["tests_log_sha256"]):
        raise ValueError("paired registration inputs or sources changed during publication")
    output.mkdir(parents=True, exist_ok=False)
    write_report(output / "cohort_manifest.json", document)
    return document


def inspect_cohort(path):
    path = safe_path(path)
    r = read_report(path)
    if (r.get("schema_version") != "p52-paired-matrix-v1"
            or r.get("status") != "training_matrix_registered" or r.get("report_path") != str(path)
            or canonical_hash(r.get("plan")) != canonical_hash(plan())
            or r.get("protocol_sha256") != protocol_identity()
            or r.get("source_sha256") != source_identity() or r.get("pending") != PENDING
            or any(type(r.get(k)) is not type(v) or r.get(k) != v for k, v in MARKERS.items())):
        raise ValueError("paired matrix sources, slots or candidate-only boundary changed")
    parent = safe_path(r["registration"]["path"])
    candidate = inspect_registration(parent)
    if (file_sha256(parent) != r["registration"]["sha256"]
            or canonical_hash(candidate["inputs"]) != r["inputs_sha256"]
            or candidate["test_bank_sha256"] != r["test_bank_sha256"]):
        raise ValueError("paired matrix parent inputs or test bank differ")
    proof = safe_path(r["verification"]["path"])
    validate_verification(proof)
    if (file_sha256(proof) != r["verification"]["sha256"]
            or file_sha256(proof.parent / "tests.txt") != r["verification"]["tests_log_sha256"]):
        raise ValueError("paired matrix verification differs")
    return r


def _backend(variant):
    if variant == "B":
        from .runner import BRunner
        return BRunner
    from .d_runner import PairedDRunner
    return PairedDRunner


def _binding(cohort, variant, seed, directory):
    return {"schema_version": "p52-candidate-run-binding-v1", "case": _case(seed, variant),
        "matrix": {"path": cohort["report_path"], "sha256": file_sha256(cohort["report_path"])},
        "matrix_identity": canonical_hash(cohort), "protocol_sha256": protocol_identity(),
        "variant": variant, "master_seed": seed, "run_dir": str(directory),
        "run_kind": "explicit_development_candidate", "fresh_cohort_eligible": False,
        "research_evaluation_executed": False, "execution_ready": False}


def _slot(matrix, variant, seed):
    _case(seed, variant)
    return safe_path(Path(matrix["report_path"]).parent / f"runs/{variant}/seed_{seed}")


def _expected_inputs(matrix, seed):
    parent = read_report(matrix["registration"]["path"])
    return parent["input_roots"], parent["inputs"]["per_seed"][str(seed)]["initial_binding"]


def _pair_check(directory, pairing, *, matrix, seed, variant):
    keys = {"actor_cpu", "critic_cpu", "input_binding_sha256", "world_model_checkpoint_sha256",
            "normalization_identity_sha256", "replay_manifest_sha256"}
    if (type(pairing) is not dict or set(pairing) != keys
            or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in pairing.values())):
        raise ValueError("candidate pairing must bind exact initial actor, critic, model, replay and normalization")
    other = _slot(matrix, "D" if variant == "B" else "B", seed) / "matrix_run.json"
    if other.is_file():
        sibling = read_report(other)
        if (sibling.get("matrix_identity") != canonical_hash(matrix)
                or sibling.get("pairing") != pairing):
            raise ValueError("paired B/D initial actor, critic or frozen inputs differ")


def prepare_run(run_dir, *, cohort, variant, seed, device="cuda:0", development_only=False):
    _case(seed, variant)
    matrix = inspect_cohort(cohort)
    if development_only is not True:
        raise ValueError("comparative training is pending P5.4/P5.5 and final cohort freeze; "
                         "--development-only explicitly labels a separate development candidate")
    directory = safe_path(run_dir)
    if directory != _slot(matrix, variant, seed):
        raise ValueError("run directory must be the exact registered matrix slot")
    from world_models.training.runner import run_lock
    # Serialize initial publication within a matrix, not the independent learners.
    with run_lock(Path(matrix["report_path"]).parent):
        if (directory / "matrix_run.sha256").exists():
            raise FileExistsError("a paired slot already exists; use run to resume, never prepare again")
        roots, expected = _expected_inputs(matrix, seed)
        binding = _binding(matrix, variant, seed, directory)
        backend = _backend(variant)
        recovering = directory.exists()
        if recovering:
            if variant == "B":
                worker = backend.resume(directory, device=device)
                if worker.manifest.get("cohort_binding") != binding:
                    raise ValueError("interrupted B preparation belongs to a different matrix")
            else:
                worker = backend.recover_prepared(directory, device=device,
                    cohort_identity=canonical_hash(matrix), input_binding=expected)
            durable = backend.inspect(directory)
            if (durable.get("local_cycles_completed") != 0
                    or durable.get("phase") not in ("local", "teacher")
                    or any(durable.get(key, 0) != 0 for key in ("units_completed", "actor_optimizer_steps",
                        "critic_optimizer_steps", "teacher_cycles_completed", "distillation_optimizer_steps"))):
                raise ValueError("only an interrupted zero-work preparation can recover a missing matrix binding")
        elif variant == "B":
            worker = backend.prepare(directory, initial_selection=roots["initial_selection"], master_seed=seed,
                data_root=roots["data_root"], statistics_root=roots["statistics_root"],
                config_root=roots["config_root"], device=device, precision="bfloat16", cohort_binding=binding)
        else:
            worker = backend.prepare(directory, seed=seed, initial_selection=roots["initial_selection"],
                replay_root=roots["data_root"], normalization_dir=roots["statistics_root"],
                config_root=roots["config_root"], device=device, precision="bfloat16",
                cohort_identity=canonical_hash(matrix), input_binding=expected)
        pairing = worker.pairing()
        if pairing.get("input_binding_sha256") != canonical_hash(expected):
            raise ValueError("candidate learner inputs differ from the registered seed")
        _pair_check(directory, pairing, matrix=matrix, seed=seed, variant=variant)
        _recheck_files(read_report(matrix["registration"]["path"]))
        if matrix["source_sha256"] != source_identity():
            raise ValueError("paired sources changed while preparing the learner")
        learner_manifest = directory / "run_manifest.json"
        read_report(learner_manifest)
        partial = directory / "matrix_run.json"
        if partial.exists():
            debris = safe_path(directory / "uncommitted")
            debris.mkdir(exist_ok=True)
            partial.replace(debris / f"matrix_run_{uuid.uuid4().hex}.json")
        write_report(directory / "matrix_run.json", {**binding, "pairing": pairing,
            "learner_manifest": {"path": str(learner_manifest), "sha256": file_sha256(learner_manifest)}})
    return {**worker.status(), "run_dir": str(directory), "run_kind": binding["run_kind"],
            "training_executed": False, "fresh_cohort_eligible": False, "execution_ready": False}


def _run_context(run_dir, *, authenticate_matrix=True):
    directory = safe_path(run_dir)
    binding = read_report(directory / "matrix_run.json")
    variant, seed = binding.get("variant"), binding.get("master_seed")
    _case(seed, variant)
    path = safe_path(binding["matrix"]["path"])
    matrix = inspect_cohort(path) if authenticate_matrix else read_report(path)
    expected = _binding(matrix, variant, seed, directory)
    if (directory != _slot(matrix, variant, seed)
            or {k: v for k, v in binding.items() if k not in ("pairing", "learner_manifest")} != expected):
        raise ValueError("candidate run binding, seed or registered slot differs")
    learner_manifest = directory / "run_manifest.json"
    read_report(learner_manifest)
    if binding.get("learner_manifest") != {"path": str(learner_manifest), "sha256": file_sha256(learner_manifest)}:
        raise ValueError("candidate learner manifest differs from its original matrix registration")
    _pair_check(directory, binding.get("pairing"), matrix=matrix, seed=seed, variant=variant)
    return directory, binding, matrix


class _Run:
    def __init__(self, worker, directory, binding):
        self.worker, self.directory, self.binding = worker, directory, binding

    def request_stop(self):
        self.worker.request_stop()

    def status(self):
        return self.worker.status()

    def run(self, *, max_units=None):
        _, current, _ = _run_context(self.directory)
        if current != self.binding or self.worker.pairing() != current["pairing"]:
            raise ValueError("candidate pairing changed before training")
        if current["variant"] == "B":
            return self.worker.run(max_cycles=max_units)
        return self.worker.run(max_units=max_units)


def resume_run(run_dir, *, device="cuda:0"):
    directory, binding, matrix = _run_context(run_dir)
    variant, seed = binding["variant"], binding["master_seed"]
    _, expected = _expected_inputs(matrix, seed)
    backend = _backend(variant)
    kwargs = {} if variant == "B" else {"cohort_identity": canonical_hash(matrix), "input_binding": expected}
    worker = backend.resume(directory, device=device, **kwargs)
    if variant == "B" and worker.manifest.get("cohort_binding") != {
            key: value for key, value in binding.items() if key not in ("pairing", "learner_manifest")}:
        raise ValueError("restored B learner belongs to a different matrix")
    if worker.pairing() != binding["pairing"]:
        raise ValueError("restored learner initial pairing differs")
    return _Run(worker, directory, binding)


def read_status(run_dir):
    directory, binding, _ = _run_context(run_dir, authenticate_matrix=False)
    variant = binding["variant"]
    worker_status = _backend(variant).inspect(directory)
    return {**worker_status, "case": binding["case"], "run_kind": binding["run_kind"],
        "local_cycles_remaining": worker_status["target_local_cycles"] - worker_status["local_cycles_completed"],
        "fresh_cohort_eligible": False, "execution_ready": False,
        "progress_source": "committed checkpoint; live status is advisory"}


def request_stop(run_dir):
    directory, binding, _ = _run_context(run_dir, authenticate_matrix=False)
    atomic_json(directory / "stop.request", {"requested_at": timestamp(), "case": binding["case"]})
    return {"status": "stop_requested", "run_dir": str(directory),
            "checkpoint_policy": "finish current complete cycle/stage unit; then stop"}
