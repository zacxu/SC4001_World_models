"""Receipt-bound orchestration of the four fixed-data D stages.

Child checkpoints own tensors, optimizers and RNG. The controller owns stage
order, input pairing, unit accounting and final publication. A child commit
survives a failed parent commit and is reconciled before any further training.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

import torch

from world_models.cycle.checkpoints import _guard_real_replay, load_cycle_checkpoint, save_cycle_checkpoint
from world_models.data.ledger import atomic_json, file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.teacher.runner import _latest, _NUMBERED, _publish_receipt, _root, _runtime
from world_models.training.runner import append_event, run_lock, selected_device
from . import stages
from .protocol import ROOT, PROFILE, load_protocol, protocol_identity, stage_targets


RUN_SCHEMA = "p44-d-campaign-run-v1"
ORDER = ("teacher", "buffer", "distillation", "local_cycle")


def source_identity():
    paths = sorted((ROOT / "src/world_models").rglob("*.py"))
    paths += sorted((ROOT / "src/world_models_campaign").rglob("*.py"))
    paths += [ROOT / f"scripts/phase{number}.py" for number in (41, 42, 43, 44)] + [PROFILE]
    return {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in paths}


def _read(path):
    path = Path(path)
    receipt = path.with_suffix(".sha256")
    if (not path.is_file() or not receipt.is_file() or path.is_symlink() or receipt.is_symlink()
            or receipt.read_text(encoding="ascii").strip() != f"{file_sha256(path)}  {path.name}"):
        raise ValueError(f"campaign publication has no valid receipt: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _publish(path, document):
    if path.with_suffix(".sha256").exists():
        if canonical_hash(_read(path)) != canonical_hash(document):
            raise ValueError(f"committed campaign publication differs: {path}")
        return
    atomic_json(path, document)
    temporary = path.with_suffix(".sha256.tmp")
    if temporary.exists():
        temporary.unlink()
    _publish_receipt(path)


def _manifest(directory):
    manifest = _read(directory / "run_manifest.json")
    if (manifest.get("schema_version") != RUN_SCHEMA or manifest.get("protocol") != load_protocol()
            or manifest.get("protocol_sha256") != protocol_identity()
            or manifest.get("mode") not in ("fresh", "adopted_development")):
        raise ValueError("campaign schema, mode or fixed-data protocol differs")
    return manifest


class CampaignController:
    """One seed's D pipeline; research optimization happens only in run()."""

    def __init__(self, directory, manifest, *, device):
        self.directory, self.manifest, self.device = _root(directory), manifest, torch.device(device)
        self.protocol, self.targets = load_protocol(), stage_targets()
        self.paths = {name: _root(path) for name, path in manifest["paths"].items()}
        self.state = {"sequence": 0, "units_completed": 0, "phase": "teacher", "stage_evidence": {}, "export": None}
        self.last_checkpoint = None
        self._worker = self._worker_path = None
        self._stop_requested = self._failed = False

    @classmethod
    def prepare(cls, directory, *, seed, initial_selection, config_root=ROOT / "configs",
                replay_root=ROOT / "data/initial_replay/pilot",
                normalization_dir=ROOT / "artifacts/phase23/normalization",
                device="cuda:0", precision="bfloat16"):
        if type(seed) is not int or seed not in load_protocol()["master_seeds"]:
            raise ValueError("the fixed-data Relay campaign permits seeds 0 and 1 only")
        if precision not in ("float32", "bfloat16"):
            raise ValueError("campaign precision must be float32 or bfloat16")
        directory = _root(directory)
        _guard_real_replay(directory)
        if directory.exists():
            raise FileExistsError("campaign prepare requires a fresh directory")
        device = selected_device(device, precision)
        inputs = {"master_seed": seed, "initial_selection": str(_root(initial_selection)),
            "config_root": str(_root(config_root)), "data_root": str(_root(replay_root)),
            "statistics_root": str(_root(normalization_dir))}
        # This immutable tensor snapshot is published before any teacher exists.
        initial = stages.save_initial_student(directory / "initial_student", inputs=inputs, mode="fresh")
        paths = {"teacher": str(directory / "stages/teacher"),
            "buffer": str(directory / "stages/distillation"),
            "distillation": str(directory / "stages/distillation"),
            "local_cycle": str(directory / "stages/refinement")}
        manifest = cls._registration(inputs, initial, paths, device, precision, mode="fresh", provenance={})
        worker = cls(directory, manifest, device=device)
        with run_lock(directory):
            _publish(directory / "run_manifest.json", manifest)
            worker._checkpoint()
            worker._status("prepared")
        return worker

    @staticmethod
    def _registration(inputs, initial, paths, device, precision, *, mode, provenance):
        return {"schema_version": RUN_SCHEMA, "protocol": load_protocol(), "protocol_sha256": protocol_identity(),
            "source_sha256": source_identity(), "runtime": _runtime(device), "precision": precision,
            "inputs": inputs, "initial_student": initial, "paths": paths, "mode": mode,
            "adoption_provenance": provenance, "starting_units": sum(stage_targets().values()) if mode != "fresh" else 0,
            "created_at": datetime.now(timezone.utc).isoformat(), "variant": "D", "task_family": "relay",
            "research_training_executed_at_prepare": False, "fresh_stage_initialization": mode == "fresh",
            "fresh_cohort_eligible": False, "cohort_freeze_required": True,
            "phase4_acceptance": False, "matched_reduced_b_controller_implemented": False,
            "endpoint_evaluator_implemented": False}

    @classmethod
    def adopt(cls, directory, *, refinement_run, device="cuda:0"):
        """Register completed historical work without training or relabelling it."""
        directory = _root(directory)
        _guard_real_replay(directory)
        if directory.exists():
            raise FileExistsError("campaign adoption requires a fresh directory")
        lineage = stages.adoption_lineage(refinement_run, device=device)
        source = _read(_root(lineage["paths"]["local_cycle"]) / "run_manifest.json")
        precision = source["precision"]
        device = selected_device(device, precision)
        for path in lineage["paths"].values():
            other = _root(path)
            if directory.is_relative_to(other) or other.is_relative_to(directory):
                raise ValueError("adoption directory must be separate from every source stage")
        initial = stages.save_initial_student(directory / "initial_student", inputs=lineage["inputs"],
            mode="adopted", distillation_run=lineage["paths"]["distillation"])
        manifest = cls._registration(lineage["inputs"], initial, lineage["paths"], device, precision,
            mode="adopted_development", provenance=lineage["provenance"])
        worker = cls(directory, manifest, device=device)
        with run_lock(directory):
            for name in ORDER:
                evidence = stages.inspect_stage(name, worker.paths[name])
                worker._validate_evidence(name, evidence)
                if evidence["completed"] != worker.targets[name]:
                    raise ValueError("adoption requires every source stage to be complete")
                worker.state["stage_evidence"][name] = evidence
            worker.state.update(units_completed=sum(worker.targets.values()), phase="export")
            _publish(directory / "run_manifest.json", manifest)
            worker._checkpoint()
            append_event(directory / "attempts.jsonl", {"event": "completed_history_adopted",
                "mode": "adopted_development", "logical_units_adopted": worker.state["units_completed"],
                "new_training_units": 0, "prior_physical_work": "preserved in source stage journals"})
            worker.state["export"] = worker._publish_export()
            worker.state.update(units_completed=worker.state["units_completed"] + 1, phase="complete")
            worker._checkpoint()
            worker._complete()
            worker._status("complete")
        return worker

    @classmethod
    def resume(cls, directory, *, device="cuda:0"):
        directory = _root(directory)
        with run_lock(directory):
            manifest = _manifest(directory)
            if manifest["source_sha256"] != source_identity():
                raise ValueError("campaign source changed; restore the registered checkout")
            device = selected_device(device, manifest["precision"])
            if manifest["runtime"] != _runtime(device):
                raise ValueError("campaign exact resume requires the same logical device and runtime")
            worker = cls(directory, manifest, device=device)
            checkpoint = _latest(directory)
            saved, metadata = load_cycle_checkpoint(checkpoint)
            if set(saved) != {"controller"}:
                raise ValueError("unsupported campaign checkpoint state")
            worker.state, worker.last_checkpoint = saved["controller"], checkpoint
            worker._verify_initial()
            worker._validate_state()
            if (metadata != worker._metadata()
                    or int(_NUMBERED.fullmatch(checkpoint.name)[1]) != worker.state["sequence"]):
                raise ValueError("campaign checkpoint metadata or sequence differs")
            worker._verify_recorded_evidence()
            return worker

    def _verify_initial(self):
        distillation = self.paths["distillation"]
        initial = stages.verify_initial_student(self.directory / "initial_student",
            distillation_run=distillation if (distillation / "run_manifest.sha256").is_file() else None)
        if canonical_hash(initial) != canonical_hash(self.manifest["initial_student"]):
            raise ValueError("campaign paired initial student or frozen inputs changed")

    def _validate_evidence(self, name, evidence):
        if (type(evidence) is not dict or evidence.get("stage") != name
                or _root(evidence["path"]) != self.paths[name]
                or type(evidence.get("completed")) is not int
                or not 0 <= evidence["completed"] <= self.targets[name]
                or evidence.get("target") != self.targets[name]):
            raise ValueError("campaign child evidence has invalid stage or progress")
        counters = evidence["counters"]
        p = self.protocol
        child = _read(self.paths[name] / "run_manifest.json")
        if name == "teacher":
            if canonical_hash(child.get("inputs")) != canonical_hash(self.manifest["inputs"]):
                raise ValueError("campaign teacher belongs to different frozen inputs or seed")
        else:
            upstream = child.get("teacher_run") if name in ("buffer", "distillation") else child.get("distillation_run")
            if (child.get("master_seed") != self.manifest["inputs"]["master_seed"] or type(upstream) is not str
                    or _root(upstream) != self.paths["teacher" if name in ("buffer", "distillation") else "distillation"]):
                raise ValueError("campaign child belongs to another seed or upstream stage")
        if any(counters.get(key) != value for key, value in {
                "model_updates": p["initial_model_updates"], "real_joint_transitions": p["initial_transitions"],
                "new_model_updates": 0, "new_environment_interactions": 0}.items()):
            raise ValueError("campaign child modified frozen data or model accounting")
        per_cycle = p["epochs"] * p["minibatches"]
        keys = ("teacher_actor_optimizer_steps", "teacher_critic_optimizer_steps") if name == "teacher" else (
               "local_actor_optimizer_steps", "local_critic_optimizer_steps") if name == "local_cycle" else ()
        if any(counters.get(key) != evidence["completed"] * per_cycle for key in keys):
            raise ValueError("campaign child optimizer count differs from complete cycles")
        if name == "distillation" and counters["distillation_optimizer_steps"] != evidence["completed"]:
            raise ValueError("campaign supervised optimizer count differs")

    def _phase(self):
        for name in ORDER:
            if self.state["stage_evidence"].get(name, {}).get("completed", 0) < self.targets[name]:
                return name
        return "complete" if self.state["export"] is not None else "export"

    def _validate_state(self):
        s = self.state
        if (set(s) != {"sequence", "units_completed", "phase", "stage_evidence", "export"}
                or type(s["sequence"]) is not int or s["sequence"] < 0
                or type(s["units_completed"]) is not int or type(s["stage_evidence"]) is not dict
                or not set(s["stage_evidence"]) <= set(ORDER)):
            raise ValueError("invalid campaign controller state")
        incomplete = False
        for name in ORDER:
            if name in s["stage_evidence"]:
                if incomplete:
                    raise ValueError("campaign stages are out of order")
                evidence = s["stage_evidence"][name]
                self._validate_evidence(name, evidence)
                incomplete = evidence["completed"] < self.targets[name]
            else:
                incomplete = True
        accounted = sum(row["completed"] for row in s["stage_evidence"].values()) + (s["export"] is not None)
        if s["units_completed"] != accounted or s["phase"] != self._phase():
            raise ValueError("campaign stage, export or unit accounting differs")
        if self.manifest["mode"] == "adopted_development" and accounted < sum(self.targets.values()):
            raise ValueError("adopted history may not restart incomplete research stages")

    def _metadata(self):
        return {"schema_version": "p44-d-campaign-checkpoint-v1", "run_identity": canonical_hash(self.manifest),
            "sequence": self.state["sequence"], "phase": self.state["phase"],
            "units_completed": self.state["units_completed"], "new_environment_interactions": 0,
            "new_model_updates": 0, "mode": self.manifest["mode"]}

    def _checkpoint(self):
        self._validate_state()
        self.state["sequence"] += 1
        path = self.directory / "checkpoints" / f"state_{self.state['sequence']:08d}_{uuid.uuid4().hex[:8]}"
        save_cycle_checkpoint(path, state={"controller": deepcopy(self.state)}, metadata=self._metadata())
        self.last_checkpoint = path

    def _verify_recorded_evidence(self):
        for row in self.state["stage_evidence"].values():
            # Old child snapshots remain immutable even when the child advanced.
            checkpoint = _root(row["checkpoint"])
            load_cycle_checkpoint(checkpoint)
            if file_sha256(checkpoint / "manifest.json") != row["checkpoint_manifest_sha256"]:
                raise ValueError("campaign recorded child checkpoint changed")
            manifest = _root(row["path"]) / "run_manifest.json"
            if file_sha256(manifest) != row["manifest_sha256"] or canonical_hash(_read(manifest)) != row["manifest_identity"]:
                raise ValueError("campaign recorded child manifest changed")
            for publication in row["publications"]:
                if file_sha256(_root(publication["path"])) != publication["sha256"]:
                    raise ValueError("campaign recorded child publication changed")

    def _child(self, name):
        path = self.paths[name]
        if self._worker is None or self._worker_path != path:
            if path.exists() and not any((path / "checkpoints").glob("*/manifest.sha256")):
                # Preparation writes no research work. Preserve interrupted
                # output rather than deleting it or pretending a saved update
                # exists. A committed child is never restarted this way.
                journal = path / "attempts.jsonl"
                if (self.manifest["mode"] != "fresh" or not path.is_relative_to(self.directory / "stages")
                        or journal.is_file() and journal.stat().st_size):
                    raise ValueError("incomplete child has possible research work; preserve it for explicit recovery")
                preserved = path.with_name(f"{path.name}_incomplete_prepare_{uuid.uuid4().hex[:8]}")
                if not preserved.is_relative_to(self.directory / "stages"):
                    raise ValueError("incomplete preparation destination escapes campaign")
                path.rename(preserved)
                append_event(self.directory / "attempts.jsonl", {"event": "incomplete_preparation_preserved",
                    "stage": name, "path": str(preserved), "committed_child_checkpoints": 0,
                    "research_training_restarted": False})
            if (path / "run_manifest.sha256").is_file():
                self._worker = stages.resume_stage(name, path, device=self.device)
            else:
                if path.exists():
                    raise ValueError("incomplete child preparation; preserve it and inspect before recovery")
                self._worker = stages.prepare_stage(name, path, inputs=self.manifest["inputs"],
                    device=self.device, precision=self.manifest["precision"],
                    teacher_run=self.paths["teacher"], distillation_run=self.paths["distillation"])
            self._worker_path = path
            if name in ("buffer", "distillation"):
                self._verify_initial()
        return self._worker

    def request_stop(self):
        self._stop_requested = True
        if self._worker is not None and hasattr(self._worker, "request_stop"):
            self._worker.request_stop()

    def _ensure_ready(self, name, evidence):
        self._validate_evidence(name, evidence)
        publication = {"teacher": "frozen_teacher", "distillation": "warm_started_student",
                       "local_cycle": "refined_student"}.get(name)
        if publication and evidence["completed"] == self.targets[name]:
            path = self.paths[name] / f"{publication}.json"
            if not path.with_suffix(".sha256").is_file():
                # Completed child, missing final publication: run() only repairs
                # publication here; the stage's own total cap forbids more PPO.
                child = self._child(name)
                stages.run_one(name, child)
                evidence = stages.inspect_stage(name, self.paths[name])
                self._validate_evidence(name, evidence)
            _read(path)
        return evidence

    def _record(self, name, evidence, *, invocation, attempt, recovered):
        before = self.state["stage_evidence"].get(name, {}).get("completed", 0)
        self._validate_evidence(name, evidence)
        if evidence["completed"] != before + 1:
            raise ValueError("campaign child advanced by more than one registered logical unit")
        self.state["stage_evidence"][name] = deepcopy(evidence)
        self.state["units_completed"] += 1
        self.state["phase"] = self._phase()
        self._checkpoint()
        append_event(self.directory / "attempts.jsonl", {"event": "unit_committed", "invocation": invocation, "attempt": attempt,
            "unit_id": f"{name}:{evidence['completed']}", "stage": name, "ordinal": evidence["completed"],
            "units_completed": self.state["units_completed"], "checkpoint": str(self.last_checkpoint),
            "child_checkpoint": evidence["checkpoint"], "recovered_child_commit": recovered,
            "new_child_work_in_this_attempt": not recovered})

    def _publish_export(self):
        if any(self.state["stage_evidence"].get(name, {}).get("completed") != self.targets[name] for name in ORDER):
            raise ValueError("every D stage must finish before final export")
        local = self.paths["local_cycle"]
        handoff = _read(local / "refined_student.json")
        source = _root(local / handoff["actor_export"])
        if not source.is_relative_to(local) or file_sha256(source / "manifest.json") != handoff["actor_export_manifest_sha256"]:
            raise ValueError("refinement actor export binding changed")
        payload, metadata = load_cycle_checkpoint(source)
        if set(payload) != {"actor"}:
            raise ValueError("campaign deployment export may contain only local actor tensors")
        committed = sorted(path for path in self.directory.glob("actor_export*")
                           if path.is_dir() and (path / "manifest.sha256").is_file())
        for path in committed:
            previous, previous_metadata = load_cycle_checkpoint(path)
            if (previous_metadata != metadata or set(previous) != {"actor"}
                    or set(previous["actor"]) != set(payload["actor"])
                    or any(not torch.equal(value, previous["actor"][key]) for key, value in payload["actor"].items())):
                raise ValueError("existing campaign actor export differs from the final student")
        if committed:
            path = committed[0]
        else:
            path = self.directory / "actor_export"
            if path.exists():
                path = self.directory / f"actor_export_attempt_{uuid.uuid4().hex[:8]}"
            save_cycle_checkpoint(path, state=payload, metadata=metadata)
        return {"path": str(path), "manifest_sha256": file_sha256(path / "manifest.json"),
            "payload_sha256": file_sha256(path / "state.pt"), "source_export": str(source),
            "source_manifest_sha256": file_sha256(source / "manifest.json")}

    def _counts(self):
        rows, p = self.state["stage_evidence"], self.protocol
        count = lambda name: rows.get(name, {}).get("completed", 0)
        per_cycle = p["epochs"] * p["minibatches"]
        imagined = sum(rows[name]["counters"]["imagined_joint_steps"] for name in ("teacher", "local_cycle") if name in rows)
        # P4.2's buffer and optimizer share one store and one imagined-step debit.
        imagined += rows.get("distillation", rows.get("buffer", {})).get("counters", {}).get("imagined_joint_steps", 0)
        return {"teacher_cycles_completed": count("teacher"), "target_teacher_cycles": self.targets["teacher"],
            "trajectories_completed": count("buffer"), "target_trajectories": self.targets["buffer"],
            "distillation_optimizer_steps": count("distillation"), "target_distillation_optimizer_steps": self.targets["distillation"],
            "local_cycles_completed": count("local_cycle"), "target_local_cycles": self.targets["local_cycle"],
            "teacher_actor_optimizer_steps": count("teacher") * per_cycle,
            "teacher_critic_optimizer_steps": count("teacher") * per_cycle,
            "local_actor_optimizer_steps": count("local_cycle") * per_cycle,
            "local_critic_optimizer_steps": count("local_cycle") * per_cycle,
            "target_local_actor_optimizer_steps": self.targets["local_cycle"] * per_cycle,
            "target_local_critic_optimizer_steps": self.targets["local_cycle"] * per_cycle,
            "imagined_joint_steps": imagined, "local_player_decisions": 2 * rows.get("local_cycle", {}).get("counters", {}).get("imagined_joint_steps", 0),
            "model_updates": p["initial_model_updates"], "real_joint_transitions": p["initial_transitions"],
            "new_model_updates": 0, "new_environment_interactions": 0, "teacher_queries_during_refinement": 0,
            "teacher_kl_weight": 0.0}

    def _complete(self):
        self._validate_state()
        if self.state["phase"] != "complete":
            raise ValueError("campaign completion requires all stages and final actor export")
        exported = self.state["export"]
        path = _root(exported["path"])
        load_cycle_checkpoint(path)
        if (file_sha256(path / "manifest.json") != exported["manifest_sha256"]
                or file_sha256(path / "state.pt") != exported["payload_sha256"]):
            raise ValueError("campaign final actor export changed")
        # Buffer and supervised optimization share an append-only journal.
        # Refresh its final digest while retaining the original immutable
        # buffer checkpoint in the controller's stage evidence.
        current = {name: stages.inspect_stage(name, self.paths[name]) for name in ORDER}
        for name, evidence in current.items():
            self._validate_evidence(name, evidence)
            if evidence["completed"] != self.targets[name]:
                raise ValueError("campaign child no longer has its prescribed completed stage")
        ledger = {"schema_version": "p44-d-accounting-v1", "run_identity": canonical_hash(self.manifest),
            **self._counts(), "mode": self.manifest["mode"], "logical_units_completed": self.state["units_completed"],
            "physical_work_by_stage": {name: row["physical_work"] for name, row in current.items()},
            "controller_attempts": str(self.directory / "attempts.jsonl"),
            "interrupted_compute_policy": "unmatched or failed physical attempts remain unknown, never assumed zero",
            "adopted_history_policy": "historical optimization remains attributed; adoption adds no research training",
            "evaluation_interactions": 0, "evaluation_executed": False}
        _publish(self.directory / "accounting.json", ledger)
        completion = {"schema_version": "p44-completed-d-student-v1", "run_identity": canonical_hash(self.manifest),
            "checkpoint": str(self.last_checkpoint), "checkpoint_manifest_sha256": file_sha256(self.last_checkpoint / "manifest.json"),
            "actor_export": exported, "accounting_sha256": file_sha256(self.directory / "accounting.json"),
            "paired_initial_student": self.manifest["initial_student"], **self._counts(),
            "mode": self.manifest["mode"], "fresh_cohort_eligible": self.manifest["fresh_cohort_eligible"],
            "d_controller_implemented": True, "phase4_acceptance": False,
            "matched_reduced_b_controller_implemented": False, "endpoint_evaluator_implemented": False,
            "evaluation_executed": False, "learned_quality_established": False}
        _publish(self.directory / "completed_student.json", completion)

    def _status(self, status, **extra):
        document = {"schema_version": "p44-d-campaign-status-v1", "status": status,
            "phase": self.state["phase"], "units_completed": self.state["units_completed"],
            "target_units": sum(self.targets.values()) + 1, "sequence": self.state["sequence"],
            "variant": "D", "task_family": "relay", "run_seed": self.manifest["inputs"]["master_seed"],
            "mode": self.manifest["mode"], **self._counts(),
            "last_checkpoint": None if self.last_checkpoint is None else str(self.last_checkpoint),
            "phase4_acceptance": False, **extra}
        atomic_json(self.directory / "status.json", document)
        return document

    def status(self):
        return json.loads((self.directory / "status.json").read_text(encoding="utf-8"))

    def run(self, *, max_units=None):
        if max_units is not None and (type(max_units) is not int or max_units < 1):
            raise ValueError("max_units must be a positive integer")
        if self._failed:
            raise RuntimeError("failed campaign must resume in a fresh controller instance")
        with run_lock(self.directory):
            if canonical_hash(_manifest(self.directory)) != canonical_hash(self.manifest):
                raise ValueError("campaign run manifest changed")
            if self.manifest["source_sha256"] != source_identity():
                raise ValueError("campaign source changed; restore the registered checkout")
            if self.manifest["runtime"] != _runtime(self.device):
                raise ValueError("campaign runtime changed")
            if self.last_checkpoint != _latest(self.directory):
                raise RuntimeError("another controller advanced this campaign; resume the latest committed checkpoint")
            self._verify_initial()
            self._validate_state()
            self._verify_recorded_evidence()
            invocation, units, attempt = uuid.uuid4().hex, 0, None
            stop_file = self.directory / "stop.request"
            try:
                while (self.state["phase"] != "complete" and not self._stop_requested and not stop_file.exists()
                       and (max_units is None or units < max_units)):
                    name, started = self.state["phase"], time.perf_counter()
                    self._status("running")
                    attempt = uuid.uuid4().hex
                    append_event(self.directory / "attempts.jsonl", {"event": "unit_started", "attempt": attempt,
                        "invocation": invocation, "stage": name, "next_unit": self.state["units_completed"] + 1,
                        "last_committed_checkpoint": str(self.last_checkpoint)})
                    if name == "export":
                        self.state["export"] = self._publish_export()
                        self.state.update(units_completed=self.state["units_completed"] + 1, phase="complete")
                        self._checkpoint()
                        append_event(self.directory / "attempts.jsonl", {"event": "unit_committed", "attempt": attempt,
                            "invocation": invocation, "stage": "export", "unit_id": "export:1",
                            "checkpoint": str(self.last_checkpoint), "new_child_work_in_this_attempt": False})
                    else:
                        path = self.paths[name]
                        before = self.state["stage_evidence"].get(name, {}).get("completed", 0)
                        if ((path / "run_manifest.sha256").is_file()
                                and any((path / "checkpoints").glob("*/manifest.sha256"))):
                            evidence = stages.inspect_stage(name, path)
                            self._validate_evidence(name, evidence)
                        else:
                            self._child(name)
                            evidence = stages.inspect_stage(name, path)
                        recovered = evidence["completed"] == before + 1
                        if evidence["completed"] < before or evidence["completed"] > before + 1:
                            raise ValueError("child/controller progress diverged beyond one interrupted unit")
                        if not recovered:
                            child = self._child(name)
                            if self._stop_requested or stop_file.exists():
                                append_event(self.directory / "attempts.jsonl", {"event": "unit_cancelled_before_training",
                                    "attempt": attempt, "invocation": invocation, "stage": name,
                                    "research_optimization_started": False})
                                break
                            stages.run_one(name, child)
                            evidence = stages.inspect_stage(name, path)
                            if evidence["completed"] == before and self._stop_requested:
                                break
                        evidence = self._ensure_ready(name, evidence)
                        self._record(name, evidence, invocation=invocation, attempt=attempt, recovered=recovered)
                    append_event(self.directory / "metrics.jsonl", {"stage": name,
                        "checkpoint_sequence": self.state["sequence"], "units_completed": self.state["units_completed"],
                        "elapsed_controller_unit_seconds": time.perf_counter() - started, **self._counts()})
                    units += 1
                    attempt = None
                    self._status("running")
                if self.state["phase"] == "complete":
                    self._complete()
                if stop_file.is_file():
                    stop_file.unlink()
                return self._status("complete" if self.state["phase"] == "complete" else "stopped", units_executed=units)
            except BaseException as error:
                self._failed = True
                try:
                    append_event(self.directory / "attempts.jsonl", {"event": "campaign_invocation_failed",
                        "attempt": attempt, "invocation": invocation, "last_committed_checkpoint": str(self.last_checkpoint),
                        "partial_physical_compute": "unknown; inspect durable child checkpoints and attempt journals",
                        "error_type": type(error).__name__, "error": str(error)})
                    self._status("failed", error_type=type(error).__name__, error=str(error))
                except OSError:
                    pass
                raise

