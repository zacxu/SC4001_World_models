"""Read-only audit of a completed D campaign, with no optimizer or rollout.

Original device labels authenticate historical hashes on CPU. This deliberately
does not resume a GPU worker under a different runtime, regenerate research
targets or load any native environment.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import math
from pathlib import Path

import torch

from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.distillation.buffer import DistillationBuffer
from world_models.distillation.handoff import _adam_counters, _restore_initial_teacher, _validate_completion
from world_models.distillation.runner import _tree_equal
from world_models.distillation.versions import module_version_at_device, origin_device
from world_models.imagination import AnchorSampler
from world_models.infrastructure.reproducibility import canonical_hash, derive_seed
from world_models.mappo.config import MAPPOConfig
from world_models.refinement.handoff import load_warm_started_student, verify_frozen_inputs, _stream_shape
from world_models.refinement.runner import _input_identity, _manifest as refinement_manifest
from world_models.rollouts.streams import RolloutStreams
from world_models.teacher.runner import _latest, _policies, _root
from world_models_campaign import controller as campaign, stages


def _finite_tree(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() and not torch.isfinite(value).all().item():
            raise ValueError("saved optimizer/diagnostic tensor is nonfinite")
    elif isinstance(value, dict):
        for item in value.values():
            _finite_tree(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _finite_tree(item)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError("saved optimizer/diagnostic scalar is nonfinite")


def _bound_checkpoint(path, bound):
    path = _root(path)
    saved, metadata = load_cycle_checkpoint(path)
    publication = campaign._read(path / "manifest.json")
    if (publication.get("metadata") != metadata
            or publication.get("payload", {}).get("sha256") != file_sha256(path / "state.pt")):
        raise ValueError("checkpoint publication changed while being read")
    for name in ("manifest.json", "manifest.sha256"):
        item = path / name
        bound[str(item)] = file_sha256(item)
    bound[str(path / "state.pt")] = publication["payload"]["sha256"]
    return saved, metadata


def _bound_json(path, bound):
    path = _root(path)
    value = campaign._read(path)
    bound[str(path)] = file_sha256(path)
    bound[str(path.with_suffix(".sha256"))] = file_sha256(path.with_suffix(".sha256"))
    return value


def _master_chain(worker, bound):
    """Prove stage ordering and one final export across every parent commit."""
    final_state = deepcopy(worker.state)
    paths = sorted(path for path in (worker.directory / "checkpoints").glob("state_*")
                   if path.is_dir() and (path / "manifest.sha256").is_file())
    starting = worker.manifest.get("starting_units")
    expected_start = 0 if worker.manifest["mode"] == "fresh" else sum(worker.targets.values())
    total = sum(worker.targets.values()) + 1
    if type(starting) is not int or starting != expected_start or len(paths) != total - starting + 1:
        raise ValueError("campaign master history has missing, duplicated or unexpected commits")
    try:
        for sequence, path in enumerate(paths, start=1):
            saved, metadata = _bound_checkpoint(path, bound)
            if set(saved) != {"controller"}:
                raise ValueError("campaign master history contains unsupported state")
            worker.state = saved["controller"]
            worker._validate_state()
            if (int(path.name.split("_")[1]) != sequence or worker.state["sequence"] != sequence
                    or worker.state["units_completed"] != starting + sequence - 1
                    or metadata != worker._metadata()
                    or (worker.state["export"] is not None) != (sequence == len(paths))):
                raise ValueError("campaign master history repeats work, changes stage order or exports early")
        if worker.state != final_state:
            raise ValueError("campaign latest master differs from the verified final history")
    finally:
        worker.state = final_state
    exports = [path for path in worker.directory.glob("actor_export*")
               if path.is_dir() and (path / "manifest.sha256").is_file()]
    if len(exports) != 1 or _root(exports[0]) != _root(final_state["export"]["path"]):
        raise ValueError("campaign requires exactly one committed final actor export")


def _optimizer(state, actor, critic, *, config, precision, seed, versions, cycles, initial=False):
    expected = {"schema_version", "config", "precision", "seed", "network_versions", "cycles_completed",
                "actor_optimizer_steps", "critic_optimizer_steps", "shuffle_rng", "actor_optimizer", "critic_optimizer"}
    steps = cycles * config.epochs * config.minibatches
    if (type(state) is not dict or set(state) != expected or state["config"] != asdict(config)
            or state["precision"] != precision or state["seed"] != seed or tuple(state["network_versions"]) != versions
            or any(type(state[key]) is not int for key in ("cycles_completed", "actor_optimizer_steps", "critic_optimizer_steps"))
            or (state["cycles_completed"], state["actor_optimizer_steps"], state["critic_optimizer_steps"])
               != (cycles, steps, steps)):
        raise ValueError("saved PPO optimizer identity or complete-cycle counters differ")
    _finite_tree(state)
    for key, network, rate in (("actor_optimizer", actor, config.actor_lr),
                               ("critic_optimizer", critic, config.critic_lr)):
        groups = state[key].get("param_groups")
        if (type(groups) is not list or len(groups) != 1 or type(groups[0]) is not dict
                or any(groups[0].get(flag, False) is not False for flag in
                       ("amsgrad", "maximize", "capturable", "differentiable", "decoupled_weight_decay"))
                or any(groups[0].get(flag) is not None for flag in ("foreach", "fused"))):
            raise ValueError("saved PPO Adam execution flags differ from the fixed ordinary optimizer")
        if initial:
            if state[key].get("state") != {}:
                raise ValueError("local PPO did not begin with fresh empty optimizer moments")
            group = state[key].get("param_groups", [])
            if (len(group) != 1 or group[0].get("params") != list(range(len(list(network.parameters()))))
                    or group[0].get("lr") != rate or group[0].get("eps") != config.adam_epsilon
                    or group[0].get("weight_decay") != 0 or group[0].get("betas") != (.9, .999)):
                raise ValueError("initial PPO optimizer settings differ")
        else:
            _adam_counters(state[key], network, steps, rate, config.adam_epsilon)
    rng = state["shuffle_rng"]
    if not isinstance(rng, torch.Tensor) or rng.dtype != torch.uint8 or rng.ndim != 1 or rng.device.type != "cpu":
        raise ValueError("saved PPO shuffle RNG is malformed")
    try:
        torch.Generator(device="cpu").set_state(rng)
    except RuntimeError as error:
        raise ValueError("saved PPO shuffle RNG is malformed") from error


def _cycle_chain(directory, manifest, context, *, actor, critic, teacher, bound):
    """Audit every committed cycle, including exact CPU shuffles/anchor draws."""
    p = manifest["protocol"]
    cycles = p["teacher_cycles" if teacher else "local_cycles"]
    seed = context.master_seed
    label = origin_device(manifest["runtime"]["logical_device"])
    names = (("teacher_initialization", "teacher_critic_initialization", "teacher_anchor_sampling",
              "teacher_anchor_prefix", "teacher_rollouts", "teacher_updates") if teacher else
             ("imagination_anchors", "imagination_prefixes", "local_policy_updates"))
    namespaces = {name: derive_seed("relay", seed, name, "D") for name in names}
    if manifest.get("rng_namespaces") != namespaces:
        raise ValueError("saved PPO random namespaces changed")
    update_seed = namespaces["teacher_updates" if teacher else "local_policy_updates"]
    sampler = AnchorSampler.from_experiment([context.dataset], context.settings,
        seed=namespaces["teacher_anchor_sampling" if teacher else "imagination_anchors"])
    streams = (RolloutStreams.from_seed(namespaces["teacher_rollouts"]) if teacher else
               RolloutStreams.from_experiment(context.settings, master_seed=seed))
    stream_identity = deepcopy(streams.identity)
    stream_identity["device"] = label
    if manifest.get("rollout_stream_identity") != stream_identity:
        raise ValueError("saved PPO rollout namespace changed")
    config = MAPPOConfig.from_experiment(context.settings)
    shuffle = torch.Generator(device="cpu").manual_seed(update_seed)
    paths = sorted(path for path in (directory / "checkpoints").glob("state_*")
                   if path.is_dir() and (path / "manifest.sha256").is_file())
    if len(paths) != cycles + 1:
        raise ValueError("acceptance needs the initial snapshot and each committed PPO cycle exactly once")
    reports, previous_steps, initial_posterior = [], 0, None
    for cycle, path in enumerate(paths):
        saved, metadata = _bound_checkpoint(path, bound)
        state, learner = saved["controller"], saved["learner"]
        if (int(path.name.split("_")[1]) != cycle + 1 or state.get("sequence") != cycle + 1
                or metadata.get("run_identity") != canonical_hash(manifest)
                or learner.get("schema_version") != ("p41-teacher-optimizer-v1" if teacher else "p34-optimizer-v1")):
            raise ValueError("saved PPO checkpoint sequence, manifest or schema differs")
        for network, key in ((actor, "teacher" if teacher else "actor"), (critic, "critic")):
            network.load_state_dict(saved[key], strict=True)
        versions = tuple(module_version_at_device(network, label) for network in (context.model, actor, critic))
        if teacher:
            if (module_version_at_device(context.model, label) != manifest.get("initial_model_version")
                    or not _tree_equal(saved["model"], dict(context.model.state_dict()))):
                raise ValueError("teacher modified the selected frozen world model")
        elif versions[0] != manifest.get("initial_imported_versions", {}).get("model"):
            raise ValueError("local refinement selected model differs")
        _optimizer(learner, actor, critic, config=config, precision=manifest["precision"],
                   seed=update_seed, versions=versions, cycles=cycle, initial=cycle == 0)
        if cycle:
            expected_refs = [asdict(ref) for ref in sampler.sample(p["starts"])]
            report = saved["last_report"]
            if (report.get("anchor_references") != expected_refs or report.get("cycles_completed") != cycle
                    or tuple(report.get(key) for key in ("world_model_version", "actor_version_after", "critic_version_after")) != versions
                    or report.get("cycle_actor_optimizer_steps") != config.epochs * config.minibatches
                    or report.get("cycle_critic_optimizer_steps") != config.epochs * config.minibatches
                    or report.get("new_environment_interactions") != 0):
                raise ValueError("PPO report differs from saved cycles, current policies or fresh anchors")
            if not teacher and any(report.get(key) != 0 for key in
                    ("teacher_kl_weight", "teacher_queries", "teacher_optimizer_steps", "distillation_optimizer_steps")):
                raise ValueError("local refinement used teacher guidance")
            for _ in range(config.epochs):
                torch.randperm(p["starts"], generator=shuffle)
            steps = report["imagined_joint_steps"]
            if (type(steps) is not int or not p["starts"] <= steps <= p["starts"] * p["horizon"]
                    or state["imagined_joint_steps"] != previous_steps + steps):
                raise ValueError("PPO imagined-step cumulative accounting differs")
            previous_steps += steps
            _finite_tree(report)
            reports.append({"cycle": cycle, "checkpoint": str(path), "metrics": deepcopy(report["metrics"]),
                            "termination_counts": deepcopy(report["termination_counts"]), "imagined_joint_steps": steps})
        elif (saved.get("last_report") is not None or state.get("imagined_joint_steps") != 0):
            raise ValueError("PPO initial snapshot already contains trajectory work")
        if not _tree_equal(sampler.state_dict(), saved["sampler"]):
            raise ValueError("PPO anchor RNG does not match exactly reproduced draws")
        if not torch.equal(shuffle.get_state(), learner["shuffle_rng"]):
            raise ValueError("PPO shuffle RNG does not match exactly reproduced permutations")
        _stream_shape(saved["streams"], stream_identity)
        posterior = saved["streams"]["world"]["posterior"]
        if cycle == 0:
            initial_posterior = posterior.clone()
            if label == "cpu" and not _tree_equal(saved["streams"], streams.state_dict()):
                raise ValueError("PPO initial rollout RNG differs from its namespace")
        elif not torch.equal(initial_posterior, posterior):
            raise ValueError("autonomous PPO consumed the posterior rollout stream")
    return reports


@torch.inference_mode(False)
def audit_campaign(run_dir, *, device="cpu"):
    """Authenticate completed artifacts; never resume, optimize, sample or write.

    Integrity failures raise ValueError. Engineering evidence from fixtures and
    an isolated local deployment check are supplied separately by the P4.5 gate.
    """
    if str(torch.device(device)) != "cpu":
        raise ValueError("the saved-campaign audit uses CPU only; no GPU allocation is needed")
    directory, bound = _root(run_dir), {}
    manifest = _bound_json(directory / "run_manifest.json", bound)
    if canonical_hash(manifest) != canonical_hash(campaign._manifest(directory)):
        raise ValueError("campaign registration differs")
    if manifest.get("source_sha256") != campaign.source_identity():
        raise ValueError("campaign source changed; preserve the registered checkout")
    for name, expected in manifest["source_sha256"].items():
        path = campaign.ROOT / name
        if path.is_file():
            bound[str(_root(path))] = expected
    worker = campaign.CampaignController(directory, manifest, device="cpu")
    checkpoint = _latest(directory)
    saved, metadata = _bound_checkpoint(checkpoint, bound)
    if set(saved) != {"controller"}:
        raise ValueError("campaign controller checkpoint is malformed")
    worker.state, worker.last_checkpoint = deepcopy(saved["controller"]), checkpoint
    worker._verify_initial()
    worker._validate_state()
    worker._verify_recorded_evidence()
    if (worker.state["phase"] != "complete" or metadata != worker._metadata()
            or int(checkpoint.name.split("_")[1]) != worker.state["sequence"]):
        raise ValueError("P4.5 requires a durably completed P4.4 campaign; leave the running job unchanged")
    _master_chain(worker, bound)
    completion = _bound_json(directory / "completed_student.json", bound)
    ledger = _bound_json(directory / "accounting.json", bound)
    evidence = {name: stages.inspect_stage(name, worker.paths[name]) for name in campaign.ORDER}
    for name, row in evidence.items():
        worker._validate_evidence(name, row)
        if row["completed"] != worker.targets[name]:
            raise ValueError("campaign child stage is incomplete")
    if worker.paths["buffer"] != worker.paths["distillation"]:
        raise ValueError("the buffer and distillation must share one immutable store")
    context = load_warm_started_student(worker.paths["distillation"], device="cpu")
    bound.update({row["path"]: row["sha256"] for row in context.provenance["bound_files"]})
    # Carry the physical input bytes forward to the gate's final TOCTOU check.
    # The strict upstream loader already verified these paths against the saved
    # selection, replay and normalization declarations; no relocation is made.
    inputs = context.provenance["initial_inputs"]
    input_paths = [path for path in context.dataset.path.rglob("*") if path.is_file()]
    selection = context.binding["selection"]
    input_paths += [Path(selection[key]) for key in ("checkpoint_path", "selection_path")]
    input_paths += [Path(inputs["statistics_root"]) / "relay" / f"seed_{context.master_seed}.json"]
    input_paths += list(Path(inputs["config_root"]).rglob("*.yaml"))
    case = Path(inputs["data_root"]) / "relay" / f"seed_{context.master_seed}" / "bindings"
    input_paths += [case / "B.json", case / "D.json"]
    for path in input_paths:
        # Tiny dependency-injected fixtures have no physical selection or
        # load_case normalization file; production loaders require both.
        if path.is_file():
            path = _root(path)
            bound[str(path)] = file_sha256(path)
            receipt = path.with_suffix(".sha256")
            if receipt.is_file():
                bound[str(receipt)] = file_sha256(receipt)
    for path in (directory / "initial_student",):
        _bound_checkpoint(path, bound)
    teacher_dir, local_dir = worker.paths["teacher"], worker.paths["local_cycle"]
    teacher_manifest = _bound_json(teacher_dir / "run_manifest.json", bound)
    local_manifest = _bound_json(local_dir / "run_manifest.json", bound)
    if canonical_hash(local_manifest) != canonical_hash(refinement_manifest(local_dir)):
        raise ValueError("refinement registration differs")
    if (local_manifest.get("warm_start_input_identity") != _input_identity(context)
            or local_manifest.get("master_seed") != context.master_seed
            or _root(local_manifest["distillation_run"]) != context.directory
            or _root(context.provenance["teacher_run"]) != teacher_dir
            or canonical_hash(manifest["inputs"]) != canonical_hash(teacher_manifest["inputs"])):
        raise ValueError("completed campaign stage lineage differs")
    teacher, critic = _policies(context.settings, context.master_seed, "cpu")
    _restore_initial_teacher(teacher_dir, teacher_manifest, context.model, teacher, critic)
    teacher_reports = _cycle_chain(teacher_dir, teacher_manifest, context, actor=teacher,
                                   critic=critic, teacher=True, bound=bound)
    teacher_saved, teacher_meta = _bound_checkpoint(_latest(teacher_dir), bound)
    teacher_handoff = _bound_json(teacher_dir / "frozen_teacher.json", bound)
    _validate_completion(teacher_saved, teacher_meta, teacher_manifest, teacher_handoff,
                          _latest(teacher_dir), context.model, teacher, critic)
    distilled_label = origin_device(context.manifest["runtime"]["logical_device"])
    if any(module_version_at_device(network, distilled_label) != context.manifest["frozen_versions"][key]
           for key, network in (("model", context.model), ("teacher", teacher), ("teacher_critic", critic))):
        raise ValueError("the distillation frozen teacher differs from its actual completed upstream weights")
    local_initial, _ = _bound_checkpoint(stages._initial(local_dir), bound)
    label = origin_device(local_manifest["runtime"]["logical_device"])
    initial_versions = {name: module_version_at_device(network, label) for name, network in
                        (("model", context.model), ("actor", context.actor), ("critic", context.critic))}
    if (local_manifest.get("initial_imported_versions") != initial_versions
            or local_manifest.get("fresh_local_ppo_optimizers") is not True
            or any(not _tree_equal(local_initial[name], dict(network.state_dict()))
                   for name, network in (("actor", context.actor), ("critic", context.critic)))):
        raise ValueError("local PPO did not start from the exact distilled actor and original paired critic")
    local_reports = _cycle_chain(local_dir, local_manifest, context, actor=context.actor,
                                  critic=context.critic, teacher=False, bound=bound)
    for publication in worker.state["stage_evidence"].values():
        for row in publication["publications"]:
            bound[row["path"]] = row["sha256"]
    for row in evidence.values():
        journal = Path(row["physical_work"]["journal"])
        if journal.is_file():
            expected = row["physical_work"].get("journal_sha256")
            if file_sha256(journal) != expected:
                raise ValueError("stage physical-work journal changed during audit")
            bound[str(_root(journal))] = expected
    if (directory / "attempts.jsonl").is_file():
        bound[str(directory / "attempts.jsonl")] = file_sha256(directory / "attempts.jsonl")
    exported = worker.state["export"]
    export_dir = _root(exported["path"])
    if not export_dir.is_relative_to(directory):
        raise ValueError("campaign deployment export escaped its directory")
    final_payload, export_meta = _bound_checkpoint(export_dir, bound)
    refined = _bound_json(local_dir / "refined_student.json", bound)
    source_export = _root(local_dir / refined["actor_export"])
    source_payload, source_metadata = _bound_checkpoint(source_export, bound)
    if (set(final_payload) != {"actor"} or export_meta != source_metadata
            or not _tree_equal(final_payload, source_payload)
            or not _tree_equal(final_payload["actor"], dict(context.actor.state_dict()))
            or exported != {"path": str(export_dir), "manifest_sha256": file_sha256(export_dir / "manifest.json"),
                "payload_sha256": file_sha256(export_dir / "state.pt"), "source_export": str(source_export),
                "source_manifest_sha256": file_sha256(source_export / "manifest.json")}):
        raise ValueError("campaign final local-only export differs from its completed refinement")
    counts = worker._counts()
    if (completion.get("schema_version") != "p44-completed-d-student-v1"
            or completion.get("run_identity") != canonical_hash(manifest)
            or completion.get("checkpoint") != str(checkpoint)
            or completion.get("checkpoint_manifest_sha256") != file_sha256(checkpoint / "manifest.json")
            or completion.get("actor_export") != exported
            or completion.get("accounting_sha256") != file_sha256(directory / "accounting.json")
            or completion.get("paired_initial_student") != manifest["initial_student"]
            or completion.get("mode") != manifest["mode"]
            or ledger.get("schema_version") != "p44-d-accounting-v1"
            or ledger.get("run_identity") != canonical_hash(manifest)
            or ledger.get("mode") != manifest["mode"]
            or ledger.get("logical_units_completed") != worker.state["units_completed"]
            or any(document.get(key) != value for document in (completion, ledger) for key, value in counts.items())
            or ledger.get("physical_work_by_stage") != {name: row["physical_work"] for name, row in evidence.items()}
            or any(completion.get(key) is not False for key in
                   ("evaluation_executed", "learned_quality_established", "phase4_acceptance"))
            or ledger.get("evaluation_interactions") != 0 or ledger.get("evaluation_executed") is not False):
        raise ValueError("campaign completion, accounting or physical-work attribution changed")
    mask_counts = {"trajectories": 0, "valid_joint_steps": 0, "valid_player_decisions": 0,
                   "padding_joint_steps": 0, "predicted_terminal_trajectories": 0, "horizon_cutoffs": 0}
    distilled, _ = load_cycle_checkpoint(context.checkpoint)
    for shard in distilled["shards"]:
        payload, _ = _bound_checkpoint(context.directory / shard["path"], bound)
        part = DistillationBuffer.from_state_dict(payload["buffer"])
        mask_counts["trajectories"] += part.batch_size
        mask_counts["valid_joint_steps"] += part.imagined_joint_steps
        mask_counts["valid_player_decisions"] += 2 * part.imagined_joint_steps
        mask_counts["padding_joint_steps"] += int((~part.valid).sum())
        mask_counts["predicted_terminal_trajectories"] += int(part.terminated.sum())
        mask_counts["horizon_cutoffs"] += int(part.cutoff.sum())
    verify_frozen_inputs(context)
    if _latest(directory) != checkpoint:
        raise ValueError("campaign changed while being audited")
    for path, expected in bound.items():
        if file_sha256(path) != expected:
            raise ValueError("bound campaign artifact changed during audit")
    return {"schema_version": "p45-campaign-audit-v1", "engineering_passed": True,
        "run_dir": str(directory), "master_seed": context.master_seed, "variant": "D", "task_family": "relay",
        "campaign_manifest_identity": canonical_hash(manifest), "mode": manifest["mode"],
        "actor_export": {"path": str(export_dir), "manifest_sha256": exported["manifest_sha256"],
                         "payload_sha256": exported["payload_sha256"], "source_device": label},
        "lineage": {"paths": {name: str(path) for name, path in worker.paths.items()},
                    "selected_input_binding": deepcopy(context.binding), "paired_initial_student": manifest["initial_student"],
                    "adoption_provenance": deepcopy(manifest.get("adoption_provenance", {}))},
        "stages": evidence, "accounting": ledger,
        "diagnostics": {"teacher_progression": teacher_reports, "local_progression": local_reports,
                        "distillation_before": distilled["diagnostics_before"],
                        "distillation_after": distilled["diagnostics_after"], "buffer_masks": mask_counts,
                        "scope": "imagined training diagnostics only; not real cooperation or superiority"},
        "reproducibility": {"cpu_shuffle_and_anchor_draws_reproduced": True,
            "all_buffer_receipts_prefixes_and_rng_chain_verified": True,
            "historical_gpu_rng_bytes_receipt_bound": True, "gpu_rng_execution_reproduced": False,
            "cross_runtime_target_regeneration_measured": False, "optimizer_constructed_or_stepped": False},
        "cohort_ready": False, "learned_quality_established": False, "native_evaluation_executed": False,
        "new_environment_interactions": 0, "new_model_updates": 0,
        "bound_files": [{"path": path, "sha256": digest} for path, digest in sorted(bound.items())]}
