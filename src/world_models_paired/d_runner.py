"""Prospective D-arm adapter over the preserved P4.4 campaign controller.

The original stage files, optimizers and RNG checkpoint contracts remain the
owners of scientific work.  This additive receipt binds a fresh campaign to a
P5.2 registration and its paired initial actor, critic and frozen model inputs.
Only ``run`` executes research optimization.  No historical adoption is offered.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re
import uuid

from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.data.ledger import file_sha256
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.teacher.runner import _latest, _NUMBERED, _root
from world_models_campaign import controller as campaign
from world_models_campaign.protocol import load_protocol as campaign_protocol
from world_models_study.protocol import ROOT, load_protocol, protocol_identity, training_seed_manifest


SCHEMA = "p52-fresh-d-binding-v1"
BINDING_NAME = "d_binding.json"


def _digest(value, name):
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"D binding requires a SHA256 {name}")
    return value


def _protocol():
    """The legacy scientific stage recipe must implement the endpoint contract."""
    endpoint, parent = load_protocol(), campaign_protocol()
    names = ("task_family", "master_seeds", "variants", "initial_transitions", "initial_model_updates",
             "new_environment_interactions", "new_model_updates", "local_cycles", "starts", "horizon",
             "epochs", "minibatches", "teacher_cycles", "trajectories", "distillation_updates",
             "distillation_sequences_per_update", "teacher_kl_weight", "teacher_refresh_cycles",
             "guidance_decay_enabled", "model_and_normalization_frozen", "shared_initial_replay_per_seed")
    if (any(parent.get(name) != endpoint[name] for name in names)
            or parent.get("local_ppo_optimizer_initialization") != "fresh_after_distillation"):
        raise ValueError("preserved D campaign recipe differs from the endpoint protocol")
    return endpoint


def _initial_binding(value, seed):
    if type(value) is not dict:
        raise ValueError("D requires the complete registered initial input binding")
    selected, provenance = value.get("selection", {}), value.get("provenance", {})
    replay, normalization = provenance.get("replays"), value.get("normalization", {})
    if (selected.get("task") != "relay" or type(selected.get("seed")) is not int
            or selected["seed"] != seed or type(selected.get("model_updates")) is not int
            or selected["model_updates"] != 4500 or selected.get("shared_initial_variants") != ["B", "D"]
            or type(replay) is not list or len(replay) != 1 or replay[0].get("transitions") != 50000
            or selected.get("replays") != replay
            or selected.get("initial_normalization") != provenance.get("initial_normalization")
            or normalization.get("identity_sha256") != provenance.get("initial_normalization", {}).get("identity_sha256")
            or value.get("settings", {}).get("task", {}).get("name") != "relay"
            or value.get("settings", {}).get("variant", {}).get("name") != "D"
            or type(value.get("replay_path")) is not str):
        raise ValueError("D initial input binding is not the registered shared 4500/50000 Relay case")
    for name, digest in (("selected model", selected.get("checkpoint_sha256")),
                         ("normalization", normalization.get("identity_sha256")),
                         ("replay manifest", replay[0].get("manifest_sha256"))):
        _digest(digest, name)
    return value


def _pairing(manifest, input_binding, seed):
    initial = manifest.get("initial_student", {})
    metadata = initial.get("metadata", {})
    provenance = metadata.get("provenance", {})
    seeds = training_seed_manifest(seed, "D")
    versions = metadata.get("paired_versions_cpu", {})
    if (metadata.get("schema_version") != "p44-paired-initial-student-v1"
            or type(metadata.get("master_seed")) is not int or metadata["master_seed"] != seed
            or canonical_hash(metadata.get("input_binding")) != canonical_hash(input_binding)
            or provenance.get("mode") != "fresh" or provenance.get("saved_before_teacher_training") is not True
            or provenance.get("chronology") != "fresh_before_teacher"
            or metadata.get("initialization_seeds") != {name: seeds["paired_initialization"][name] for name in
                ("local_actor_initialization", "local_critic_initialization")}
            or any(type(metadata.get(name)) is not int or metadata[name] != 0 for name in
                ("actor_optimizer_steps", "critic_optimizer_steps", "new_environment_interactions", "new_model_updates"))
            or set(versions) != {"actor", "critic"}):
        raise ValueError("D paired initialization, fresh chronology or registered input binding differs")
    return {"actor_cpu": _digest(versions["actor"], "initial actor"),
            "critic_cpu": _digest(versions["critic"], "initial critic"),
            "input_binding_sha256": canonical_hash(input_binding),
            "world_model_checkpoint_sha256": input_binding["selection"]["checkpoint_sha256"],
            "normalization_identity_sha256": input_binding["normalization"]["identity_sha256"],
            "replay_manifest_sha256": input_binding["provenance"]["replays"][0]["manifest_sha256"]}


def _validate_manifest(directory, manifest, *, input_binding, seed):
    _protocol()
    if (type(seed) is not int or seed not in (0, 1) or manifest.get("schema_version") != campaign.RUN_SCHEMA
            or manifest.get("variant") != "D" or manifest.get("task_family") != "relay"
            or manifest.get("mode") != "fresh" or manifest.get("fresh_stage_initialization") is not True
            or manifest.get("adoption_provenance") != {} or type(manifest.get("starting_units")) is not int
            or manifest["starting_units"] != 0 or manifest.get("research_training_executed_at_prepare") is not False
            or manifest.get("precision") != "bfloat16" or manifest.get("protocol") != campaign_protocol()
            or manifest.get("protocol_sha256") != canonical_hash(campaign_protocol())
            or manifest.get("source_sha256") != campaign.source_identity()
            or type(manifest.get("inputs", {}).get("master_seed")) is not int
            or manifest["inputs"]["master_seed"] != seed):
        raise ValueError("P5.2 D requires a fresh, unmodified, bfloat16 fixed-data campaign")
    expected_paths = {"teacher": str(directory / "stages/teacher"),
        "buffer": str(directory / "stages/distillation"), "distillation": str(directory / "stages/distillation"),
        "local_cycle": str(directory / "stages/refinement")}
    if manifest.get("paths") != expected_paths:
        raise ValueError("P5.2 D stages must remain inside their fresh campaign")
    _initial_binding(input_binding, seed)
    if (manifest["inputs"] != manifest.get("initial_student", {}).get("metadata", {}).get("inputs")
            or _root(input_binding["selection"]["selection_path"]) != _root(manifest["inputs"]["initial_selection"])):
        raise ValueError("D campaign input paths or selected initial inputs differ")
    return _pairing(manifest, input_binding, seed)


def _read_binding(directory, *, cohort_identity=None, input_binding=None):
    binding = campaign._read(directory / BINDING_NAME)
    if (type(binding) is not dict or binding.get("schema_version") != SCHEMA
            or binding.get("endpoint_protocol_sha256") != protocol_identity()
            or binding.get("variant") != "D" or binding.get("task_family") != "relay"
            or binding.get("historical_adoption_permitted") is not False
            or binding.get("research_training_executed_at_prepare") is not False
            or binding.get("fresh_cohort_eligible") is not False):
        raise ValueError("invalid prospective D binding; historical or adopted runs cannot be promoted")
    _digest(binding.get("cohort_identity"), "cohort identity")
    if cohort_identity is not None and binding["cohort_identity"] != _digest(cohort_identity, "cohort identity"):
        raise ValueError("D arm belongs to another P5.2 cohort")
    if input_binding is not None and canonical_hash(binding.get("input_binding")) != canonical_hash(input_binding):
        raise ValueError("D arm belongs to another registered initial input binding")
    seed = binding.get("master_seed")
    if (type(seed) is not int or seed not in (0, 1)
            or binding.get("training_seeds") != training_seed_manifest(seed, "D")):
        raise ValueError("D arm training RNG namespace differs from the endpoint registration")
    manifest = campaign._manifest(directory)
    pairing = _validate_manifest(directory, manifest, input_binding=binding.get("input_binding"), seed=seed)
    if (binding.get("campaign_manifest_sha256") != file_sha256(directory / "run_manifest.json")
            or binding.get("campaign_identity") != canonical_hash(manifest)
            or binding.get("paired_initialization") != pairing):
        raise ValueError("D campaign manifest or paired initialization changed after registration")
    return binding, manifest


def _new_binding(directory, manifest, *, cohort_identity, input_binding):
    seed = manifest["inputs"]["master_seed"]
    _digest(cohort_identity, "cohort identity")
    pairing = _validate_manifest(directory, manifest, input_binding=input_binding, seed=seed)
    return {"schema_version": SCHEMA, "endpoint_protocol_sha256": protocol_identity(),
        "cohort_identity": cohort_identity, "master_seed": seed, "variant": "D", "task_family": "relay",
        "campaign_manifest_sha256": file_sha256(directory / "run_manifest.json"),
        "campaign_identity": canonical_hash(manifest), "paired_initialization": pairing,
        "input_binding": deepcopy(input_binding), "training_seeds": training_seed_manifest(seed, "D"),
        "historical_adoption_permitted": False, "research_training_executed_at_prepare": False,
        "fresh_cohort_eligible": False}


def _assert_untrained_preparation(worker):
    state = worker.state
    if (state != {"sequence": 1, "units_completed": 0, "phase": "teacher", "stage_evidence": {}, "export": None}
            or any(path.exists() for path in set(worker.paths.values()))
            or any((worker.directory / name).is_file() and (worker.directory / name).stat().st_size
                   for name in ("attempts.jsonl", "metrics.jsonl"))
            or any(worker.directory.glob("actor_export*"))
            or any((worker.directory / name).exists() for name in ("accounting.json", "completed_student.json"))):
        raise ValueError("D preparation recovery permits only an untouched sequence-one, zero-work campaign")


class PairedDRunner:
    """Expose one fresh D arm without relabelling its historical stage receipts."""

    def __init__(self, worker, binding):
        self.worker, self.manifest = worker, binding
        self.directory = worker.directory

    @classmethod
    def prepare(cls, directory, *, seed, initial_selection, cohort_identity, input_binding,
                config_root=ROOT / "configs", replay_root=ROOT / "data/initial_replay/pilot",
                normalization_dir=ROOT / "artifacts/phase23/normalization", device="cuda:0", precision="bfloat16"):
        _protocol()
        if type(seed) is not int or seed not in (0, 1):
            raise ValueError("P5.2 D permits master seeds 0 and 1 only")
        _digest(cohort_identity, "cohort identity")
        _initial_binding(input_binding, seed)
        if precision != "bfloat16":
            raise ValueError("P5.2 keeps the registered bfloat16 frozen world model")
        directory = _root(directory)
        worker = campaign.CampaignController.prepare(directory, seed=seed, initial_selection=initial_selection,
            config_root=config_root, replay_root=replay_root, normalization_dir=normalization_dir,
            device=device, precision=precision)
        binding = _new_binding(directory, worker.manifest, cohort_identity=cohort_identity, input_binding=input_binding)
        campaign._publish(directory / BINDING_NAME, binding)
        return cls(worker, binding)

    @classmethod
    def recover_prepared(cls, directory, *, device="cuda:0", cohort_identity, input_binding):
        """Finish only an interrupted, untrained registration publication.

        No campaign state is regenerated.  Any child store, attempted unit or
        advanced checkpoint requires ordinary recovery through its committed
        binding, never this narrowly scoped initialization repair.
        """
        directory = _root(directory)
        _digest(cohort_identity, "cohort identity")
        path = _root(directory / BINDING_NAME)
        receipt = _root(path.with_suffix(".sha256"))
        if receipt.exists():
            restored = cls.resume(directory, device=device, cohort_identity=cohort_identity,
                                  input_binding=input_binding)
            _assert_untrained_preparation(restored.worker)
            return restored
        worker = campaign.CampaignController.resume(directory, device=device)
        _assert_untrained_preparation(worker)
        binding = _new_binding(directory, worker.manifest, cohort_identity=cohort_identity, input_binding=input_binding)
        # A receipt never existed for these bytes. Preserve them as evidence,
        # then finish the original campaign's external binding publication.
        if path.exists():
            if not path.is_file():
                raise ValueError("uncommitted D binding is not an ordinary file")
            preserved = _root(directory / "uncommitted")
            destination = _root(preserved / f"d_binding_{uuid.uuid4().hex}.json")
            if not destination.is_relative_to(directory):
                raise ValueError("D uncommitted binding preservation escapes its campaign")
            preserved.mkdir(exist_ok=True)
            path.rename(destination)
        campaign._publish(path, binding)
        return cls(worker, binding)

    @classmethod
    def resume(cls, directory, *, device="cuda:0", cohort_identity=None, input_binding=None):
        directory = _root(directory)
        binding, _ = _read_binding(directory, cohort_identity=cohort_identity, input_binding=input_binding)
        worker = campaign.CampaignController.resume(directory, device=device)
        return cls(worker, binding)

    def pairing(self):
        return deepcopy(self.manifest["paired_initialization"])

    def status(self):
        result = self.worker.status()
        return {**result, "p52_binding_sha256": file_sha256(self.directory / BINDING_NAME),
                "cohort_identity": self.manifest["cohort_identity"], "fresh_cohort_eligible": False}

    def run(self, *, max_units=None):
        binding, manifest = _read_binding(self.directory, cohort_identity=self.manifest["cohort_identity"],
                                         input_binding=self.manifest["input_binding"])
        if (canonical_hash(binding) != canonical_hash(self.manifest)
                or canonical_hash(manifest) != canonical_hash(self.worker.manifest)):
            raise ValueError("D adapter or campaign changed after construction")
        self.worker.run(max_units=max_units)
        return self.status()

    def request_stop(self):
        self.worker.request_stop()


def inspect(directory, *, cohort_identity=None, input_binding=None):
    """Read durable parent/child evidence on CPU without rebuilding a GPU worker.

    The parent checkpoint may trail an independently committed child by one
    unit. The original controller reconciles that durable child on resume; this
    inspection reports the parent boundary honestly without replaying work.
    """
    directory = _root(directory)
    binding, manifest = _read_binding(directory, cohort_identity=cohort_identity, input_binding=input_binding)
    checkpoint = _latest(directory)
    saved, metadata = load_cycle_checkpoint(checkpoint)
    if set(saved) != {"controller"}:
        raise ValueError("D checkpoint contains unsupported parent state")
    worker = campaign.CampaignController(directory, manifest, device=manifest["runtime"]["logical_device"])
    worker.state, worker.last_checkpoint = saved["controller"], checkpoint
    worker._verify_initial()
    worker._validate_state()
    if (metadata != worker._metadata() or int(_NUMBERED.fullmatch(checkpoint.name)[1]) != worker.state["sequence"]):
        raise ValueError("D parent checkpoint metadata or sequence differs")
    worker._verify_recorded_evidence()
    complete = worker.state["phase"] == "complete"
    published = _verify_completion(worker) if complete else False
    status = "complete" if published else "publication_pending" if complete else "incomplete"
    return {"schema_version": "p52-d-durable-status-v1", "variant": "D", "task_family": "relay",
        "run_seed": binding["master_seed"], "phase": worker.state["phase"],
        "status": status, "completion_publications_verified": published,
        "units_completed": worker.state["units_completed"], "target_units": sum(worker.targets.values()) + 1,
        "last_checkpoint": str(checkpoint), "paired_initialization": deepcopy(binding["paired_initialization"]),
        "cohort_identity": binding["cohort_identity"], "fresh_cohort_eligible": False,
        "p52_binding_sha256": file_sha256(directory / BINDING_NAME), **worker._counts()}


def _verify_completion(worker):
    """A final parent commit alone does not establish a published final actor."""
    directory, manifest = worker.directory, worker.manifest
    exported = worker.state["export"]
    if type(exported) is not dict or type(exported.get("path")) is not str:
        raise ValueError("D completion has no final actor export binding")
    export_path = _root(exported["path"])
    if (not export_path.is_relative_to(directory)
            or file_sha256(export_path / "manifest.json") != exported.get("manifest_sha256")
            or file_sha256(export_path / "state.pt") != exported.get("payload_sha256")):
        raise ValueError("D final actor export bytes differ from the committed parent")
    state, _ = load_cycle_checkpoint(export_path)
    if set(state) != {"actor"}:
        raise ValueError("D final deployment export must contain only local actor tensors")
    publications = [directory / name for name in ("accounting.json", "completed_student.json")]
    # An interrupted publication is repairable by the original controller's
    # run(), which has already capped every scientific child at its target.
    if not all(path.with_suffix(".sha256").is_file() for path in publications):
        return False
    accounting, completion = [campaign._read(path) for path in publications]
    counts = worker._counts()
    if (accounting.get("schema_version") != "p44-d-accounting-v1"
            or completion.get("schema_version") != "p44-completed-d-student-v1"
            or any(row.get("run_identity") != canonical_hash(manifest) or row.get("mode") != "fresh"
                   or any(row.get(name) != value for name, value in counts.items())
                   for row in (accounting, completion))
            or accounting.get("logical_units_completed") != worker.state["units_completed"]
            or completion.get("actor_export") != exported
            or completion.get("paired_initial_student") != manifest["initial_student"]
            or completion.get("accounting_sha256") != file_sha256(publications[0])
            or completion.get("checkpoint") != str(worker.last_checkpoint)
            or completion.get("checkpoint_manifest_sha256") != file_sha256(worker.last_checkpoint / "manifest.json")
            or completion.get("fresh_cohort_eligible") is not False):
        raise ValueError("D completed student receipt or accounting differs from its final checkpoint")
    return True


# Keep the backend interface identical to the independently implemented B arm.
PairedDRunner.inspect = staticmethod(inspect)
