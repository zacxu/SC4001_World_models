"""Fail-closed Phase 3 evidence aggregation; this module never trains a policy."""
from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path

from scripts.phase36_support.evidence import (
    INVARIANT_TESTS, ROOT, file_sha256, load_report, source_identity, write_report,
)
from world_models.cycle.checkpoints import load_cycle_checkpoint
from world_models.cycle.schedule import Schedule
from world_models.cycle.provenance import source_identity as training_source_identity
from world_models.infrastructure.reproducibility import canonical_hash
from world_models.infrastructure.config import resolve_config
from world_models.policies.config import PolicyConfig
from dataclasses import asdict


# This is a deliberately small diagnostic bank, not a significance guarantee or
# a final research performance threshold. Negative outcomes remain in the report.
MIN_PAIRED_EPISODES = 8


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, name):
    _require(type(value) in (int, float) and math.isfinite(value), f"{name} must be finite")
    return float(value)


def _integer(value, name, minimum=0):
    _require(type(value) is int and value >= minimum, f"{name} must be an integer >= {minimum}")
    return value


def _artifact(record, base, name):
    _require(type(record) is dict and type(record.get("path")) is str, f"missing {name} artifact")
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    _require(not any(p.is_symlink() for p in (path, *path.parents)), f"{name} cannot use symbolic links")
    _require(file_sha256(path) == record.get("sha256"), f"{name} artifact checksum differs")
    return path.resolve()


def _invariants(report, base):
    _require(report.get("schema_version") == "p36-implementation-v1", "unsupported implementation report")
    _require(report.get("status") == "complete_p36_implementation", "implementation checks did not pass")
    for key in ("failures", "errors", "skipped"):
        _require(type(report.get(key)) is int and report[key] == 0, f"implementation {key} must be zero")
    _require(report.get("source_unchanged_during_verification") is True, "verification source changed during checks")
    passed = report.get("passed_tests")
    _require(type(passed) is list and all(type(item) is str for item in passed), "missing passed test IDs")
    _require(len(passed) == len(set(passed)) == report.get("tests_run"), "passed test count differs")
    groups = report.get("invariants")
    _require(type(groups) is dict and set(groups) == set(INVARIANT_TESTS), "invariant coverage is incomplete")
    sources = report.get("test_source_sha256")
    _require(type(sources) is dict, "missing invariant test source hashes")
    current_tests = {path.relative_to(ROOT).as_posix(): file_sha256(path) for path in sorted((ROOT / "tests").glob("*.py"))}
    _require(sources == current_tests, "verification test or helper source differs from this checkout")
    for category, required in INVARIANT_TESTS.items():
        group = groups[category]
        _require(type(group) is dict and group.get("status") == "passed"
                 and set(group.get("test_ids", [])) == set(required), f"{category}: coverage differs")
        _require(set(required).issubset(passed), f"{category}: required tests did not pass")
        for test_id in required:
            relative = "/".join(test_id.split(".")[:-2]) + ".py"
            _require(sources.get(relative) == file_sha256(ROOT / relative), f"stale invariant test source: {relative}")
    _artifact(report.get("tests_log"), base, "tests log")
    return {name: {"status": "passed", "test_ids": list(ids)} for name, ids in INVARIANT_TESTS.items()}


def _mechanics(report, base):
    _require(report.get("schema_version") == "p36-mechanics-v1", "unsupported mechanics report")
    _require(report.get("engineering_only") is True
             and report.get("native_environment_evaluation") is False
             and report.get("phase3_acceptance") is False,
             "mechanics must remain explicitly separate from native acceptance")
    checks = report.get("checks")
    required = {"world_unchanged", "actor_changed", "critic_changed", "improvement_over_before",
                "above_chance", "finite_ppo_diagnostics", "initial_ratios_replayed", "independent_actor_streams"}
    _require(type(checks) is dict and required.issubset(checks), "mechanics checks are incomplete")
    _require(report.get("source_sha256_after") == report.get("source_sha256"), "mechanics source changed")
    _require(report.get("learning_algorithm") == "production RecurrentMAPPO", "mechanics did not use the production PPO updater")
    cycles = _integer(report.get("cycles"), "mechanics cycles", 1)
    updates = _artifact({"path": report.get("ppo_updates_path"), "sha256": report.get("ppo_updates_sha256")},
                        base, "mechanics PPO diagnostics")
    rows = [json.loads(line) for line in updates.read_text(encoding="utf-8").splitlines()]
    _require(len(rows) == cycles, "mechanics PPO diagnostic count differs")
    for name in ("actor_optimizer_steps", "critic_optimizer_steps"):
        total = _integer(report.get(name), f"mechanics {name}", 1)
        _require(total == rows[-1].get(name), f"mechanics {name} differs from optimizer diagnostics")
    for index, row in enumerate(rows, 1):
        _require(row.get("cycles_completed") == index, "mechanics optimizer cycles are missing")
        metrics = row.get("metrics", {})
        _require(bool(metrics), "mechanics optimizer metrics are missing")
        for name, value in metrics.items():
            if value is not None:
                _number(value, f"mechanics optimizer metric {name}")
        _require(_number(metrics.get("initial_ratio_max_error"), "mechanics initial ratio error") < 2e-5,
                 "mechanics initial PPO ratio failed")
    exports = report.get("actor_exports", {})
    _require(set(exports) == {"before", "after"}, "mechanics actor exports are missing")
    export_versions = []
    for label, record in exports.items():
        path = base / record["path"]
        state, metadata = load_cycle_checkpoint(path.resolve(), device="cpu")
        _require(set(state) == {"actor"} and metadata.get("schema_version") == "p36-mechanics-actor-v1"
                 and metadata.get("engineering_only") is True and metadata.get("phase3_acceptance") is False,
                 "mechanics export has incorrect scope")
        _require(file_sha256(path / "manifest.json") == record.get("manifest_sha256")
                 and file_sha256(path / "state.pt") == record.get("payload_sha256"), "mechanics export checksum differs")
        _require(metadata.get("actor_version") == record.get("actor_version") and metadata.get("label") == label
                 and metadata.get("world_model_version") == report.get("world_model_version"), "mechanics export identity differs")
        export_versions.append(metadata["actor_version"])
    _require(len(set(export_versions)) == 2, "mechanics actor exports did not change")
    before, after, chance = report.get("before", {}), report.get("after", {}), report.get("chance", {})
    initial = _number(before.get("expected_joint_success"), "mechanics baseline success")
    final = _number(after.get("expected_joint_success"), "mechanics trained success")
    random = _number(chance.get("expected_joint_success"), "mechanics chance success")
    initial_return = _number(before.get("mean_expected_team_return"), "mechanics baseline return")
    final_return = _number(after.get("mean_expected_team_return"), "mechanics trained return")
    _require(0 <= initial <= 1 and 0 <= final <= 1 and 0 < random < 1, "invalid mechanics probabilities")
    ready = (report.get("status") == "passed" and all(checks[key] is True for key in required)
             and final > initial and final > random and final_return > initial_return)
    return {"status": "passed" if ready else "needs_review", "engineering_only": True,
            "baseline_expected_joint_success": initial, "trained_expected_joint_success": final,
            "chance_expected_joint_success": random, "expected_return_delta": final_return-initial_return,
            "checks": checks}


def _export(record, base):
    _require(type(record) is dict and type(record.get("path")) is str, "missing actor export")
    path = Path(record["path"])
    path = path if path.is_absolute() else base / path
    from scripts.phase36_support.evaluation import load_actor_export
    actor, verified = load_actor_export(path.resolve(), device="cpu", source_device=record.get("verified_origin_device"))
    del actor
    metadata = verified["metadata"]
    _require(metadata == record.get("metadata"), "actor export metadata differs")
    _require(file_sha256(path / "manifest.json") == record.get("manifest_sha256"), "actor manifest differs")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    _require(manifest["payload"]["sha256"] == record.get("payload_sha256"), "actor payload differs")
    _require(verified["portable_weights_sha256"] == record.get("portable_weights_sha256"), "portable actor weight hash differs")
    run_path = _artifact(record.get("run_manifest"), base, "P3.5 run manifest")
    run = json.loads(run_path.read_text(encoding="utf-8"))
    _require(run.get("schema_version") == "p35-run-v1", "unsupported actor training run")
    _require(run.get("source_sha256") == training_source_identity(), "actor training source differs from this checkout")
    settings = run.get("settings", {})
    schedule = Schedule.from_manifest(run)
    config = json.loads(json.dumps(asdict(PolicyConfig.from_experiment(settings))))
    _require(config == metadata.get("actor_config") and run.get("schedule") == schedule.describe(),
             "actor architecture or schedule differs from its training run")
    _require(metadata.get("actual_initial_replay_interactions") == schedule.initial_transitions,
             "actor initial interaction accounting differs from the training schedule")
    if metadata.get("untrained_zero_interaction_baseline"):
        expected_cycles, expected_model = 0, schedule.initial_model_updates
    else:
        real = _integer(metadata.get("real_joint_transitions"), "actor real interactions", 1)
        targets = [schedule.block_target(index) for index in range(schedule.blocks+1)]
        _require(real in targets, "actor export is not a completed collection block")
        block = targets.index(real)
        expected_cycles = schedule.completed_local_cycles(block)
        expected_model = schedule.initial_model_updates + block*schedule.online_model_updates
    steps = expected_cycles*settings["mappo"]["epochs"]*settings["mappo"]["recurrent_minibatches"]
    _require(metadata.get("local_ppo_cycles") == expected_cycles and metadata.get("model_updates") == expected_model
             and metadata.get("actor_optimizer_steps") == metadata.get("critic_optimizer_steps") == steps,
             "actor learning counters differ from the completed training schedule")
    _require(record.get("run_identity") == canonical_hash(run), "actor run identity differs")
    _require(path.resolve().parent == run_path.parent / "evaluation", "actor export is outside its training run")
    _require(run.get("master_seed") == metadata.get("run_seed"), "actor training seed differs")
    _require(metadata.get("variant") == "B", "Phase 3 accepts variant B only")
    return metadata


def _ledger(path, report, episodes):
    """Reconcile the durable physical debit stream, not just its summary JSON."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    _require(bool(rows), "evaluation ledger is empty")
    _require(rows[0].get("event") == "created", "evaluation ledger is missing its creation record")
    lengths = {row["episode_id"]: row["length"] for row in episodes}
    _require(len(lengths) == len(episodes), "evaluation repeats an episode ID")
    pending, completed, seen_sources = {}, set(), {}
    physical = initialization = published = tics = 0
    for index, row in enumerate(rows):
        _require(row.get("schema_version") == "real-replay-ledger-v1" and row.get("sequence") == index
                 and row.get("task_family") == report["task_family"] and row.get("run_seed") == report["run_seed"],
                 "evaluation ledger schema, sequence or identity differs")
        delta = tuple(_integer(row.get(name, 0), f"ledger {name}") for name in
                      ("physical_delta", "initialization_delta", "published_delta", "failed_delta"))
        cost, reset, decisions, failed = delta
        _require(cost == reset+decisions+failed and failed == 0, "evaluation ledger debit is failed or inconsistent")
        event, episode = row.get("event"), row.get("episode_id")
        if event == "created":
            _require(index == 0 and row.get("learning_eligible") is False
                     and row.get("data_source") == "evaluation_environment" and not any(delta),
                     "evaluation ledger creation is invalid")
        elif event == "reset":
            _require(episode in lengths and episode not in pending and episode not in completed
                     and cost == reset and not decisions, "invalid evaluation reset debit")
            pending[episode] = 0
        elif event == "transition":
            _require(episode in pending and delta == (1, 0, 1, 0)
                     and row.get("decision_index") == pending[episode]
                     and row.get("collector_kind") == "local_actor_evaluation", "invalid evaluation transition debit")
            tics += _integer(row.get("actual_tics"), "evaluation simulator tics", 1)
            pending[episode] += 1
        elif event == "evaluation_episode_completed":
            _require(episode in pending and row.get("valid_length") == pending[episode] == lengths[episode]
                     and not any(delta), "evaluation episode completion differs from ledger")
            pending.pop(episode)
            completed.add(episode)
        else:
            raise ValueError("evaluation ledger contains an error or unsupported event")
        if "coordinator_run_dir" in row:
            source = row["coordinator_run_dir"]
            now = (_integer(row.get("physical_consumed_at_source"), "source physical counter"),
                   _integer(row.get("initialization_consumed_at_source"), "source reset counter"))
            old = seen_sources.get(source, (0, 0))
            _require((now[0]-old[0], now[1]-old[1]) == (cost, reset), "evaluation source physics counters differ")
            seen_sources[source] = now
        physical += cost
        initialization += reset
        published += decisions
    accounting = report["interaction_accounting"]
    _require(not pending and completed == set(lengths), "evaluation ledger contains incomplete episodes")
    _require(published == sum(lengths.values()) and published == accounting["published_real_joint_transitions"]
             and physical == accounting["physical_joint_decisions_including_initialization"]
             and initialization == accounting["initialization_joint_decisions"]
             and tics == accounting.get("published_real_simulator_tics_total"), "durable evaluation debit totals differ")


def _evaluation(report, base):
    _require(report.get("schema_version") == "p36-real-evaluation-v1" and report.get("status") == "complete",
             "native evaluation is absent or incomplete")
    _require(report.get("native_backend") is True and report.get("native_engine") is True,
             "mock engines cannot provide real evaluation acceptance")
    _require(report.get("source_unchanged") is True, "evaluation source changed during execution")
    _require(report.get("actor_version_before") == report.get("actor_version_after")
             and type(report.get("actor_version_before")) is str, "evaluation changed actor weights")
    _require(all(report.get(name) is False for name in ("world_model_loaded", "critic_loaded", "learning_eligible")),
             "evaluation must deploy only local actors and remain outside learning replay")
    for key in ("training_interactions_added", "real_ppo_updates", "policy_optimizer_updates"):
        _require(type(report.get(key)) is int and report[key] == 0, f"evaluation changed training: {key}")
    metadata = _export(report.get("actor_export"), base)
    _require(all(report.get(key) == metadata.get(key) for key in ("task_family", "run_seed", "variant")),
             "evaluation identity differs from actor export")
    protocol = report.get("protocol", {})
    _require(protocol.get("split") == "validation", "acceptance cannot tune on the untouched test bank")
    _require(protocol.get("precision") in ("float32", "bfloat16"), "evaluation precision is missing")
    _require(protocol.get("mode") == "stochastic" and protocol.get("temperature") == 1.0,
             "acceptance requires stochastic temperature-one evaluation; greedy is diagnostic only")
    settings = resolve_config(report["task_family"], "B", "pilot", ROOT / "configs").settings
    _require(protocol.get("deadline") == settings["task"]["deadline"]
             and protocol.get("frame_skip") == settings["environment"]["frame_skip"]
             and protocol.get("evaluation_data_enters_training") is False,
             "evaluation task clock, frame skip or data isolation differs from the current protocol")
    runtime = report.get("runtime")
    _require(type(runtime) is dict and all(name in runtime for name in
             ("python", "torch", "vizdoom", "device", "precision", "deterministic_algorithms", "cudnn_benchmark", "cpu_threads")),
             "evaluation runtime identity is missing")
    _require(type(runtime.get("vizdoom")) is str and bool(runtime["vizdoom"]), "native ViZDoom version is missing")
    _require(runtime["precision"] == protocol["precision"], "runtime precision differs from the evaluation protocol")
    runtime = {**runtime, "device": str(runtime["device"]).split(":")[0]}
    episodes = report.get("episodes")
    _require(type(episodes) is list and episodes, "native evaluation contains no episodes")
    _require(protocol.get("episodes") == len(episodes), "evaluation bank size differs")
    from scripts.phase36_support.evaluation import validation_bank
    bank = validation_bank(report.get("task_family"), episodes=len(episodes), bank_seed=protocol.get("bank_seed"))
    _require(canonical_hash(bank) == protocol.get("bank_sha256"), "validation bank checksum differs")
    files = report.get("files", {})
    needed = {"bank.json", "episodes.jsonl", "transitions.jsonl", "evaluation_interactions.jsonl"}
    _require(needed.issubset(files), "evaluation trace artifacts are missing")
    paths = {name: _artifact(files[name], base, name) for name in needed}
    _require(json.loads(paths["bank.json"].read_text(encoding="utf-8")) == bank, "saved validation bank differs")
    _require([json.loads(line) for line in paths["episodes.jsonl"].read_text(encoding="utf-8").splitlines()] == episodes,
             "evaluation episode journal differs from report")
    event_keys = {"relay": ("latch_activations",), "cover": ("console_1_completions", "console_2_completions")}
    _require(report.get("task_family") in event_keys, "unknown evaluated task family")
    keys, successes, returns, progress, lengths = [], [], [], [], []
    for index, episode in enumerate(episodes):
        _require(type(episode) is dict and episode.get("index") == index, "episode index is missing or duplicated")
        _require(type(episode.get("success")) is bool, "success outcomes must be Boolean")
        _require(all(episode.get(name) == value for name, value in bank["episodes"][index].items()),
                 "episode does not match the deterministic validation bank")
        _require(episode["success"] == (episode.get("terminal_reason") == "success"), "episode terminal success differs")
        _require(episode.get("terminal_reason") in ("success", "death", "deadline", "failure"), "evaluation episode is incomplete")
        key = tuple(episode.get(name) for name in ("template_id", "layout_seed", "engine_seed"))
        _require(type(key[0]) is str and type(key[1]) is int and type(key[2]) is int, "episode bank identity is missing")
        keys.append(key)
        successes.append(int(episode["success"]))
        returns.append(_number(episode.get("return"), "episode return"))
        lengths.append(_integer(episode.get("length"), "episode length", 1))
        _require(lengths[-1] <= protocol["deadline"] and (episode["terminal_reason"] != "deadline"
                 or lengths[-1] == protocol["deadline"]), "episode clock differs from exact deadline")
        events = episode.get("event_counts")
        _require(type(events) is dict, "episode event counts are missing")
        progress.append(sum(_integer(events.get(key, 0), key) for key in event_keys[report["task_family"]]))
    _require(len(keys) == len(set(keys)), "native bank repeats an episode")
    accounting = report.get("interaction_accounting", {})
    total = sum(lengths)
    _require(accounting.get("learning_eligible") is False and accounting.get("training_interactions_added") == 0
             and accounting.get("data_source") == "evaluation_environment", "evaluation accounting entered learning data")
    _require(accounting.get("ledger_sha256") == files["evaluation_interactions.jsonl"]["sha256"], "evaluation ledger checksum differs")
    _require(all(accounting.get(name) == total for name in
                 ("real_joint_transitions_total", "published_real_joint_transitions", "completed_episode_joint_transitions"))
             and accounting.get("failed_dispatch_attempts") == 0, "evaluation interaction counters differ")
    _require(accounting.get("physical_joint_decisions_including_initialization")
             == total + accounting.get("initialization_joint_decisions", -1), "evaluation reset overhead does not reconcile")
    transitions = [json.loads(line) for line in paths["transitions.jsonl"].read_text(encoding="utf-8").splitlines()]
    _require(len(transitions) == total, "evaluation transition journal length differs")
    _ledger(paths["evaluation_interactions.jsonl"], report, episodes)
    offset = 0
    for episode in episodes:
        segment = transitions[offset:offset+episode["length"]]
        offset += episode["length"]
        _require(all(row.get("episode_id") == episode.get("episode_id") and row.get("decision_index") == index
                     and row.get("terminated") is (index == len(segment)-1) for index, row in enumerate(segment)),
                 "evaluation transition journal is not a complete aligned episode")
        reward = sum(_number(row.get("team_reward"), "trace reward") for row in segment)
        _require(math.isclose(reward, episode["return"], rel_tol=1e-7, abs_tol=1e-7), "evaluation trace return differs")
        _require(segment[-1].get("terminal_reason") == episode["terminal_reason"], "trace terminal reason differs")
        event_sums, previous = {}, [12, 12]
        for row in segment:
            actions = row.get("actions")
            _require(type(actions) is list and len(actions) == 2
                     and all(type(action) is int and 0 <= action < 12 for action in actions)
                     and row.get("previous_actions") == previous, "trace executed-action history differs")
            previous = actions
            for name, value in row.get("event_deltas", {}).items():
                event_sums[name] = event_sums.get(name, 0) + _integer(value, "trace event delta")
        _require(event_sums == episode["event_counts"], "trace task-event counts differ")
    n = len(episodes)
    computed = {"episodes": n, "successes": sum(successes), "success_rate": sum(successes)/n,
                "mean_return": sum(returns)/n, "mean_length": sum(lengths)/n, "mean_event_progress": sum(progress)/n}
    declared = report.get("metrics", {})
    for key, value in computed.items():
        actual = _number(declared.get(key), f"evaluation metric {key}")
        _require(math.isclose(actual, value, rel_tol=1e-7, abs_tol=1e-7), f"evaluation metric {key} differs from episodes")
    return {"metadata": metadata, "protocol": protocol, "keys": keys, "successes": successes,
            "returns": returns, "progress": progress, "metrics": computed,
            "run_identity": report["actor_export"]["run_identity"],
            "portable_weights_sha256": report["actor_export"].get("portable_weights_sha256"), "runtime": runtime}


def _paired(baseline, trained):
    _require(baseline["protocol"] == trained["protocol"] and baseline["keys"] == trained["keys"],
             "baseline and trained evaluation banks or protocols differ")
    _require(baseline["runtime"] == trained["runtime"], "baseline and trained evaluation runtimes differ")
    left, right = baseline["metadata"], trained["metadata"]
    _require(all(left.get(key) == right.get(key) for key in ("task_family", "run_seed", "variant", "actor_config"))
             and baseline["run_identity"] == trained["run_identity"], "evaluation policies are not a paired B run")
    _require(left.get("untrained_zero_interaction_baseline") is True and left.get("local_ppo_cycles") == 0
             and left.get("actor_optimizer_steps") == 0 and left.get("critic_optimizer_steps") == 0,
             "baseline must be the untrained local actor from the same run")
    _require(right.get("untrained_zero_interaction_baseline") is False, "trained export is still marked baseline")
    for key in ("local_ppo_cycles", "actor_optimizer_steps", "critic_optimizer_steps", "real_joint_transitions"):
        _integer(right.get(key), f"trained {key}", 1)
    _require(left.get("actor_version") != right.get("actor_version"), "actor did not change after training")
    _require(type(baseline["portable_weights_sha256"]) is str and len(baseline["portable_weights_sha256"]) == 64
             and type(trained["portable_weights_sha256"]) is str and len(trained["portable_weights_sha256"]) == 64
             and baseline["portable_weights_sha256"] != trained["portable_weights_sha256"], "actor weight bytes did not change")
    n = baseline["metrics"]["episodes"]
    delta = {name: trained["metrics"][name]-baseline["metrics"][name]
             for name in ("success_rate", "mean_return", "mean_event_progress")}
    improved = delta["success_rate"] > 0 or (delta["mean_event_progress"] > 0 and delta["mean_return"] > 0)
    enough = n >= MIN_PAIRED_EPISODES
    wins = sum(b > a for a, b in zip(baseline["successes"], trained["successes"]))
    losses = sum(b < a for a, b in zip(baseline["successes"], trained["successes"]))
    return {"status": "passed" if improved and enough else "needs_review", "paired_episodes": n,
            "minimum_diagnostic_episodes": MIN_PAIRED_EPISODES, "baseline": baseline["metrics"],
            "trained": trained["metrics"], "trained_minus_baseline": delta,
            "success_pair_wins": wins, "success_pair_losses": losses,
            "observed_behavioral_improvement": improved, "sufficient_diagnostic_count": enough,
            "statistical_significance_established": False,
            "baseline_interpretation": "paired untrained actor; task chance level is not assumed",
            "scope": {"task_family": left["task_family"], "run_seed": left["run_seed"], "variant": "B"}}


def aggregate_acceptance(*, invariant_report, mechanics_report=None, baseline_report=None,
                         trained_report=None, output_path):
    """Record exactly which evidence is missing, invalid, negative or sufficient.

    Even a passing single-run exploratory check never launches a campaign, grants
    P2.5 model quality acceptance, or establishes the B-versus-D research claim.
    """
    before = source_identity()
    reasons, references, components, values = [], {}, {}, {}
    specs = (("invariants", invariant_report, _invariants), ("mechanics", mechanics_report, _mechanics),
             ("baseline", baseline_report, _evaluation), ("trained", trained_report, _evaluation))
    for name, path, validate in specs:
        if path is None:
            components[name] = {"status": "missing"}
            reasons.append(f"{name}: evidence has not been provided")
            continue
        try:
            path = Path(path).resolve()
            report = load_report(path)
            _require(report.get("source_sha256") == before, f"{name}: source hashes differ from this checkout")
            values[name] = validate(report, path.parent)
            components[name] = {"status": values[name].get("status", "passed")}
            references[name] = {"path": str(path), "sha256": file_sha256(path)}
            if components[name]["status"] != "passed":
                reasons.append(f"{name}: policy behavior needs review")
        except (ValueError, OSError, KeyError, TypeError, RuntimeError) as exc:
            components[name] = {"status": "invalid", "reason": str(exc)}
            reasons.append(f"{name}: {exc}")
    learning = None
    if "baseline" in values and "trained" in values:
        try:
            learning = _paired(values["baseline"], values["trained"])
            if not learning["sufficient_diagnostic_count"]:
                reasons.append(f"native learning: fewer than {MIN_PAIRED_EPISODES} paired diagnostic episodes")
            if not learning["observed_behavioral_improvement"]:
                reasons.append("native learning: no observed behavioral improvement over the paired untrained actor; diagnose before a large campaign")
        except (ValueError, KeyError, TypeError) as exc:
            components["pairing"] = {"status": "invalid", "reason": str(exc)}
            reasons.append(f"native learning: {exc}")
    if before != source_identity():
        components["source"] = {"status": "invalid", "reason": "source changed during aggregation"}
        reasons.append("source changed during aggregation")
    accepted = not reasons and learning is not None and learning["status"] == "passed"
    if accepted:
        status = "accepted_exploratory_pilot"
    elif any(item["status"] == "invalid" for item in components.values()):
        status = "invalid_evidence"
    elif components["baseline"]["status"] == "missing" or components["trained"]["status"] == "missing":
        status = "pending_real_evidence"
    else:
        status = "needs_review"
    result = {"schema_version": "p36-acceptance-v1", "created_at": datetime.now(timezone.utc).isoformat(),
              "status": status, "phase3_acceptance": accepted, "automatic_campaign_launch": False,
              "scope": "single paired B training-seed/task exploratory integration pilot",
              "accepted_scope": learning["scope"] if accepted else None,
              "reasons": reasons, "components": components, "invariants": values.get("invariants"),
              "mechanics": values.get("mechanics"), "native_learning": learning,
              "evidence": references, "source_sha256": before,
              "new_environment_interactions": 0, "training_executed_by_gate": False,
              "P2.5_quality_acceptance": False,
              "unchecked_classes": ["P2.5 world-model quality acceptance", "other tasks or training seeds",
                                    "B versus D efficacy", "statistical significance", "unseen test-layout generalization"]}
    write_report(output_path, result)
    return result
