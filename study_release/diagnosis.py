"""P6.1: fixed-order diagnosis of saved native endpoint traces.

This module never opens a simulator, loads model weights or executes a learner.
The caller authenticates the completed P5 context before calling this builder;
the native loader additionally checks every selected attempt and trace byte.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re


SCHEMA = "p61-fixed-trace-diagnosis-v1"
_CASES = {(seed, variant) for seed in (0, 1) for variant in ("B", "D")}
_AUTH = ("cohort", "analysis", "final_ledger", "final_actors", "evaluation_traces",
         "input_hashes", "frozen_sources")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _same(left, right):
    return _canonical(left) == _canonical(right)


def _resolve_reference(base, reference):
    if (type(reference) is not dict or type(reference.get("path")) is not str
            or Path(reference["path"]).is_absolute() or "\\" in reference["path"]):
        raise ValueError("diagnostic evidence reference must be a confined relative path")
    base = Path(base).resolve()
    path = (base / reference["path"]).resolve()
    if not path.is_relative_to(base) or not path.is_file():
        raise ValueError("diagnostic evidence escaped its evaluation directory or is missing")
    if _sha(path) != reference.get("sha256") or path.stat().st_size != reference.get("bytes"):
        raise ValueError("diagnostic evidence bytes changed")
    return path


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": _sha(path), "bytes": path.stat().st_size}


def _native_trace(evaluation, evaluation_path, episode):
    # Import the frozen receipt reader only when native saved evidence is read.
    from world_models_study.evidence import read_report

    evaluation_path = Path(evaluation_path).resolve()
    base, index = evaluation_path.parent, episode["episode_index"]
    references = []
    for reference in evaluation.get("attempts", []):
        match = re.fullmatch(r"episodes/([0-9]{6})/attempt_([0-9]{2})/attempt\.json", reference.get("path", ""))
        if match is None:
            raise ValueError("native diagnostic attempt has an invalid fixed-bank path")
        if int(match[1]) == index:
            references.append((int(match[2]), reference))
    references.sort(key=lambda item: item[0])
    if not references or [number for number, _ in references] != list(range(len(references))) or len(references) > 3:
        raise ValueError("selected episode lacks its ordered native attempt receipts")
    sources, complete = [], []
    for number, reference in references:
        attempt_path = _resolve_reference(base, reference)
        attempt = read_report(attempt_path)
        sources.append(_source(attempt_path))
        if (attempt.get("schema_version") != "p54-episode-attempt-v1"
                or attempt.get("episode_index") != index or attempt.get("attempt") != number
                or attempt.get("native_backend") is not True or attempt.get("native_engine") is not True):
            raise ValueError("selected native attempt identity changed")
        if attempt.get("status") == "complete":
            complete.append((attempt_path, attempt))
        elif attempt.get("status") != "failed" or attempt.get("retryable_infrastructure") is not True:
            raise ValueError("selected episode contains an invalid prior retry")
    if len(complete) != 1 or complete[0][1]["attempt"] != references[-1][0]:
        raise ValueError("selected episode must have exactly one final valid outcome")
    attempt_path, attempt = complete[0]
    if not _same(attempt.get("record"), episode) or attempt.get("journal_incomplete") is not False:
        raise ValueError("selected attempt does not reproduce the frozen final record")
    journal_path = _resolve_reference(attempt_path.parent, attempt["journal"])
    sources.extend([_source(journal_path), _source(evaluation_path)])
    raw, previous, transitions, pointers, completions = journal_path.read_bytes(), "0" * 64, [], [], []
    for line_number, line in enumerate(raw.splitlines(keepends=True), 1):
        if not line.endswith(b"\n"):
            raise ValueError("diagnostic native journal has an incomplete row")
        row = json.loads(line)
        body = {key: value for key, value in row.items() if key != "sha256"}
        expected = hashlib.sha256(_canonical(body).encode("utf-8")).hexdigest()
        if (type(row.get("sequence")) is not int or row["sequence"] != line_number - 1
                or row.get("previous_sha256") != previous or row.get("sha256") != expected):
            raise ValueError("diagnostic native journal hash chain changed")
        previous = row["sha256"]
        if row.get("event") == "transition":
            transitions.append(row["transition"])
            pointers.append({"path": str(journal_path), "line": line_number,
                             "sequence": row["sequence"], "row_sha256": row["sha256"]})
        elif row.get("event") == "episode_complete":
            completions.append(row.get("record"))
    if len(completions) != 1 or not _same(completions[0], episode):
        raise ValueError("selected trace has no unique matching completion summary")
    return {"transitions": transitions, "pointers": pointers, "sources": sources,
            "attempt": _source(attempt_path), "journal": _source(journal_path)}


def _trace_summary(loaded, record):
    rows, pointers = loaded.get("transitions"), loaded.get("pointers")
    if (type(rows) is not list or not 1 <= len(rows) <= 400
            or type(pointers) is not list or len(pointers) != len(rows)
            or record.get("joint_transitions") != len(rows)):
        raise ValueError("diagnostic trace length or line pointers differ from the final episode")
    events, first_events = Counter(), {}
    rewards, actions, entropies = Counter(), [Counter(), Counter()], []
    previous, end_tic, total_return, tics = [12, 12], None, 0.0, 0
    for index, row in enumerate(rows):
        if (type(row.get("decision_index")) is not int or row["decision_index"] != index
                or row.get("previous_actions") != previous
                or type(row.get("actions")) is not list or len(row["actions"]) != 2
                or any(type(a) is not int or not 0 <= a < 12 for a in row["actions"])
                or row.get("terminated") is not (index == len(rows) - 1)
                or row.get("truncated") is not False):
            raise ValueError("diagnostic trace action, terminal or decision alignment changed")
        start = _integer(row.get("start_tic"), "start tic")
        end = _integer(row.get("end_tic"), "end tic", 1)
        actual = _integer(row.get("actual_tics"), "actual tics", 1)
        if actual > 4 or end - start != actual or end_tic is not None and start != end_tic:
            raise ValueError("diagnostic trace native tic alignment changed")
        event = row.get("event_deltas")
        if type(event) is not dict or any(type(v) is not int or v < 0 for v in event.values()):
            raise ValueError("diagnostic event increments must be nonnegative integers")
        for name, value in event.items():
            if type(name) is not str:
                raise ValueError("diagnostic event names must be strings")
            if value and name not in first_events:
                first_events[name] = {"decision_index": index, "count": value, "pointer": deepcopy(pointers[index])}
        events.update(event)
        value = _number(row.get("team_reward"), "team reward")
        components = row.get("reward_components")
        if type(components) is not dict:
            raise ValueError("diagnostic reward decomposition is missing")
        values = {name: _number(components.get(name), name) for name in ("step", "terminal", "milestone", "total")}
        if (not math.isclose(sum(values[k] for k in ("step", "terminal", "milestone")), value, abs_tol=1e-9)
                or not math.isclose(values["total"], value, abs_tol=1e-9)
                or components.get("shared_rewards") != [row["team_reward"], row["team_reward"]]):
            raise ValueError("diagnostic reward decomposition differs from real shared reward")
        rewards.update(values)
        entropy = row.get("action_entropy")
        if type(entropy) is not list or len(entropy) != 2:
            raise ValueError("diagnostic trace lacks both local actor entropies")
        local_entropy = [_number(v, "local entropy") for v in entropy]
        if any(v < 0 or v > math.log(12) + 1e-5 for v in local_entropy):
            raise ValueError("diagnostic local entropy is outside the 12-action distribution")
        entropies.extend(local_entropy)
        for player, action in enumerate(row["actions"]):
            actions[player][action] += 1
        previous, end_tic = row["actions"], end
        total_return += value
        tics += actual
    final = rows[-1]
    reason = final.get("terminal_reason")
    if (reason not in ("success", "deadline", "failure", "death")
            or reason == "deadline" and len(rows) != 400
            or record.get("termination_cause") != reason
            or record.get("success") is not (reason == "success")
            or record.get("latch_activated") is not (events["latch_activations"] > 0)
            or not math.isclose(_number(record.get("episode_return"), "episode return"), total_return, abs_tol=1e-9)
            or record.get("simulator_tics") != tics
            or record.get("event_counts") != dict(events)
            or not math.isclose(_number(record.get("mean_action_entropy"), "mean entropy"), sum(entropies) / len(entropies), abs_tol=1e-9)):
        raise ValueError("diagnostic trace does not reproduce its final endpoint summary")
    return {"joint_transitions": len(rows), "simulator_tics": tics, "termination_cause": reason,
            "episode_return": total_return, "event_counts": dict(sorted(events.items())),
            "first_positive_event": dict(sorted(first_events.items())),
            "reward_component_totals": dict(sorted(rewards.items())),
            "action_counts_by_player": [{str(a): counts[a] for a in range(12)} for counts in actions],
            "action_entropy": {"mean": sum(entropies) / len(entropies), "minimum": min(entropies),
                               "maximum": max(entropies), "finite": True},
            "first_transition": deepcopy(pointers[0]), "last_transition": deepcopy(pointers[-1]),
            "no_images_or_player_positions_recorded": True,
            "matching_model_reward_or_terminal_predictions_recorded": False}


def _classifications(record, summary):
    success, latch = record["success"], record["latch_activated"]
    if success:
        coordination = ("observed_task_completion", "The recorded native task reached authoritative success.")
    elif latch:
        coordination = ("observed_incomplete_after_latch", "A latch activation was recorded, but the episode ended without success.")
    else:
        coordination = ("observed_missing_latch_milestone", "No latch activation or task success was recorded; coordination was not completed.")
    return {
        "mechanics_local_ambiguity": {"status": "undetermined", "causal_effect_established": False,
            "reason": "Saved action/event traces lack local RGB frames, player positions and counterfactual mechanics tests."},
        "incomplete_coordination": {"status": coordination[0], "observation": coordination[1],
            "causal_effect_established": False, "basis": [summary["last_transition"]]},
        "model_reward_termination_mismatch": {"status": "undetermined", "causal_effect_established": False,
            "reason": "Native endpoint records contain no aligned world-model reward or termination predictions; earlier P2.5 issues are separate evidence."},
        "unstable_optimization": {"status": "undetermined", "causal_effect_established": False,
            "finite_endpoint_rewards_and_action_entropies": True,
            "reason": "Finite saved endpoint outputs do not rule out optimization instability or establish its causal role."},
        "unsuccessful_transfer": {"status": "assessed_at_paired_cohort_level", "causal_effect_established": False,
            "reason": "An individual failed episode cannot isolate the effect of teacher warm-start."}}


def _interpretation(arms):
    indexed = {(r["master_seed"], r["variant"]): r for r in arms}
    differences = []
    for seed in (0, 1):
        b, d = indexed[seed, "B"], indexed[seed, "D"]
        differences.append({"master_seed": seed,
            "success_rate_difference_D_minus_B": d["successes"] / 20 - b["successes"] / 20,
            "mean_return_difference_D_minus_B": d["mean_return"] - b["mean_return"],
            "latch_episode_rate_difference_D_minus_B": d["latch_episodes"] / 20 - b["latch_episodes"] / 20})
    signs = [r["success_rate_difference_D_minus_B"] for r in differences]
    if all(r["successes"] == 0 for r in arms):
        code, text = "both_arms_unsuccessful", "Both arms remain unsuccessful under this tested model, task and budget; successful transfer is not established."
    elif all(value > 0 for value in signs):
        code, text = "D_higher_success_in_both_pairs", "Consistent exploratory endpoint evidence for this warm-start procedure in both tested pairs; broader replication and causal ablations remain absent."
    elif all(value == 0 for value in signs) and any(
            r["mean_return_difference_D_minus_B"] > 0 or r["latch_episode_rate_difference_D_minus_B"] > 0 for r in differences):
        code, text = "secondary_progress_without_success_benefit", "At least one pair shows secondary progress, with no observed success-rate benefit; all per-seed signs must remain visible."
    else:
        code, text = "inconsistent_or_unhelpful_transfer", "Transfer is inconsistent, unhelpful or worse under this tested protocol; the two pairs do not establish a general or isolated causal effect."
    return {"code": code, "observation": text, "paired_differences": differences,
            "causal_component_effect_established": False, "statistical_superiority_established": False,
            "successful_transfer_established": False, "training_seed_pairs": 2,
            "scope": "relay_four_day_v1; four final actors; two held-out authored Relay layouts; 20 episodes per actor"}


def _history(context):
    provided = context.get("historical_evidence", [])
    if type(provided) is not list or any(type(row) is not dict for row in provided):
        raise ValueError("historical evidence must be a separate list")
    rows, sources = [], []
    for kind, label in (("cover_model_validation", "Cover initial model validation"),
                        ("B_development", "relay_week_v1 long B development run")):
        matched = [row for row in provided if row.get("kind") == kind]
        for row in matched:
            path = Path(row["report_path"]).resolve()
            if not path.is_file() or _sha(path) != row.get("sha256"):
                raise ValueError("separate historical evidence changed")
            sources.append(_source(path))
        rows.append({"kind": kind, "label": label, "status": "archived_separately" if matched else "not_packaged",
            "included_in_paired_endpoint_metrics": False, "records": deepcopy(matched),
            "scope_note": ("Initial Cover model failures informed narrowing to Relay; no Cover policy outcome is inferred."
                           if kind == "cover_model_validation" else
                           "Development B uses a different longer budget; it is not pooled with either 20-cycle final cohort arm.")})
    if any(row.get("kind") not in {"cover_model_validation", "B_development"} for row in provided):
        raise ValueError("unknown historical evidence category")
    return rows, sources


def validate_diagnosis_sources(report):
    rows = report.get("diagnostic_sources")
    if type(rows) is not list:
        raise ValueError("diagnosis lacks its exact saved evidence references")
    for reference in rows:
        if (type(reference) is not dict or type(reference.get("path")) is not str
                or not Path(reference["path"]).is_absolute()
                or type(reference.get("sha256")) is not str
                or re.fullmatch(r"[0-9a-f]{64}", reference["sha256"]) is None
                or type(reference.get("bytes")) is not int or reference["bytes"] < 0):
            raise ValueError("diagnosis contains an invalid saved evidence reference")
        path = Path(reference["path"])
        if not path.is_file() or _sha(path) != reference["sha256"] or path.stat().st_size != reference["bytes"]:
            raise ValueError("saved diagnostic evidence changed")


def render_markdown(report):
    """Render the archived JSON without rereading or executing scientific code."""
    if report.get("schema_version") != SCHEMA or report.get("status") != "complete":
        raise ValueError("only a complete fixed-trace diagnosis can be rendered")
    lines = ["# P6.1 fixed trace diagnosis", "",
        f"Cohort: `{report['cohort_identity']}`.", "",
        f"Selected {report['selected_count']} of at most eight saved traces: the first successful and first failed episode per arm in final-bank order.", "",
        report["endpoint_interpretation"]["observation"], "",
        "This is descriptive endpoint evidence. The selected traces do not establish isolated causal component effects, statistical superiority or successful transfer.", "",
        "| Seed | Variant | Category | Episode | Layout | Cause | Return | Latch |", "|---|---|---|---|---|---|---|---|"]
    for row in report["selected_traces"]:
        record = row["record"]
        lines.append(f"| {row['master_seed']} | {row['variant']} | {row['category']} | {row['episode_index']} | {row['template_id']} | {record['termination_cause']} | {record['episode_return']:.8g} | {record['latch_activated']} |")
    lines.extend(["", "## Missing outcome categories", ""])
    if report["missing_categories"]:
        for row in report["missing_categories"]:
            lines.append(f"- {row['variant']}/seed_{row['master_seed']}: no {row['category']} episode exists in the complete final bank.")
    else:
        lines.append("Both outcome categories exist in each final arm.")
    lines.extend(["", "## Trace-backed observations and limits", ""])
    for row in report["selected_traces"]:
        summary = row["summary"]
        last = summary["last_transition"]
        lines.extend([f"### {row['variant']}/seed_{row['master_seed']}, episode {row['episode_index']} ({row['category']})", "",
            row["classifications"]["incomplete_coordination"]["observation"], "",
            f"Native terminal transition: `{last['path']}`, line {last['line']}; row SHA-256 `{last['row_sha256']}`.", "",
            f"Saved event totals: `{_canonical(summary['event_counts'])}`. Reward component totals: `{_canonical(summary['reward_component_totals'])}`.", ""])
        for name in ("mechanics_local_ambiguity", "model_reward_termination_mismatch", "unstable_optimization", "unsuccessful_transfer"):
            diagnosis = row["classifications"][name]
            lines.append(f"- {name.replace('_', ' ')}: {diagnosis['status']}. {diagnosis['reason']}")
        lines.append("")
    lines.extend(["## Separate development history", ""])
    for row in report["separate_history"]:
        lines.extend([f"- {row['label']}: {row['status']}. {row['scope_note']}"])
    lines.extend(["", "No simulator execution, new episodes, training or model updates were performed by this reporting command.", ""])
    return "\n".join(lines)


def build_diagnosis(context, *, trace_loader=None):
    """Select first success/failure in bank order for each authenticated arm.

    ``trace_loader`` is solely a synthetic test seam. Its use marks the output
    fixture-only and cannot serve as native P6 acceptance evidence.
    """
    if any(context.get("authentication", {}).get(key) is not True for key in _AUTH):
        raise ValueError("P6.1 requires the authenticated completed P5 evidence context")
    cohort, reports, paths = context["cohort"], context.get("evaluation_reports"), context.get("evaluation_paths")
    if (type(reports) is not list or len(reports) != 4 or type(paths) is not list or len(paths) != 4
            or any(type(r) is not dict or type(r.get("master_seed")) is not int for r in reports)
            or {(r.get("master_seed"), r.get("variant")) for r in reports} != _CASES):
        raise ValueError("P6.1 requires exactly all four final paired evaluations")
    bank = cohort.get("bank", {}).get("episodes")
    if (type(bank) is not list or len(bank) != 20
            or [r.get("episode_index") for r in bank] != list(range(20))):
        raise ValueError("P6.1 requires the complete fixed bank order")
    cohort_id = cohort["cohort_identity"]
    indexed = {}
    for evaluation, path in zip(reports, paths):
        if (evaluation.get("schema_version") != "p54-final-test-evaluation-v1"
                or evaluation.get("status") != "complete" or evaluation.get("cohort_identity") != cohort_id
                or evaluation.get("protocol_sha256") != cohort.get("protocol_sha256")
                or evaluation.get("bank_sha256") != cohort.get("bank_sha256")
                or evaluation.get("task_family") != "relay" or evaluation.get("final_local_cycles") != 20
                or evaluation.get("fixture_only") is not False
                or any(evaluation.get(key) is not True for key in ("native_backend", "native_engine",
                       "native_environment_evaluation_executed", "research_evaluation_executed"))):
            raise ValueError("diagnosis cannot use incomplete, non-native, development or mismatched cohort evidence")
        episodes = evaluation.get("episodes")
        if (type(episodes) is not list or len(episodes) != 20
                or {r.get("episode_index") for r in episodes} != set(range(20))):
            raise ValueError("diagnosis requires every final-bank episode without duplication")
        episode_map = {row["episode_index"]: row for row in episodes}
        for entry in bank:
            record = episode_map[entry["episode_index"]]
            if (not _same(record.get("bank_entry"), entry) or record.get("template_id") != entry.get("template_id")
                    or type(record.get("success")) is not bool
                    or record.get("termination_cause") not in ("success", "deadline", "failure", "death")
                    or record["success"] is not (record["termination_cause"] == "success")
                    or type(record.get("latch_activated")) is not bool):
                raise ValueError("episode outcome differs from its registered final bank")
            _number(record.get("episode_return"), "episode return")
        indexed[evaluation["master_seed"], evaluation["variant"]] = (evaluation, path, episode_map)
    loader, selected, missing, arms, sources = trace_loader or _native_trace, [], [], [], {}
    for seed, variant in sorted(_CASES):
        evaluation, path, episode_map = indexed[seed, variant]
        ordered = [episode_map[entry["episode_index"]] for entry in bank]
        arms.append({"master_seed": seed, "variant": variant, "episodes": 20,
                     "successes": sum(row["success"] for row in ordered),
                     "mean_return": math.fsum(row["episode_return"] for row in ordered) / 20,
                     "latch_episodes": sum(row["latch_activated"] for row in ordered)})
        for category in ("successful", "failed"):
            candidate = next((row for row in ordered if row["success"] is (category == "successful")), None)
            if candidate is None:
                missing.append({"master_seed": seed, "variant": variant, "category": category,
                                "reason": "no such episode in the complete final bank", "eligible_episodes": 0})
                continue
            loaded = loader(evaluation, path, candidate)
            summary = _trace_summary(loaded, candidate)
            for reference in loaded.get("sources", []):
                if reference["path"] in sources and sources[reference["path"]] != reference:
                    raise ValueError("a selected trace changed during diagnostic reading")
                sources[reference["path"]] = deepcopy(reference)
            if trace_loader is None and not loaded.get("sources"):
                raise ValueError("native diagnostic trace has no exact source bindings")
            selected.append({"master_seed": seed, "variant": variant, "category": category,
                "episode_index": candidate["episode_index"], "episode_id": candidate.get("episode_id"),
                "template_id": candidate["template_id"], "evaluation_path": str(path),
                "record": deepcopy(candidate), "summary": summary,
                "classifications": _classifications(candidate, summary),
                "selection_rule": "first matching outcome in the frozen final-bank episode order"})
    history, historical_sources = _history(context)
    for reference in historical_sources:
        sources[reference["path"]] = reference
    result = {"schema_version": SCHEMA, "status": "complete", "cohort_identity": cohort_id,
        "protocol_sha256": cohort["protocol_sha256"], "bank_sha256": cohort["bank_sha256"],
        "study_profile": "relay_four_day_v1", "sampling_rule": "first success and first failure per seed/variant in final-bank order",
        "maximum_selected_traces": 8, "selected_count": len(selected), "selected_traces": selected,
        "missing_categories": missing, "outcome_categories_complete": len(selected) + len(missing) == 8,
        "full_bank_arm_outcomes": arms, "endpoint_interpretation": _interpretation(arms),
        "separate_history": history, "diagnostic_sources": [sources[p] for p in sorted(sources)],
        "fixtures_only": trace_loader is not None, "causal_component_effects_established": False,
        "manual_followup": "Review the trace pointers and explicit undetermined categories; no new ablation campaign is required.",
        "new_training_or_test_execution": False, "research_training_executed": False,
        "research_evaluation_executed": False, "native_environment_opened": False,
        "new_model_updates": 0, "new_policy_optimizer_updates": 0, "new_environment_interactions": 0}
    validate_diagnosis_sources(result)
    return result
