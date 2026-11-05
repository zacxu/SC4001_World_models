"""Bounded P2.5 engineering evidence for an exploratory B/D integration pilot.

This evaluator never fits, collects simulator data, finalizes a P2.4 run, or
declares formal quality acceptance. Recorded actions are fixed; future images
and privileged labels reach scoring only after prior-only imagination.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
import sys
import time

import numpy as np
import torch

from world_models.data.replay import canonical_hash, sha256_file
from world_models.data.schema import REASONS
from world_models.models.heads import TaskSpec, encode_auxiliary_targets, encode_transition_targets
from world_models.models.world_model import SamplingStreams, build_model
from world_models.training.checkpoints import read_checkpoint
from world_models.training.runner import CASES, load_case, runtime_inventory, selected_device, source_identity
from .metrics import MetricsAccumulator
from .selection import select_diagnostic_cohort
from .verification import prepare_verification, source_identity as evaluator_source_identity, write_json


VERSION = "compact-p25-engineering-v1"
HORIZONS = (1, 5, 10, 15)
# Fixed before server scores. These are modest engineering alarms, not a
# statistically calibrated acceptance test or permission to claim efficacy.
PILOT_THRESHOLDS = {
    "minimum_uniform_windows": 16, "minimum_full_15_step_uniform_windows": 8,
    "minimum_distinct_positives_per_milestone": 5,
    "minimum_distinct_successes": 5, "minimum_success_recall": .5,
    "minimum_sampled_terminal_recall": .5,
    "minimum_prediction_coverage": .75, "maximum_false_early_window_rate": .25,
    "rgb_mse_allowance_above_persistence": .02,
    "outcome_brier_allowance_above_fitting_baseline": .05,
    "reward_cumulative_baseline_multiplier": 1.25,
    "reward_cumulative_absolute_allowance": .05,
    "minimum_milestone_recall": .5, "minimum_milestone_precision": .25,
    "milestone_brier_allowance_above_fitting_baseline": .02,
}


def milestone_names(task):
    return (("event/latch_activations",) if task.task_family == "relay" else
            ("event/console_1_completions", "event/console_2_completions"))


def _seed(*parts):
    return int(canonical_hash({"version": VERSION, "parts": list(parts)})[:15], 16)


def _array(value):
    return value.detach().float().cpu().numpy()


def _tensor(value, device, *, dtype=None):
    # Replay arrays are immutable; explicitly copy instead of aliasing them.
    return torch.tensor(value, device=device, dtype=dtype)


@contextmanager
def _inference(precision, device):
    with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                                enabled=precision == "bfloat16"):
        yield


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def fitting_baseline(stats, task):
    """Frozen fitting frequencies, never estimates from diagnostic labels.

    Expected bonus is implied by fitting milestone frequencies and task rules;
    it is explicitly not advertised as an empirical mean of team reward.
    """
    profile = stats.provenance["label_profile"]
    learned = profile["learned_outcome"]
    counts = np.asarray(learned["counts"], dtype=np.float64)
    if counts.shape != (3,) or learned["valid"] <= 0 or counts.sum() != learned["valid"]:
        raise ValueError("frozen fitting outcome frequencies are unavailable")
    probabilities = counts / learned["valid"]
    events, support = {}, {}
    for name in milestone_names(task):
        field = profile["binary_fields"][name]
        if field["alignment"] != "transition" or field["valid"] <= 0:
            raise ValueError(f"frozen fitting event frequency unavailable: {name}")
        events[name] = field["positive"] / field["valid"]
        support[name] = {"valid": field["valid"], "positive": field["positive"]}
    return {"outcome_probabilities": probabilities.tolist(), "milestones": events,
            "expected_bonus": task.milestone_bonus * sum(events.values()),
            "source": "frozen initial fitting-only normalization label_profile",
            "outcome_support": deepcopy(learned), "milestone_support": support,
            "reward_rule": "milestone-frequency implied bonus + step cost + expected terminal reward; exact clock expiry",
            "normalization_identity_sha256": stats.as_dict()["identity_sha256"]}


def window_baseline(baseline, task, source_clock):
    probabilities = np.asarray(baseline["outcome_probabilities"], dtype=np.float64)
    at_deadline = np.asarray(source_clock) + 1 == task.deadline
    termination = np.full(len(source_clock), probabilities[1] + probabilities[2])
    termination[at_deadline] = 1.
    reward = np.full(len(source_clock), baseline["expected_bonus"] + task.step_cost +
                     probabilities[1]*task.success_reward + probabilities[2]*task.failure_reward)
    reward[at_deadline] += probabilities[0]*task.failure_reward
    return {**baseline, "reward": reward, "termination_probability": termination}


class PrefixCache:
    """Model-local posterior states; full real prefix, no decoding during burn-in.

    States at selected earlier anchors are retained when a later anchor is
    encountered first. Named episode streams make cache/order/cohort size
    independent of the future imagination stream. No real suffix is encoded.
    """
    def __init__(self, model, *, device, seed, checkpoint_sha256, records):
        self.model, self.device, self.seed = model, torch.device(device), seed
        self.identity = checkpoint_sha256
        self.anchors, self.entries = {}, {}
        self.observed_steps, self.hits = 0, 0
        for record in records:
            key = (record["dataset_index"], record["episode_id"])
            self.anchors.setdefault(key, set()).add(record["anchor"])

    def get(self, window):
        if window.start != 0 or not bool(window.states["image_valid"][:window.anchor+1].all()):
            raise ValueError("P2.5 conditioning requires a complete two-view real prefix from reset")
        key = (window.dataset_index, window.episode_id)
        entry = self.entries.get(key)
        if entry is None:
            streams = SamplingStreams.from_seed(_seed(self.seed, self.identity, *key, "prefix"), self.device)
            state = self.model.initialize(_tensor(window.states["rgb"][0:1], self.device), streams=streams)
            entry = {"index": 0, "state": state, "streams": streams, "anchors": {}}
            self.entries[key] = entry
            if 0 in self.anchors[key]:
                entry["anchors"][0] = state.detach()
        if window.anchor in entry["anchors"]:
            self.hits += 1
            return entry["anchors"][window.anchor]
        if window.anchor < entry["index"]:
            raise ValueError("prefix cache was not constructed for the selected anchor")
        state, streams = entry["state"], entry["streams"]
        for index in range(entry["index"], window.anchor):
            actions = _tensor(window.transitions["actions"][index:index+1], self.device, dtype=torch.long)
            prior = self.model.core.prior_step(state, actions, generator=streams.prior)
            image = self.model._images(_tensor(window.states["rgb"][index+1:index+2], self.device))
            state = self.model.core.observe(prior, image, generator=streams.posterior)
            if int(state.decision_index.item()) != index+1:
                raise ValueError("reconstructed prefix clock differs from stored decision index")
            self.observed_steps += 1
            if index+1 in self.anchors[key]:
                entry["anchors"][index+1] = state.detach()
        entry.update(index=window.anchor, state=state)
        return entry["anchors"][window.anchor]


def targets_for_window(window, task):
    source = slice(window.anchor, window.stop)
    destination = slice(window.anchor+1, window.stop+1)
    transitions = {name: value[source] for name, value in window.transitions.items()}
    states = {name: value[destination] for name, value in window.states.items()}
    labels = encode_transition_targets(transitions, task)
    auxiliary = encode_auxiliary_targets(states, transitions, task, window.event_fields)
    return {"rgb": states["rgb"], "image_valid": states["image_valid"],
            "reward": _array(labels.total_reward), "outcome": labels.outcome.cpu().numpy(),
            "outcome_valid": labels.outcome_valid.cpu().numpy(),
            "terminated": transitions["terminated"],
            "deadline": transitions["termination_reason"] == REASONS.index("deadline"),
            "transition_index": transitions["decision_index"],
            "stage": auxiliary["task/stage"].value.cpu().numpy(),
            "stage_valid": auxiliary["task/stage"].valid.cpu().numpy(),
            "milestones": {name: {"value": _array(auxiliary[name].value),
                                   "valid": auxiliary[name].valid.cpu().numpy()}
                           for name in milestone_names(task)}}


def predict_window(model, anchor, window, *, device, seed, checkpoint_sha256):
    steps = window.loss_steps
    actions = _tensor(window.transitions["actions"][window.anchor:window.stop], device, dtype=torch.long)[:, None]
    streams = SamplingStreams.from_seed(_seed(seed, checkpoint_sha256, window.dataset_index,
        window.episode_id, window.anchor, "future"), device)
    # Exactly one rollout: extract all four scoring horizons from its outputs.
    imagined = model.imagine(anchor, actions, streams=streams)
    result = {"active": np.zeros(steps, dtype=bool), "rgb": np.zeros((steps, 2, 3, 96, 128), np.float32),
              "reward": np.zeros(steps), "outcome_probabilities": np.zeros((steps, 3)),
              "termination_probability": np.zeros(steps), "terminated": np.zeros(steps, dtype=bool),
              "stage_probabilities": np.zeros((steps, model.task.stage_classes)),
              "milestones": {name: np.zeros(steps) for name in milestone_names(model.task)}}
    for index, item in enumerate(imagined):
        if index >= steps or not bool(item.valid.item()) or bool(item.transition.posterior_valid.any()):
            raise ValueError("invalid prior-only imagination validity")
        if int(item.next.world.decision_index.item()) != window.anchor+index+1:
            raise ValueError("imagined decision clock differs from the recorded action prefix")
        probabilities = _array(item.transition.predictions.outcome_logits.softmax(-1))[0]
        result["active"][index] = True
        result["rgb"][index] = _array(item.transition.rgb)[0]
        result["reward"][index] = item.reward.float().item()
        result["outcome_probabilities"][index] = probabilities
        result["termination_probability"][index] = (1. if window.anchor+index+1 == model.task.deadline
                                                       else float(probabilities[1:].sum()))
        result["terminated"][index] = bool(item.terminated.item())
        result["stage_probabilities"][index] = _array(item.transition.auxiliary["task/stage"].softmax(-1))[0]
        for name in result["milestones"]:
            result["milestones"][name][index] = item.transition.auxiliary[name].sigmoid().float().item()
    return result


def _load_model(checkpoint, *, data_root, statistics_root, config_root, device, precision):
    payload = read_checkpoint(checkpoint)
    meta = payload["metadata"]
    if (meta.get("profile"), meta.get("stage"), meta.get("variant")) != ("pilot", "initial", "shared_initial:B,D"):
        raise ValueError("compact P2.5 requires a shared initial pilot checkpoint; overfit/refresh models are excluded")
    task, seed = meta.get("task_family"), meta.get("run_seed")
    if (task, seed) not in CASES or payload["update"] < 1:
        raise ValueError("checkpoint must be an updated relay/cover pilot seed 0/1 model")
    cfg, datasets, stats, provenance = load_case(task, seed, data_root=data_root,
        statistics_root=statistics_root, config_root=config_root, profile="pilot")
    common = dict(cfg.settings)
    common.pop("variant")
    expected = {"source_sha256": source_identity(), "common_config_sha256": canonical_hash(common), **provenance}
    for name, value in expected.items():
        if meta.get(name) != value:
            raise ValueError(f"evaluation {name} differs from the checkpoint")
    precision = precision or meta["precision"]
    if precision not in ("float32", "bfloat16"):
        raise ValueError("unsupported evaluation precision")
    chosen = selected_device(device, precision)
    model = build_model(cfg.settings, master_seed=seed, variant="B")
    current = model.state_dict()
    saved = payload["model"]
    if set(saved) != set(current) or any(saved[name].shape != value.shape or saved[name].dtype != value.dtype
                                         for name, value in current.items()):
        raise ValueError("checkpoint model tensor names/shapes/dtypes differ")
    model.load_state_dict(saved, strict=True)
    model.to(chosen).eval()
    return model, datasets, stats, {"metadata": meta, "update": payload["update"],
                                    "precision": precision, "device": chosen}


def pilot_readiness(metrics, *, task, full_heldout_support):
    thresholds = deepcopy(PILOT_THRESHOLDS)
    reasons, unchecked = [], ["unseen-layout generalization", "real-policy learning", "B versus D efficacy",
                              "trained ordinary/event overfit and action-effect evidence: retain separate existing results if available"]
    uniform, event = metrics["uniform"]["15"], metrics["event"]["15"]
    def require(condition, reason):
        if not condition:
            reasons.append(reason)
    require(uniform["windows"] >= thresholds["minimum_uniform_windows"], "insufficient uniform windows; timing alone is not readiness")
    require(uniform["full_horizon_windows"] >= thresholds["minimum_full_15_step_uniform_windows"], "insufficient complete 15-step real uniform suffixes")
    for stratum, report in (("uniform", uniform), ("event", event)):
        require((report["coverage"]["fraction"] or 0) >= thresholds["minimum_prediction_coverage"], f"{stratum}: excessive missing model predictions")
        require(report["termination"]["false_early_window_rate"] is not None and
                report["termination"]["false_early_window_rate"] <= thresholds["maximum_false_early_window_rate"],
                f"{stratum}: excessive false early termination or no termination support")
        for loss, baseline, allowance, label in (
            (report["rgb"]["coverage_adjusted_mse"]["mean"], report["rgb"]["persistence_mse"]["mean"],
             thresholds["rgb_mse_allowance_above_persistence"], "RGB versus persistence"),
            (report["outcome"]["coverage_adjusted_brier"]["mean"], report["outcome"]["baseline_brier"]["mean"],
             thresholds["outcome_brier_allowance_above_fitting_baseline"], "learned outcome versus fitting frequency")):
            require(loss is not None and baseline is not None and loss <= baseline+allowance,
                    f"{stratum}: {label} needs review")
        reward = report["reward"]
        for loss_name, baseline_name in (("cumulative_mae", "baseline_cumulative_mae"),
                                         ("cumulative_path_mae", "baseline_cumulative_path_mae")):
            model_error, baseline_error = reward[loss_name]["mean"], reward[baseline_name]["mean"]
            require(model_error is not None and baseline_error is not None and model_error <=
                    thresholds["reward_cumulative_baseline_multiplier"]*baseline_error +
                    thresholds["reward_cumulative_absolute_allowance"], f"{stratum}: {loss_name} reward needs review")
    for name in milestone_names(task):
        score = event["milestones"].get(name, {})
        require((score.get("unique_positive_events") or 0) >= thresholds["minimum_distinct_positives_per_milestone"],
                f"insufficient distinct held-out positives for {name}")
        require((score.get("recall") or 0) >= thresholds["minimum_milestone_recall"] and
                (score.get("precision") or 0) >= thresholds["minimum_milestone_precision"], f"{name}: event behavior needs review")
        loss = score.get("coverage_adjusted_brier", {}).get("mean")
        baseline = score.get("baseline_brier", {}).get("mean")
        require(loss is not None and baseline is not None and loss <= baseline+
                thresholds["milestone_brier_allowance_above_fitting_baseline"], f"{name}: event Brier needs review")
    success = event["outcome"]["per_class"]["success"]
    require((success.get("unique_positive_events") or 0) >= thresholds["minimum_distinct_successes"],
            "insufficient distinct held-out successes")
    require((success.get("recall") or 0) >= thresholds["minimum_success_recall"],
            "learned success recall needs review; continuing forever is not adequate")
    require((event["termination"]["per_class"]["terminal"]["recall"] or 0) >=
            thresholds["minimum_sampled_terminal_recall"], "sampled terminal recall needs review")
    # Learned failure/death is different from known-clock deadline failure.
    failure_support = sum(metrics[stratum]["15"]["outcome"]["class_support"][2] for stratum in ("uniform", "event"))
    if failure_support == 0:
        unchecked.append("learned nondeadline failure/death prediction: no selected positives")
    reasons_in_source = full_heldout_support.get("transition_reasons", {})
    for reason in ("death", "failure"):
        if reasons_in_source.get(reason, 0) == 0:
            unchecked.append(f"{reason} prediction: zero positives in the complete diagnostic source")
    return {"status": "needs_review" if reasons else "ready_for_exploratory_phase3", "reasons": reasons,
            "unchecked_classes": unchecked, "thresholds": thresholds,
            "scope": "provisional engineering screen for a small exploratory Phase 3 integration pilot only",
            "P2.5_quality_acceptance": False, "automatic_phase3_launch": False,
            "full_heldout_support": deepcopy(full_heldout_support)}


def _fresh(directory):
    directory = Path(directory)
    if directory.exists() and any(directory.iterdir()):
        raise FileExistsError("use a fresh P2.5 output directory; evidence is not overwritten")
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _bounds(windows, seed):
    if type(windows) is not int or not 1 <= windows <= 64 or type(seed) is not int or seed < 0:
        raise ValueError("windows must be 1..64 and seed must be nonnegative")


def _failure(directory, error, **context):
    report = {"schema_version": VERSION, "P2.5_quality_acceptance": False,
              "pilot_readiness": {"status": "failed", "reasons": [f"{type(error).__name__}: {error}"]},
              "new_optimizer_updates": 0, "new_real_interactions": 0, **context}
    path = Path(directory)/"report.json"
    if path.exists():
        path = Path(directory)/"failure.json"
    write_json(path, report)
    return report


def _evaluate_one(checkpoint, directory, *, verification, data_root, statistics_root, config_root,
                  device, windows, seed, precision):
    start = time.perf_counter()
    times = {"prefix_seconds": 0., "rollout_seconds": 0., "scoring_seconds": 0., "window_io_seconds": 0.}
    before = evaluator_source_identity()
    model, datasets, stats, loaded = _load_model(checkpoint, data_root=data_root, statistics_root=statistics_root,
                                                config_root=config_root, device=device, precision=precision)
    chosen = loaded["device"]
    times["checkpoint_and_provenance_seconds"] = time.perf_counter()-start
    checkpoint_hash = sha256_file(checkpoint)
    selection_start = time.perf_counter()
    cohort = select_diagnostic_cohort(datasets, seed=seed, max_windows=windows, horizon=15, task=model.task)
    times["selection_and_replay_index_seconds"] = time.perf_counter()-selection_start
    baseline = fitting_baseline(stats, model.task)
    accumulators = {stratum: {str(h): MetricsAccumulator() for h in HORIZONS} for stratum in ("uniform", "event")}
    records, rows = cohort.records, []
    cache = PrefixCache(model, device=chosen, seed=seed, checkpoint_sha256=checkpoint_hash, records=records)
    with _inference(loaded["precision"], chosen):
        for position, record in enumerate(records, start=1):
            tick = time.perf_counter()
            window = cohort.window(record)
            times["window_io_seconds"] += time.perf_counter()-tick
            _sync(chosen)
            tick = time.perf_counter()
            anchor = cache.get(window)
            _sync(chosen)
            times["prefix_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            predictions = predict_window(model, anchor, window, device=chosen, seed=seed, checkpoint_sha256=checkpoint_hash)
            _sync(chosen)
            times["rollout_seconds"] += time.perf_counter()-tick
            tick = time.perf_counter()
            targets = targets_for_window(window, model.task)
            predictor = window_baseline(baseline, model.task, targets["transition_index"])
            episode_key = f"{record['dataset_identity_sha256']}:{window.episode_id}"
            for horizon in HORIZONS:
                accumulators[record["stratum"]][str(horizon)].add_window(targets, predictions,
                    anchor_rgb=window.states["rgb"][window.anchor], baseline=predictor,
                    episode_id=episode_key, horizon=horizon)
            rows.append({**record, "predicted_steps": int(predictions["active"].sum()),
                         "sampled_termination": predictions["terminated"].tolist()})
            times["scoring_seconds"] += time.perf_counter()-tick
            del targets, predictions, predictor, window
            if position % 8 == 0 or position == len(records):
                print(f"P2.5 {cohort.task_family}/seed_{cohort.run_seed}: {position}/{len(records)} windows",
                      file=sys.stderr, flush=True)
    metrics = {stratum: {h: acc.report() for h, acc in values.items()} for stratum, values in accumulators.items()}
    if evaluator_source_identity() != before:
        raise RuntimeError("source changed during evaluation; evidence invalid")
    meta = loaded["metadata"]
    readiness = pilot_readiness(metrics, task=model.task, full_heldout_support=cohort.report["full_heldout_support"])
    times["total_seconds"] = time.perf_counter()-start
    report = {"schema_version": VERSION, "P2.5_quality_acceptance": False,
              "checkpoint": {"path": str(Path(checkpoint).resolve()), "sha256": checkpoint_hash,
                  "update": loaded["update"], "task_family": meta["task_family"], "run_seed": meta["run_seed"],
                  "shared_by_variants": ["B", "D"], "original_target_stage_updates": meta["target_stage_updates"],
                  "original_training_target_completed": loaded["update"] >= meta["target_stage_updates"],
                  "training_metadata": meta},
              "selection": cohort.report, "windows": rows, "metrics": metrics,
              "baseline": baseline, "pilot_readiness": readiness, "verification": verification,
              "overfit_evidence": {"status": "not_supplied", "re_run": False,
                  "scope": "retain existing trained-checkpoint ordinary/event overfit evidence separately if available; structural fixtures do not establish learned action effects"},
              "source_sha256": before, "runtime": runtime_inventory(chosen),
              "evaluation_precision": loaded["precision"], "seed": seed,
              "timing": times, "prefix_cache": {"observed_joint_steps": cache.observed_steps, "hits": cache.hits,
                  "identity": "checkpoint SHA + replay/episode + exact prefix + named sampling streams"},
              "new_optimizer_updates": 0, "new_real_interactions": 0,
              "interpretation": ["fixed recorded actions; no policy evaluation", "diagnostic episodes on training geometry",
                  "windows and horizons overlap; not independent trials", "one stochastic rollout per anchor",
                  "known-clock deadlines do not validate learned death/failure", "timing includes full prefix and I/O; no runtime guarantee"]}
    write_json(Path(directory)/"report.json", report)
    return report


def evaluate_checkpoint(checkpoint, output_dir, *, data_root, statistics_root, config_root=None,
                        device="cuda:0", windows=8, seed=0, precision=None, verification_report=None):
    _bounds(windows, seed)
    directory = _fresh(output_dir)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        evidence = prepare_verification(directory/"verification", reuse_path=verification_report)
        return _evaluate_one(Path(checkpoint), directory, verification=evidence, data_root=data_root,
            statistics_root=statistics_root, config_root=config_root, device=device, windows=windows,
            seed=seed, precision=precision)
    except Exception as error:
        _failure(directory, error, checkpoint_path=str(checkpoint))
        raise
    finally:
        torch.set_num_threads(threads)


def evaluate_campaign(run_root, checkpoint_update, output_dir, *, data_root, statistics_root,
                      config_root=None, device="cuda:0", windows=8, seed=0, precision=None, verification_report=None):
    _bounds(windows, seed)
    if type(checkpoint_update) is not int or checkpoint_update < 1:
        raise ValueError("campaign checkpoint update must be a positive explicit common update")
    directory = _fresh(output_dir)
    start, threads = time.perf_counter(), torch.get_num_threads()
    reports, current_case = [], None
    try:
        torch.set_num_threads(1)
        evidence = prepare_verification(directory/"verification", reuse_path=verification_report)
        paths = [Path(run_root)/task/f"seed_{run_seed}"/"checkpoints"/f"update_{checkpoint_update:08d}.pt"
                 for task, run_seed in CASES]
        # Check all four committed pairs before spending GPU inference time.
        for path in paths:
            if not path.is_file() or not path.with_suffix(".sha256").is_file():
                raise FileNotFoundError(f"missing common-update checkpoint/receipt: {path}")
        for (task, run_seed), path in zip(CASES, paths):
            current_case = {"task_family": task, "run_seed": run_seed, "checkpoint_path": str(path)}
            case_dir = _fresh(directory/task/f"seed_{run_seed}")
            report = _evaluate_one(path, case_dir, verification=evidence, data_root=data_root,
                statistics_root=statistics_root, config_root=config_root, device=device, windows=windows,
                seed=seed, precision=precision)
            cp = report["checkpoint"]
            if (cp["task_family"], cp["run_seed"], cp["update"]) != (task, run_seed, checkpoint_update):
                raise ValueError("campaign checkpoint metadata does not match its explicit task/seed/update path")
            reports.append({"report_path": str((case_dir/"report.json").resolve()),
                            "report_sha256": sha256_file(case_dir/"report.json"),
                            "checkpoint": cp, "pilot_readiness": report["pilot_readiness"], "timing": report["timing"]})
        ready = all(row["pilot_readiness"]["status"] == "ready_for_exploratory_phase3" for row in reports)
        result = {"schema_version": VERSION, "P2.5_quality_acceptance": False,
            "pilot_readiness": {"status": "ready_for_exploratory_phase3" if ready else "needs_review",
                "reasons": [f"{row['checkpoint']['task_family']}/seed_{row['checkpoint']['run_seed']}: {reason}"
                            for row in reports for reason in row["pilot_readiness"]["reasons"]],
                "unchecked_classes": sorted({item for row in reports for item in row["pilot_readiness"]["unchecked_classes"]}),
                "scope": "four shared initial models; exploratory integration pilot only", "automatic_phase3_launch": False},
            "cases": reports, "checkpoint_update": checkpoint_update, "windows_per_model_cap": windows,
            "seed": seed, "verification": evidence, "source_sha256": evaluator_source_identity(),
            "new_optimizer_updates": 0, "new_real_interactions": 0, "total_seconds": time.perf_counter()-start}
        # Separate pilot selection evidence does not edit P2.4's final bindings.
        selection = write_json(directory/"pilot_model_selection.json", {"schema_version": VERSION,
            "P2.5_quality_acceptance": False, "pilot_readiness": result["pilot_readiness"],
            "campaign_identity_sha256": canonical_hash(result),
            "models": [{"checkpoint": row["checkpoint"], "evaluation_report_path": row["report_path"],
                        "evaluation_report_sha256": row["report_sha256"]} for row in reports]})
        result["pilot_model_selection"] = selection
        # The final report is the campaign commit marker, published only after
        # all four reports and the selection receipt have been committed.
        write_json(directory/"report.json", result)
        return result
    except Exception as error:
        _failure(directory, error, cases=reports, checkpoint_update=checkpoint_update, failed_case=current_case)
        raise
    finally:
        torch.set_num_threads(threads)
