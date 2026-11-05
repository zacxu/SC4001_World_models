"""Additive P2.5 prior-rollout diagnostics with explicit prediction coverage.

Inputs are detached NumPy arrays (or array-like values), with one row per real
transition/destination state. No model, fitting data, RNG, or pass threshold is
consulted here. Missing predictions remain visible in coverage and confusion
matrices; available-only scores never stand in for full-window performance.
"""
from __future__ import annotations

from collections import Counter
import math

import numpy as np


OUTCOME_NAMES = ("continue", "success", "failure")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _vector(value, steps, name, *, boolean=False, integer=False, default=None):
    if value is None:
        if default is None:
            raise ValueError(f"{name} is required")
        value = np.full(steps, default, dtype=bool if boolean else np.int64 if integer else float)
    array = np.asarray(value)
    if array.ndim != 1 or len(array) < steps:
        raise ValueError(f"{name} must have at least {steps} vector rows")
    array = array[:steps]
    if boolean and array.dtype != np.bool_:
        raise ValueError(f"{name} must be boolean")
    if integer and array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer labels")
    return array


def _prediction(value, steps, shape, name, *, fill=np.nan):
    array = np.asarray(value)
    if array.ndim != len(shape) + 1 or array.shape[1:] != shape:
        raise ValueError(f"{name} must have row shape {shape}")
    output = np.full((steps, *shape), fill, dtype=np.float64)
    output[:min(steps, len(array))] = array[:steps]
    return output


def _finite(value, name):
    if not np.isfinite(value).all():
        raise ValueError(f"valid {name} must be finite")


def _probabilities(value, name):
    _finite(value, name)
    if np.any((value < 0) | (value > 1)) or not np.allclose(value.sum(-1), 1., atol=1e-5, rtol=0):
        raise ValueError(f"{name} must be probability distributions")


def _binary_probabilities(value, name):
    _finite(value, name)
    if np.any((value < 0) | (value > 1)):
        raise ValueError(f"{name} must lie in [0,1]")


def _baseline_vector(value, steps, name):
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 0:
        array = np.full(steps, float(array))
    return _vector(array, steps, name).astype(np.float64)


def _baseline_distribution(value, steps, classes, name):
    array = np.asarray(value, dtype=np.float64)
    if array.shape == (classes,):
        array = np.broadcast_to(array, (steps, classes))
    if array.ndim != 2 or array.shape[0] < steps or array.shape[1] != classes:
        raise ValueError(f"{name} must have shape [{classes}] or [T,{classes}]")
    array = array[:steps]
    _probabilities(array, name)
    return array


def _images(value, name):
    array = np.asarray(value)
    if array.dtype == np.uint8:
        return array.astype(np.float64) / 255.
    if array.dtype.kind != "f":
        raise ValueError(f"{name} must be uint8 or floating [0,1]")
    return array.astype(np.float64)


def _image_range(value, name):
    _finite(value, name)
    if np.any((value < 0) | (value > 1)):
        raise ValueError(f"valid {name} must lie in [0,1]")


def _classification(matrix, names, unique_events=None):
    """Last prediction column means MISSING, rather than a fabricated class."""
    support = int(matrix.sum())
    classes = matrix.shape[0]
    available = int(matrix[:, :classes].sum())
    correct = int(np.trace(matrix[:, :classes]))
    per_class = {}
    for index, name in enumerate(names):
        positive = int(matrix[index].sum())
        tp = int(matrix[index, index])
        fp = int(matrix[:, index].sum()) - tp
        fn = positive - tp
        per_class[name] = {"support": positive, "true_positive": tp, "false_positive": fp,
                           "false_negative": fn, "precision": tp/(tp+fp) if tp+fp else None,
                           "recall": tp/positive if positive else None,
                           # No observed positive class cannot establish F1.
                           "f1": 2*tp/(2*tp+fp+fn) if positive else None}
        if unique_events is not None:
            per_class[name]["unique_positive_events"] = unique_events.get(name, 0)
    return {"support": support, "available_predictions": available, "missing_predictions": support-available,
            "class_names": list(names), "prediction_columns": [*names, "MISSING"],
            "class_support": matrix.sum(axis=1).astype(int).tolist(),
            "confusion_matrix": matrix.astype(int).tolist(), "accuracy": correct/support if support else None,
            "accuracy_available": correct/available if available else None, "per_class": per_class}


class MetricsAccumulator:
    """Aggregate windows within ONE horizon and ONE externally chosen stratum.

    ``targets``: rgb[T,2,3,H,W], image_valid[T,2], reward[T], outcome[T],
    outcome_valid[T], terminated[T], milestones{name:{value[T],valid[T]}}.
    Optional: reward_valid[T], termination_valid[T], deadline[T], stage[T],
    stage_valid[T], transition_index[T] (or decision_index[T]).

    ``predictions``: active[T] boolean prefix; rgb/reward/outcome_probabilities
    [T,3]/terminated; milestones{name:probability[T]}; optional
    termination_probability[T] and stage_probabilities[T,C] (or stage[T]).
    Prediction arrays may be shorter than the true suffix: omitted rows are
    missing. Nonfinite padding is ignored only where active is false.

    ``baseline``: fitting-derived reward scalar or vector; outcome_probabilities
    [3] or [T,3]; milestones{name:frequency}; optional termination_probability.
    ``anchor_rgb`` supplies the persistence predictor. Baselines must be fitted
    elsewhere using fitting data only; this module never estimates them.
    """

    def __init__(self):
        self._sums = {}
        self._counts = Counter()
        self._outcome = np.zeros((3, 4), dtype=np.int64)
        self._outcome_baseline = np.zeros((3, 4), dtype=np.int64)
        self._termination = np.zeros((2, 3), dtype=np.int64)
        self._termination_baseline = np.zeros((2, 3), dtype=np.int64)
        self._milestones, self._milestone_baselines = {}, {}
        self._events = {}
        self._stage = None
        self._episodes, self._windows = set(), set()
        self._horizon = None
        self._ids_complete = True

    def _add(self, name, values):
        values = np.asarray(values, dtype=np.float64)
        _finite(values, name)
        total = float(values.sum())
        if not math.isfinite(total):
            raise ValueError(f"metric sum overflows: {name}")
        row = self._sums.setdefault(name, [0., 0])
        row[0] += total
        row[1] += int(values.size)

    def _record_event(self, name, episode_id, indices):
        events = self._events.setdefault(name, set())
        events.update((episode_id, int(index)) for index in indices)

    def add_window(self, targets: dict, predictions: dict, *, anchor_rgb, baseline: dict,
                   episode_id: str, horizon: int = 15) -> None:
        """Validate and score transactionally; malformed input adds no totals."""
        scored = type(self)()
        scored._score(targets, predictions, anchor_rgb=anchor_rgb, baseline=baseline,
                      episode_id=episode_id, horizon=horizon)
        if self._horizon is not None and self._horizon != scored._horizon:
            raise ValueError("use separate metric accumulators for different horizons")
        if self._stage is not None and scored._stage is not None and self._stage.shape != scored._stage.shape:
            raise ValueError("stage class counts differ between windows")
        for name, (total, count) in scored._sums.items():
            if not math.isfinite(self._sums.get(name, (0., 0))[0] + total):
                raise ValueError("metric accumulation overflow")
        self._horizon = scored._horizon
        self._ids_complete = self._ids_complete and scored._ids_complete
        for name, (total, count) in scored._sums.items():
            row = self._sums.setdefault(name, [0., 0])
            row[0] += total
            row[1] += count
        self._counts.update(scored._counts)
        for name in ("_outcome", "_outcome_baseline", "_termination", "_termination_baseline"):
            getattr(self, name)[:] += getattr(scored, name)
        for name in ("_milestones", "_milestone_baselines"):
            target = getattr(self, name)
            for field, matrix in getattr(scored, name).items():
                target.setdefault(field, np.zeros((2, 3), dtype=np.int64))[:] += matrix
        if scored._stage is not None:
            if self._stage is None:
                self._stage = np.zeros_like(scored._stage)
            self._stage += scored._stage
        for name, events in scored._events.items():
            self._events.setdefault(name, set()).update(events)
        self._episodes.update(scored._episodes)
        self._windows.update(scored._windows)

    def _score(self, targets, predictions, *, anchor_rgb, baseline, episode_id, horizon):
        _integer(horizon, "horizon", 1)
        if not isinstance(episode_id, str) or not episode_id:
            raise ValueError("a stable nonempty episode ID is required")
        reward = np.asarray(targets["reward"])
        if reward.ndim != 1 or not len(reward):
            raise ValueError("real suffix must contain at least one transition")
        steps = min(horizon, len(reward))
        self._horizon = horizon
        encoded_index = targets.get("transition_index", targets.get("decision_index"))
        self._ids_complete = encoded_index is not None
        index = (np.arange(steps, dtype=np.int64) if encoded_index is None else
                 _vector(encoded_index, steps, "absolute transition indices", integer=True))
        if np.any(index < 0) or np.any(np.diff(index) != 1):
            raise ValueError("absolute transition indices must be consecutive nonnegative integers")
        self._episodes.add(episode_id)
        self._windows.add((episode_id, int(index[0]), steps))
        self._counts.update(windows=1, target_steps=steps, full_horizon_windows=int(steps == horizon),
                            short_real_suffix_windows=int(steps < horizon))
        active_raw = np.asarray(predictions["active"])
        if active_raw.dtype != np.bool_ or active_raw.ndim != 1:
            raise ValueError("prediction active must be a boolean vector")
        active = np.zeros(steps, dtype=bool)
        active[:min(steps, len(active_raw))] = active_raw[:steps]
        if np.any(np.diff(active.astype(int)) > 0):
            raise ValueError("active predictions must be a contiguous prefix")
        self._counts.update(predicted_steps=int(active.sum()), missing_steps=int((~active).sum()))

        # Joint RGB is eligible only when both required real views are present.
        rgb = _images(targets["rgb"], "real RGB")
        if rgb.ndim != 5 or rgb.shape[0] < steps or rgb.shape[1:3] != (2, 3) or min(rgb.shape[-2:]) < 1:
            raise ValueError("real RGB must have shape [T,2,3,H,W]")
        rgb = rgb[:steps]
        image_valid = np.asarray(targets["image_valid"])
        if image_valid.dtype != np.bool_ or image_valid.ndim != 2 or image_valid.shape[0] < steps or image_valid.shape[1] != 2:
            raise ValueError("image validity must have shape [T,2] and boolean dtype")
        eligible = image_valid[:steps].all(-1)
        available = eligible & active
        image_shape = rgb.shape[1:]
        predicted_rgb = _prediction(_images(predictions["rgb"], "predicted RGB"), steps, image_shape, "predicted RGB")
        anchor = _images(anchor_rgb, "anchor RGB")
        if anchor.shape != image_shape:
            raise ValueError("anchor RGB shape differs from destination views")
        _image_range(anchor, "anchor RGB")
        _image_range(rgb[eligible], "real RGB")
        _image_range(predicted_rgb[available], "predicted RGB")
        self._add("rgb/model_mse", np.square(predicted_rgb[available]-rgb[available]))
        self._add("rgb/persistence_mse", np.square(anchor-rgb[eligible]))
        self._add("rgb/persistence_mse_available", np.square(anchor-rgb[available]))
        self._add("rgb/coverage_adjusted_mse", np.square(predicted_rgb[available]-rgb[available]))
        missing_pixels = int((eligible & ~active).sum()) * math.prod(image_shape)
        row = self._sums["rgb/coverage_adjusted_mse"]
        row[0] += missing_pixels  # Maximum [0,1] squared error, not a fabricated image.
        row[1] += missing_pixels
        self._counts.update(rgb_eligible_steps=int(eligible.sum()), rgb_available_steps=int(available.sum()),
                            rgb_missing_predictions=int((eligible & ~active).sum()))

        # After a model terminates, its future reward is absorbing zero. Its
        # absence remains separately recorded; future real rewards still count.
        reward = _vector(targets["reward"], steps, "real reward").astype(float)
        reward_valid = _vector(targets.get("reward_valid"), steps, "reward validity", boolean=True, default=True)
        predicted_reward = _prediction(predictions["reward"], steps, (), "predicted reward")
        baseline_reward = _baseline_vector(baseline["reward"], steps, "baseline reward")
        _finite(reward[reward_valid], "real reward")
        _finite(predicted_reward[active & reward_valid], "predicted reward")
        _finite(baseline_reward[reward_valid], "baseline reward")
        effective_reward = np.where(active, predicted_reward, 0.)
        self._add("reward/model_mae", np.abs(effective_reward[reward_valid]-reward[reward_valid]))
        self._add("reward/model_mae_available", np.abs(predicted_reward[active & reward_valid]-reward[active & reward_valid]))
        self._add("reward/baseline_mae", np.abs(baseline_reward[reward_valid]-reward[reward_valid]))
        self._counts["reward_missing_predictions"] += int((reward_valid & ~active).sum())
        if bool(reward_valid.all()):
            model_errors = np.cumsum(effective_reward-reward)
            baseline_errors = np.cumsum(baseline_reward-reward)
            self._add("reward/cumulative_mae", [abs(float(model_errors[-1]))])
            self._add("reward/baseline_cumulative_mae", [abs(float(baseline_errors[-1]))])
            self._add("reward/cumulative_path_mae", np.abs(model_errors))
            self._add("reward/baseline_cumulative_path_mae", np.abs(baseline_errors))
        else:
            self._counts["windows_with_incomplete_reward_labels"] += 1

        outcome = _vector(targets["outcome"], steps, "learned outcome", integer=True)
        outcome_valid = _vector(targets["outcome_valid"], steps, "outcome validity", boolean=True)
        if np.any((outcome[outcome_valid] < 0) | (outcome[outcome_valid] > 2)):
            raise ValueError("learned outcomes must be continue/success/nondeadline-failure")
        probabilities = _prediction(predictions["outcome_probabilities"], steps, (3,), "outcome probabilities")
        baseline_outcome = _baseline_distribution(baseline["outcome_probabilities"], steps, 3, "baseline outcome")
        _probabilities(probabilities[active & outcome_valid], "predicted outcome")
        predicted_class = np.full(steps, 3, dtype=int)
        predicted_class[active & outcome_valid] = probabilities[active & outcome_valid].argmax(-1)
        np.add.at(self._outcome, (outcome[outcome_valid], predicted_class[outcome_valid]), 1)
        np.add.at(self._outcome_baseline, (outcome[outcome_valid], baseline_outcome[outcome_valid].argmax(-1)), 1)
        available_outcomes = active & outcome_valid
        truth = np.eye(3)[outcome[outcome_valid]]
        raw_error = np.square(probabilities[available_outcomes]-np.eye(3)[outcome[available_outcomes]])
        self._add("outcome/brier", raw_error.sum(-1))
        self._add("outcome/coverage_adjusted_brier", raw_error.sum(-1))
        missing = int((outcome_valid & ~active).sum())
        self._sums["outcome/coverage_adjusted_brier"][0] += 2*missing
        self._sums["outcome/coverage_adjusted_brier"][1] += missing
        self._add("outcome/baseline_brier", np.square(baseline_outcome[outcome_valid]-truth).sum(-1))
        for category, name in enumerate(OUTCOME_NAMES):
            binary = outcome == category
            self._add(f"outcome/{name}/brier", np.square(probabilities[available_outcomes, category]-binary[available_outcomes]))
            self._add(f"outcome/{name}/baseline_brier", np.square(baseline_outcome[outcome_valid, category]-binary[outcome_valid]))
            self._record_event(f"outcome/{name}", episode_id, index[outcome_valid & binary])

        terminal = _vector(targets["terminated"], steps, "real termination", boolean=True)
        terminal_valid = _vector(targets.get("termination_valid"), steps, "termination validity", boolean=True, default=True)
        deadline = _vector(targets.get("deadline"), steps, "deadline", boolean=True, default=False)
        raw_terminal = np.asarray(predictions["terminated"])
        if raw_terminal.dtype != np.bool_ or raw_terminal.ndim != 1:
            raise ValueError("predicted termination must be boolean")
        predicted_terminal = np.zeros(steps, dtype=bool)
        predicted_terminal[:min(steps, len(raw_terminal))] = raw_terminal[:steps]
        if np.any(predicted_terminal & ~active):
            raise ValueError("inactive model rows cannot introduce new termination events")
        ended = np.cumsum(predicted_terminal.astype(int))
        if np.any(active & (np.concatenate(([0], ended[:-1])) > 0)):
            raise ValueError("model predictions continued after their sampled terminal")
        term_probability = (_prediction(predictions["termination_probability"], steps, (), "termination probability")
                            if "termination_probability" in predictions else probabilities[:, 1:].sum(-1))
        if "termination_probability" not in predictions:
            term_probability = np.where(deadline & active, 1., term_probability)
        base_terminal = (_baseline_vector(baseline["termination_probability"], steps, "baseline termination")
                         if "termination_probability" in baseline else np.where(deadline, 1., baseline_outcome[:, 1:].sum(-1)))
        _binary_probabilities(term_probability[active & terminal_valid], "termination probability")
        _binary_probabilities(base_terminal[terminal_valid], "baseline termination probability")
        term_class = np.where(active, predicted_terminal.astype(int), 2)
        np.add.at(self._termination, (terminal[terminal_valid].astype(int), term_class[terminal_valid]), 1)
        np.add.at(self._termination_baseline, (terminal[terminal_valid].astype(int),
                                             (base_terminal[terminal_valid] >= .5).astype(int)), 1)
        self._add("termination/brier", np.square(term_probability[active & terminal_valid]-terminal[active & terminal_valid]))
        self._add("termination/coverage_adjusted_brier", np.square(term_probability[active & terminal_valid]-terminal[active & terminal_valid]))
        missing_terminal = int((terminal_valid & ~active).sum())
        self._sums["termination/coverage_adjusted_brier"][0] += missing_terminal
        self._sums["termination/coverage_adjusted_brier"][1] += missing_terminal
        self._add("termination/baseline_brier", np.square(base_terminal[terminal_valid]-terminal[terminal_valid]))
        early = active & predicted_terminal & ~terminal & terminal_valid
        self._counts.update(false_early_terminations=int(early.sum()), false_early_windows=int(early.any()),
                            missed_terminals=int((terminal & terminal_valid & (~active | ~predicted_terminal)).sum()),
                            deadline_terminals=int((terminal & terminal_valid & deadline).sum()),
                            nondeadline_terminals=int((terminal & terminal_valid & ~deadline).sum()))
        self._record_event("termination", episode_id, index[terminal_valid & terminal])

        target_events = targets.get("milestones", {})
        prediction_events, baseline_events = predictions.get("milestones", {}), baseline.get("milestones", {})
        if any(not isinstance(name, str) or not name for name in target_events):
            raise ValueError("milestone names must be nonempty strings")
        if set(target_events) != set(prediction_events) or set(target_events) != set(baseline_events):
            raise ValueError("model, target and fitting-baseline milestone names differ")
        for name, record in target_events.items():
            values = _vector(record["value"], steps, f"{name} target")
            valid = _vector(record["valid"], steps, f"{name} validity", boolean=True)
            if not np.isin(values[valid], [0, 1]).all():
                raise ValueError("valid milestone targets must be zero or one")
            predicted = _prediction(prediction_events[name], steps, (), f"{name} probability")
            constant = _baseline_vector(baseline_events[name], steps, f"{name} baseline")
            _binary_probabilities(predicted[valid & active], f"{name} probability")
            _binary_probabilities(constant[valid], f"{name} baseline")
            matrix = np.zeros((2, 3), dtype=np.int64)
            base_matrix = np.zeros_like(matrix)
            predicted_label = np.where(active, predicted >= .5, 2).astype(int)
            np.add.at(matrix, (values[valid].astype(int), predicted_label[valid]), 1)
            np.add.at(base_matrix, (values[valid].astype(int), (constant[valid] >= .5).astype(int)), 1)
            self._milestones[name], self._milestone_baselines[name] = matrix, base_matrix
            self._add(f"milestone/{name}/brier", np.square(predicted[valid & active]-values[valid & active]))
            self._add(f"milestone/{name}/coverage_adjusted_brier", np.square(predicted[valid & active]-values[valid & active]))
            missing_event = int((valid & ~active).sum())
            self._sums[f"milestone/{name}/coverage_adjusted_brier"][0] += missing_event
            self._sums[f"milestone/{name}/coverage_adjusted_brier"][1] += missing_event
            self._add(f"milestone/{name}/baseline_brier", np.square(constant[valid]-values[valid]))
            self._record_event(f"milestone/{name}", episode_id, index[valid & (values == 1)])

        if "stage" in targets:
            stage = _vector(targets["stage"], steps, "stage", integer=True)
            valid = _vector(targets.get("stage_valid"), steps, "stage validity", boolean=True, default=True)
            if "stage_probabilities" in predictions:
                stage_values = np.asarray(predictions["stage_probabilities"])
                if stage_values.ndim != 2 or stage_values.shape[1] < 2:
                    raise ValueError("stage probabilities need a class dimension")
                classes = stage_values.shape[1]
                stage_values = _prediction(stage_values, steps, (classes,), "stage probabilities")
                _probabilities(stage_values[active & valid], "stage probabilities")
                selected_stage = np.zeros(steps, dtype=int)
                selected_stage[active & valid] = stage_values[active & valid].argmax(-1)
            else:
                raw = np.asarray(predictions.get("stage"))
                if raw.dtype.kind not in "iu" or raw.ndim != 1:
                    raise ValueError("stage predictions require integer classes or probabilities")
                selected_stage = np.zeros(steps, dtype=int)
                selected_stage[:min(steps, len(raw))] = raw[:steps]
                classes = max(2, int(stage[valid].max(initial=0))+1,
                              int(selected_stage[valid & active].max(initial=0))+1)
            if np.any((stage[valid] < 0) | (stage[valid] >= classes)):
                raise ValueError("valid stage labels lie outside predicted classes")
            if np.any((selected_stage[active & valid] < 0) | (selected_stage[active & valid] >= classes)):
                raise ValueError("active stage prediction lies outside its classes")
            predicted_stage = np.where(active, selected_stage, classes)
            self._stage = np.zeros((classes, classes+1), dtype=np.int64)
            np.add.at(self._stage, (stage[valid], predicted_stage[valid]), 1)

    def _term(self, name):
        total, count = self._sums.get(name, (0., 0))
        return {"sum": float(total), "count": int(count), "mean": total/count if count else None}

    def report(self) -> dict:
        """JSON-safe sufficient statistics, descriptive means and support only."""
        counts = self._counts
        outcomes = _classification(self._outcome, OUTCOME_NAMES,
            {name: len(self._events.get(f"outcome/{name}", ())) if self._ids_complete else None for name in OUTCOME_NAMES})
        outcomes.update(brier=self._term("outcome/brier"),
                        coverage_adjusted_brier=self._term("outcome/coverage_adjusted_brier"),
                        baseline_brier=self._term("outcome/baseline_brier"),
                        baseline=_classification(self._outcome_baseline, OUTCOME_NAMES))
        for name in OUTCOME_NAMES:
            outcomes["per_class"][name].update(brier=self._term(f"outcome/{name}/brier"),
                baseline_brier=self._term(f"outcome/{name}/baseline_brier"))
        milestones = {}
        for name, matrix in sorted(self._milestones.items()):
            summary = _classification(matrix, ("absent", "present"))
            positive = summary["per_class"]["present"]
            summary.update(positives=positive["support"], unique_positive_events=(len(self._events.get(f"milestone/{name}", ())) if self._ids_complete else None),
                           probability_threshold=.5,
                           precision=positive["precision"], recall=positive["recall"], f1=positive["f1"],
                           brier=self._term(f"milestone/{name}/brier"),
                           coverage_adjusted_brier=self._term(f"milestone/{name}/coverage_adjusted_brier"),
                           baseline_brier=self._term(f"milestone/{name}/baseline_brier"),
                           baseline=_classification(self._milestone_baselines[name], ("absent", "present")))
            milestones[name] = summary
        termination = _classification(self._termination, ("continue", "terminal"))
        termination.update(brier=self._term("termination/brier"),
            coverage_adjusted_brier=self._term("termination/coverage_adjusted_brier"),
            baseline_brier=self._term("termination/baseline_brier"),
            baseline=_classification(self._termination_baseline, ("continue", "terminal")),
            false_early_terminations=counts["false_early_terminations"], false_early_windows=counts["false_early_windows"],
            false_early_window_rate=counts["false_early_windows"]/counts["windows"] if counts["windows"] else None,
            missed_terminals=counts["missed_terminals"], deadline_terminals=counts["deadline_terminals"],
            nondeadline_terminals=counts["nondeadline_terminals"], unique_terminals=(len(self._events.get("termination", ())) if self._ids_complete else None))
        return {"schema_version": "phase25-rollout-metrics-v1", "horizon": self._horizon,
                "windows": counts["windows"], "unique_windows": len(self._windows) if self._ids_complete else None,
                "unique_episodes": len(self._episodes), "absolute_event_ids_available": self._ids_complete,
                "full_horizon_windows": counts["full_horizon_windows"],
                "short_real_suffix_windows": counts["short_real_suffix_windows"],
                "coverage": {"target_steps": counts["target_steps"], "predicted_steps": counts["predicted_steps"],
                             "missing_steps": counts["missing_steps"],
                             "fraction": counts["predicted_steps"]/counts["target_steps"] if counts["target_steps"] else None},
                "rgb": {name: self._term(f"rgb/{name}") for name in
                        ("model_mse", "persistence_mse", "persistence_mse_available", "coverage_adjusted_mse")}
                       | {"eligible_steps": counts["rgb_eligible_steps"], "available_steps": counts["rgb_available_steps"],
                          "missing_predictions": counts["rgb_missing_predictions"]},
                "reward": {name: self._term(f"reward/{name}") for name in
                           ("model_mae", "model_mae_available", "baseline_mae", "cumulative_mae",
                            "baseline_cumulative_mae", "cumulative_path_mae", "baseline_cumulative_path_mae")}
                          | {"missing_predictions": counts["reward_missing_predictions"],
                             "windows_with_incomplete_labels": counts["windows_with_incomplete_reward_labels"],
                             "missing_prediction_rule": "absorbing zero reward; real future rewards remain scored"},
                "outcome": outcomes, "termination": termination, "milestones": milestones,
                "stage": (_classification(self._stage, tuple(str(i) for i in range(self._stage.shape[0])))
                          if self._stage is not None else {"support": 0, "accuracy": None, "confusion_matrix": []}),
                "missing_prediction_penalties": {"rgb_squared_error": 1., "categorical_brier": 2., "binary_brier": 1.},
                "quality_acceptance": None}


def score_window(targets: dict, predictions: dict, *, anchor_rgb, baseline: dict,
                 episode_id: str, horizon: int = 15) -> dict:
    """Convenience wrapper for one window; no statistical pass is implied."""
    accumulator = MetricsAccumulator()
    accumulator.add_window(targets, predictions, anchor_rgb=anchor_rgb, baseline=baseline,
                           episode_id=episode_id, horizon=horizon)
    return accumulator.report()
