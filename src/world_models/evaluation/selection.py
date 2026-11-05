"""Lazy, bounded held-out cohorts for P2.5 predictive-quality diagnostics.

Uniform and event-enriched strata are reported separately. They are sampled
only from the frozen diagnostic episodes of real training-geometry stores;
no validation/test geometry is imported and no learning sampler is advanced.
Full real prefixes are loaded one window at a time.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import asdict, dataclass, field

import numpy as np

from world_models.data.replay import ReplayDataset, canonical_hash, sha256_file
from world_models.data.schema import REASONS
from world_models.models.heads import TaskSpec
from world_models.training.sampling import ReplaySampler, ReplayWindow


VERSION = "phase25-diagnostic-selection-v1"
_TRANSITION_COLUMNS = ("event_deltas", "event_delta_validity_mask", "team_reward",
                       "termination_reason", "terminated", "truncated", "decision_index")


def _integer(value, name, minimum=0, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def _transition_chunks(dataset, episode):
    """Read only small transition columns after replay's source/index checks."""
    for chunk in episode["chunks"]:
        path = (dataset.path / chunk["path"]).resolve()
        if not path.is_relative_to(dataset.path.resolve()):
            raise ValueError("replay chunk path escapes its source store")
        if sha256_file(path) != chunk["sha256"]:
            raise ValueError("diagnostic replay chunk bytes changed")
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in _TRANSITION_COLUMNS}
        if any(len(value) != chunk["transition_count"] for value in arrays.values()):
            raise ValueError("diagnostic transition columns differ from source row counts")
        yield chunk, arrays


def _event_id(identity, episode_id, transition):
    return canonical_hash({"dataset_identity_sha256": identity,
                           "episode_id": episode_id, "transition_index": transition})


def _overlaps(records):
    target_counts = {stratum: Counter() for stratum in ("uniform", "event")}
    anchors = {stratum: set() for stratum in ("uniform", "event")}
    events = {stratum: Counter() for stratum in ("uniform", "event")}
    for record in records:
        stratum = record["stratum"]
        key = record["dataset_index"], record["episode_id"]
        anchors[stratum].add((*key, record["anchor"]))
        target_counts[stratum].update((*key, index) for index in range(record["anchor"], record["stop"]))
        events[stratum].update(record["event_ids"])
    return {
        "identical_anchors_between_strata": len(anchors["uniform"] & anchors["event"]),
        "target_transitions_between_strata": len(target_counts["uniform"].keys() & target_counts["event"].keys()),
        "target_transitions_repeated_within_uniform": sum(count-1 for count in target_counts["uniform"].values()),
        "target_transitions_repeated_within_event": sum(count-1 for count in target_counts["event"].values()),
        "rare_event_instances_between_strata": len(events["uniform"].keys() & events["event"].keys()),
        "rare_event_instances_repeated_within_uniform": sum(count-1 for count in events["uniform"].values()),
        "rare_event_instances_repeated_within_event": sum(count-1 for count in events["event"].values()),
    }


@dataclass(frozen=True)
class DiagnosticCohort:
    """Immutable coordinate selection; each real-prefix window is loaded lazily."""

    _sampler: ReplaySampler = field(repr=False, compare=False)
    _records: tuple[dict, ...] = field(repr=False)
    _base_report: dict = field(repr=False, compare=False)

    @property
    def task_family(self) -> str:
        return self._sampler.task_family

    @property
    def run_seed(self) -> int:
        return self._sampler.run_seed

    @property
    def records(self) -> tuple[dict, ...]:
        # Returned report edits must not alter the selected coordinates.
        return deepcopy(self._records)

    @property
    def report(self) -> dict:
        report = deepcopy(self._base_report)
        counts = Counter(record["stratum"] for record in self._records)
        event_support = {}
        for stratum in ("uniform", "event"):
            selected = [record for record in self._records if record["stratum"] == stratum]
            unique = {event["event_id"]: event for record in selected for event in record["event_instances"]}
            event_support[stratum] = {
                "windows": counts[stratum], "unique_event_instances": len(unique),
                "event_kinds": dict(sorted(Counter(kind for event in unique.values() for kind in event["kinds"]).items())),
                "unique_episodes": len({(record["dataset_index"], record["episode_id"]) for record in selected}),
            }
        report.update(selected_windows=len(self), selected_strata={name: counts[name] for name in ("uniform", "event")},
                      selected_event_support=event_support, overlap_counts=_overlaps(self._records))
        report["cohort_identity"] = canonical_hash({"protocol_identity": report["protocol_identity"],
                                                    "records": self._records})
        return report

    @property
    def cohort_identity(self) -> str:
        return self.report["cohort_identity"]

    def __len__(self) -> int:
        return len(self._records)

    def prefix(self, count: int) -> "DiagnosticCohort":
        _integer(count, "prefix count", 1, 64)
        return DiagnosticCohort(self._sampler, self._records[:count], self._base_report)

    def window(self, index_or_record: int | Mapping) -> ReplayWindow:
        """Load one selected episode prefix; never pad or cross its boundary."""
        if type(index_or_record) is int:
            if not 0 <= index_or_record < len(self):
                raise IndexError("diagnostic cohort window index out of range")
            record = self._records[index_or_record]
        elif isinstance(index_or_record, Mapping):
            record = dict(index_or_record)
            if record not in self._records:
                raise ValueError("window record is outside the frozen diagnostic cohort")
        else:
            raise ValueError("select a cohort window by integer index or selected record")
        return self._sampler.window(record["dataset_index"], record["episode_id"], record["anchor"],
                                    burn_in=0, loss_steps=record["requested_horizon"], full_prefix=True)

    def iter_windows(self) -> Iterator[ReplayWindow]:
        for index in range(len(self)):
            yield self.window(index)


def select_diagnostic_cohort(datasets: Sequence[ReplayDataset], *, seed: int, max_windows: int = 64,
                             horizon: int = 15, uniform_limit: int = 32, event_limit: int = 32,
                             event_lead: int = 4, task: TaskSpec | None = None) -> DiagnosticCohort:
    """Select at most 32 uniform and 32 distinct event-enriched windows.

    The full bounded coordinate plan is deterministic and independent of
    ``max_windows``. Uniform/event records alternate, so a four/eight-window
    timing sample is an exact prefix of the larger evaluation cohort. Event
    windows begin up to four real transitions before a milestone/terminal;
    their rare-event target instances are disjoint within that stratum.
    Cross-stratum overlap is retained explicitly and reported separately.
    """
    _integer(seed, "seed")
    _integer(max_windows, "max_windows", 1, 64)
    _integer(horizon, "horizon", 1, 15)
    _integer(uniform_limit, "uniform_limit", 0, 32)
    _integer(event_limit, "event_limit", 0, 32)
    _integer(event_lead, "event_lead")
    if not uniform_limit and not event_limit:
        raise ValueError("select at least one diagnostic stratum")
    sampler = ReplaySampler(datasets, seed=seed, burn_in=0, loss_steps=horizon, partition="diagnostic")
    task = TaskSpec.for_family(sampler.task_family) if task is None else task
    if not isinstance(task, TaskSpec) or task.task_family != sampler.task_family:
        raise ValueError("selection task specification differs from replay family")
    event_fields = tuple(sampler.datasets[0].manifest["schema"]["event_fields"])
    milestone_names = (("latch_activations",) if sampler.task_family == "relay" else
                       ("console_1_completions", "console_2_completions"))
    milestone_columns = [(name, event_fields.index(name)) for name in milestone_names]
    events_by_episode = defaultdict(list)
    reason_support, episode_ends = Counter(), Counter()
    event_positive, event_valid = Counter(), Counter()
    sources = [{"dataset_index": index, "manifest_sha256": dataset.manifest_sha256,
                "dataset_identity_sha256": dataset.manifest["dataset_identity_sha256"]}
               for index, dataset in enumerate(sampler.datasets)]
    heldout_transitions = 0
    all_events = []
    for (dataset_index, episode_id), episode in sampler._episodes.items():
        dataset = sampler.datasets[dataset_index]
        identity = dataset.manifest["dataset_identity_sha256"]
        heldout_transitions += episode["valid_length"]
        episode_ends[episode["end_reason"]] += 1
        for chunk, arrays in _transition_chunks(dataset, episode):
            deltas, valid = arrays["event_deltas"], arrays["event_delta_validity_mask"]
            for column, name in enumerate(event_fields):
                event_valid[name] += int(valid[:, column].sum())
                event_positive[name] += int(np.count_nonzero(valid[:, column] & (deltas[:, column] > 0)))
            for local in range(chunk["transition_count"]):
                transition = chunk["transition_start"] + local
                code = int(arrays["termination_reason"][local])
                if not 0 <= code < len(REASONS):
                    raise ValueError("diagnostic transition has invalid boundary reason")
                reason = REASONS[code]
                reason_support[reason] += 1
                kinds = [f"milestone/{name}" for name, column in milestone_columns
                         if valid[local, column] and deltas[local, column] > 0]
                terminal_reward = task.success_reward if reason == "success" else (
                    task.failure_reward if reason in ("death", "failure", "deadline") else 0.)
                bonus = float(arrays["team_reward"][local])-task.step_cost-terminal_reward
                if bonus > 1e-6:
                    kinds.append("milestone_bonus")
                if bool(arrays["terminated"][local]):
                    kinds.append(f"terminal/{reason}")
                if kinds:
                    event = {"event_id": _event_id(identity, episode_id, transition),
                             "dataset_index": dataset_index, "episode_id": episode_id,
                             "transition_index": transition, "kinds": kinds}
                    events_by_episode[dataset_index, episode_id].append(event)
                    all_events.append(event)

    def record(dataset_index, episode_id, anchor, stratum):
        episode = sampler._episodes[dataset_index, episode_id]
        stop = min(episode["valid_length"], anchor+horizon)
        events = [deepcopy(event) for event in events_by_episode[dataset_index, episode_id]
                  if anchor <= event["transition_index"] < stop]
        return {"stratum": stratum, "dataset_index": dataset_index, "episode_id": episode_id,
                "dataset_identity_sha256": sources[dataset_index]["dataset_identity_sha256"],
                "start": 0, "anchor": anchor, "stop": stop, "requested_horizon": horizon,
                "available_horizon": stop-anchor, "event_ids": [event["event_id"] for event in events],
                "event_instances": events}

    rng = np.random.default_rng(seed)
    uniform = [record(*sampler._anchors[int(index)], "uniform")
               for index in rng.permutation(sampler.anchor_count)[:uniform_limit]]
    enriched, selected_anchors, selected_events = [], set(), set()
    rejected = Counter()
    for index in rng.permutation(len(all_events)):
        event = all_events[int(index)]
        key = event["dataset_index"], event["episode_id"]
        transition = event["transition_index"]
        valid_anchors = sampler._valid_anchors[key]
        eligible = valid_anchors[(valid_anchors <= transition) & (valid_anchors >= transition-horizon+1)]
        if not len(eligible):
            rejected["no_valid_source_anchor"] += 1
            continue
        desired = max(0, transition-min(event_lead, horizon-1))
        anchor = int(eligible[np.argmin(np.abs(eligible-desired))])
        if (*key, anchor) in selected_anchors:
            rejected["duplicate_anchor"] += 1
            continue
        proposal = record(*key, anchor, "event")
        if selected_events.intersection(proposal["event_ids"]):
            rejected["duplicate_rare_event_target"] += 1
            continue
        if len(enriched) >= event_limit:
            rejected["event_window_cap"] += 1
            continue
        enriched.append(proposal)
        selected_anchors.add((*key, anchor))
        selected_events.update(proposal["event_ids"])
    ordered = []
    for index in range(max(len(uniform), len(enriched))):
        if index < len(uniform):
            ordered.append(uniform[index])
        if index < len(enriched):
            ordered.append(enriched[index])
    protocol = {"version": VERSION, "seed": seed, "horizon": horizon, "uniform_limit": uniform_limit,
                "event_limit": event_limit, "event_lead": event_lead, "task": asdict(task),
                "sources": sources, "records": ordered}
    support_kinds = Counter(kind for event in all_events for kind in event["kinds"])
    report = {"version": VERSION, "task_family": sampler.task_family, "run_seed": sampler.run_seed,
              "geometry_split": "train", "partition": "diagnostic", "learning_data": False,
              "seed": seed, "horizon": horizon, "event_lead": event_lead, "task": asdict(task), "sources": sources,
              "source_cohort_identity": sampler.cohort_identity,
              "protocol_identity": canonical_hash(protocol), "planned_windows": len(ordered),
              "limits": {"uniform": uniform_limit, "event": event_limit, "total": 64},
              "ordering": "alternate uniform,event; deterministic nested prefixes",
              "event_target_rule": "each rare-event transition appears at most once within event stratum",
              "event_candidate_rejections": dict(sorted(rejected.items())),
              "full_heldout_support": {"episodes": len(sampler._episodes), "transitions": heldout_transitions,
                  "eligible_anchors": sampler.anchor_count, "rare_event_instances": len(all_events),
                  "rare_event_kinds": dict(sorted(support_kinds.items())),
                  "episode_end_reasons": {reason: episode_ends[reason] for reason in REASONS if reason != "continue"},
                  "transition_reasons": {reason: reason_support[reason] for reason in REASONS},
                  "events": {name: {"valid": event_valid[name], "positive": event_positive[name]} for name in event_fields}}}
    return DiagnosticCohort(sampler, tuple(ordered[:max_windows]), report)
