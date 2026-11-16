"""Deterministic manuscript tables from authenticated saved records."""
from __future__ import annotations


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(map(str, row)) + " |" for row in rows)])


def replace_block(text, name, content):
    start, end = f"<!-- P63:{name}:BEGIN -->", f"<!-- P63:{name}:END -->"
    if text.count(start) != 1 or text.count(end) != 1 or text.index(start) >= text.index(end):
        raise ValueError(f"missing, reversed or duplicate manuscript marker: {name}")
    before, rest = text.split(start)
    _, after = rest.split(end)
    return before + start + "\n\n" + content + "\n\n" + end + after


def blocks(metrics, secondary, resources, evaluations):
    native = {(e["master_seed"], e["variant"]): e for e in secondary["per_seed"]}
    endpoint = [[r["master_seed"], r["variant"], f"{r['successes']}/{r['episodes']}",
        f"{r['mean_return']:.2f}", r["latch_episodes"], f"{r['mean_length']:.0f}",
        native[(r["master_seed"], r["variant"])]["termination_counts"]["deadline"]]
        for r in metrics["arms"]]
    paired = [[r["master_seed"], f"{r['success_rate_difference_D_minus_B']:.2f}",
        f"{r['mean_return_difference_D_minus_B']:.2f}",
        f"{r['latch_episode_rate_difference_D_minus_B']:.2f}"] for r in metrics["paired_differences"]]
    distillation = []
    for r in secondary["training_diagnostics"]:
        if r["variant"] != "D":
            continue
        d, teacher = r["distillation"], r["teacher_PPO"]
        distillation.append([r["master_seed"], d["trajectories"], d["valid_player_decisions"],
            f"{d['diagnostics_before']['forward_kl']:.4f}", f"{d['diagnostics_after']['forward_kl']:.4f}",
            f"{teacher['metrics']['mean_imagined_return']:.4f}"])
    resource = []
    for r in resources["per_run"]:
        c = r["logical_counters"]
        resource.append([r["master_seed"], r["variant"], f"{c['imagined_joint_transitions_local']:,}",
            f"{c['imagined_joint_transitions_teacher']:,}", f"{c['imagined_joint_transitions_buffer_generation']:,}",
            f"{c['imagined_joint_transitions_total']:,}", f"{r['recorded_scientific_unit_elapsed_seconds']/3600:.3f}"])
    traces = []
    for r in evaluations:
        # Match the registered first-failure rule; do not choose a visually interesting episode.
        episode = next((e for e in r["episodes"] if not e["success"]), None)
        if episode is not None:
            traces.append([r["master_seed"], r["variant"], episode["episode_index"],
                episode["template_id"], episode["event_counts"]["gate_opens"],
                episode["event_counts"]["latch_activations"], episode["termination_cause"]])
    return {
        "ENDPOINT_TABLE": table(["Seed", "Arm", "Successes", "Mean return", "Latch episodes", "Mean decisions", "Deadlines"], endpoint),
        "PAIRED_TABLE": table(["Seed", "Success difference", "Return difference", "Latch-rate difference"], paired),
        "DISTILLATION_TABLE": table(["Seed", "Buffer trajectories", "Valid player decisions", "KL before", "KL after", "Teacher imagined return"], distillation),
        "RESOURCE_TABLE": table(["Seed", "Arm", "Local steps", "Teacher steps", "Buffer steps", "Total imagined steps", "Recorded unit hours"], resource),
        "TRACE_TABLE": table(["Seed", "Arm", "Episode", "Layout", "Gate openings", "Latch activations", "End reason"], traces),
    }
