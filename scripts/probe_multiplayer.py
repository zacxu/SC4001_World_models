"""Small real-engine P1.1 probe; the full step-specific report is separate."""

from pathlib import Path
import json

import vizdoom

from world_models.env.coordinator import CoordinatorConfig, TwoPlayerCoordinator
from world_models.env.observations import adapt_rgb, save_rgb


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output = root / "artifacts" / "phase1_steps1_4" / "multiplayer_probe"
    config = CoordinatorConfig(
        scenario_path=str(Path(vizdoom.__file__).parent / "freedoom2.wad"),
        work_root=str(output), seed=7, startup_timeout_seconds=25,
    )
    coordinator = TwoPlayerCoordinator(config)
    report = {"vizdoom": vizdoom.__version__, "steps": []}
    try:
        result = coordinator.start()
        report["initial_tics"] = [p.tic for p in result.participants]
        for index in range(10):
            result = coordinator.step((0, 0))
            report["steps"].append({"decision": result.decision_index, "tics": [p.tic for p in result.participants]})
        for participant in result.participants:
            save_rgb(adapt_rgb(participant.screen_buffer), output / f"player{participant.player_index}.png")
        reset = coordinator.reset(seed=8)
        report["reset_tics"] = [p.tic for p in reset.participants]
        report["status"] = "passed"
    except Exception as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        coordinator.close()
        output.mkdir(parents=True, exist_ok=True)
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report))


if __name__ == "__main__":
    main()
