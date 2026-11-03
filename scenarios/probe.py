"""Small manual smoke probe; run with the project's installed environment."""

import json
import sys
from world_models.env.coop import TwoPlayerCoopEnv


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv)>1 else "relay"
    with TwoPlayerCoopEnv(task,seed=0,work_root="artifacts/phase1/scenario_probe") as env:
        result = env.reset()
        print(json.dumps(result.training_only_info,indent=2))
        for step in range(3):
            next_result = env.step(0,0)
            print(step,next_result.training_only_info["task_flags"])
