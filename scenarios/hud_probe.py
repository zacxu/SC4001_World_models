"""Capture actual local frames after native spawn fog for task-HUD visual QA."""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from world_models.env.coop import TwoPlayerCoopEnv


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("relay","cover"))
    args=parser.parse_args()
    output=Path("artifacts/phase1/hud_probe")
    output.mkdir(parents=True,exist_ok=True)
    env=TwoPlayerCoopEnv(args.task,seed=0,work_root=output/"environments")
    try:
        result=env.reset()
        for _ in range(12):
            result=env.step(0,0)
        paths=[]
        for player,frame in enumerate(result.observations):
            array=np.frombuffer(frame.data,dtype=np.uint8).reshape(3,96,128).transpose(1,2,0)
            path=output/f"{args.task}_step12_p{player}.png"
            Image.fromarray(array).save(path)
            paths.append(str(path.resolve()))
        report={"task_family":args.task,"task_flags":result.training_only_info["task_flags"],
                "image_paths":paths,"scenario_hash":env.scenario["hashes"]["wad_sha256"]}
        (output/f"{args.task}_report.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2))
    finally:
        env.close()
