"""CLI for the privileged mechanics diagnostic, not a learned policy."""

import argparse
import json
from world_models.env.scripted import run_scripted_task


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task",choices=("relay","cover"))
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--inactive-player",type=int,choices=(0,1))
    args=parser.parse_args()
    print(json.dumps(run_scripted_task(args.task,args.seed,inactive_player=args.inactive_player),indent=2))
