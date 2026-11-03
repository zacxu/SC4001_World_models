# Gate-and-relay scenario

`task.acs` implements authoritative plate, latch, one-time milestone, shared
extraction, deadline, and death mechanics. `scenario_builder.py` generates two
separated lanes, an upper plate-controlled gate, a lower latch-controlled gate,
and a shared extraction area. The role-to-spawn assignment, route length, pillar
placement, and wall texture vary deterministically by seed.

The plate-holder cannot reach the remote latch through its lane before the
runner releases it. The runner cannot pass its gate without the plate-holder.
The permanent latch lets the plate-holder leave; success requires both living
players at extraction. Visible floor markers and rendered persistent HUD cues
identify the mechanics. Reward is calculated once by the coordinator from the
authoritative milestone and outcome flags.

The shared extraction rectangle has a distinctive checkerboard floor and
wall-mounted EXIT signs. Both change from amber LOCKED to green OPEN when the
latch activates. This full marked floor footprint and the half-open player-center
success boundary are generated from the same coordinates, with a consistent
appearance across randomized layouts.

Build and run actual action-only diagnostic completion:

```powershell
.\.venv\Scripts\python.exe scenarios\diagnostic_controller.py relay --seed 0
```

This privileged diagnostic uses engine positions to choose ordinary actions.
Its results are task-mechanics evidence, not a learned-policy baseline.
