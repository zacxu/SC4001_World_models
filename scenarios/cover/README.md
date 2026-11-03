# Cover-and-activate scenario

`task.acs` implements two exposed USE-zone consoles, each requiring 20 qualifying
joint decisions at the configured four-tic repeat. Only a living player in the
zone holding USE advances progress; interruption pauses it. At most one dwell
increment occurs per decision. Either player may operate either or both consoles;
role swapping is a possible strategy. Both living players then reach extraction.

One ordinary moving ZombieMan spawns initially with stable TID 200. A second
spawns once at the first console's completion with stable TID 201. The first
survives that milestone unless killed through ordinary engine combat. Original
pillar geometry provides cover. Finite enemies may be cleared before operating
consoles; simultaneous suppression is not a success requirement. Players start
with a bounded 200 pistol rounds and green armor; health/ammo remain visible.
This is an initial mechanics setting, with difficulty to be frozen after pilots.

Extraction has its own checkerboard floor and wall-mounted EXIT signs. They
change from amber LOCKED to green OPEN when both consoles complete. The visible
floor footprint and success rectangle come from the same manifest definition;
success uses both living player centers within the marked half-open rectangle.

The controller probes progress pause with three no-USE decisions and probes
irreversible completion by repeating USE after the first milestone. It records
ordinary-action completion, rewards, health, clock alignment, and actual frames:

```powershell
.\.venv\Scripts\python.exe scenarios\diagnostic_controller.py cover --seed 0
```

These privileged scripted diagnostics validate mechanics and stay separate from
learning data unless explicitly counted by the future collection pipeline.
