# Requested P1.1-P1.4 fixes: implementation and verification

The final [acceptance report](acceptance.json) passed with **110 tests**, all
native checks below, unchanged source/runtime hashes across the run, and the
same explicitly pinned Freedoom2 IWAD in every participant and case.

## Implemented

- One extraction rectangle generates both the marked floor and exact half-open
  player-center success bounds. Original amber LOCKED/padlock artwork changes
  to green OPEN/check artwork after unlocking. These are physical floors and
  wall signs, visible in the actual resized actor RGB.
- Operators start USE on their own readiness. Guard arrival and enemy death do
  not gate console progress or extraction. Fire/no-fire diagnostics share
  navigation, turning, USE, stage rules, maps, seeds and initial states.
- Either player can operate either or both consoles. Enemies move normally,
  have distinct stable IDs 200/201, and survive console completion unless killed
  by players. Clearing finite enemies before activation remains legitimate.
- Explicit wrapper lifecycle and atomic transition publication. Post-dispatch
  integrity failures raise typed durable errors and require fresh recovery;
  retries consume nothing, cumulative accounting survives, and both recurrent
  memories must reset.
- Explicit IWAD selection, SHA-256 pinning, and engine path readback before and
  after initialization. UDP preflight plus cross-process port leases and bounded
  collision retry protect simultaneous environment startup.

The extraction rectangles were aligned to the existing 64-unit floor grid so
there is no invisible inset. This expands relay extraction and partly shifts and
expands cover extraction; the old/new bounds are explicitly documented in
[the implementation notes](../../docs/phase1_steps1_4.md). The cover task retains
two stages, 20 USE decisions per console, a 600-decision deadline and its reward
schedule. The research comparison remains **B versus D only**.

## Verification

| Check | Result |
| --- | --- |
| Full unit suite | 110 passed |
| Phase 0 regression | Passed; [record](phase0_regression.json) |
| Native smoke and repeated resets | Passed for both task families |
| Normal scripted completion | Relay 3/3; cover 3/3, seeds 0/1/2 |
| Each-player-inactive checks | All four failed to achieve success and reached the deadline |
| Paired covering-fire diagnostics | Both conditions completed 3/3; no-fire used both consoles and attempted extraction with zero attacks |
| Same-player activation | Player 1 operated both consoles and extracted; 127 decisions |
| Combat-rule traces | Both enemies moved in every no-fire layout; first enemy survived the first console alongside the second enemy |
| Native injected conversion failure | Count 2 to 3; wrapper clock unchanged; blocked retries consumed zero |
| Fresh native recovery | New episode and both new engine instances; counts continued through 4 then 5 |
| Marker audits | All six passed; 63 designated camera views inspected, with 126 raw actor PNGs saved |
| Source/runtime/IWAD provenance | All consistency gates passed |
| Dependency check | No broken requirements |

The wrapper fault tests include regressing counters, disagreeing flags, malformed
RGB, unexpected engine completion, bad rewards/metadata, failed resets/recovery,
and zero-consumption retry blocking. Port tests use real UDP sockets and a second
process for lease exclusion, plus an injected backend bind collision for retry.

Independent audits: [runtime ledger](runtime_trace_audit.json),
[cover traces](cover_trace_audit.json), and
[marker images](marker_visual_review_scenarios.json).
The root's [visual review](marker_visual_review_root.json) records exact frame
and contact-sheet hashes. The [change inventory](requested_change_inventory.json)
compares all 19 changed/added source files against the archived baseline.
All six verified WADs, generated ACS/TEXTMAP sources and marker assets are also
preserved under [verified_scenarios](verified_scenarios/snapshot_index.json),
with hashes checked against the native cases before copying. This protects the
tested bytes if a later scenario build overwrites the canonical build directory.

## Covering-fire diagnostic result

These are descriptive results on three development layouts, not learned-policy
results or a statistical research conclusion.

| Mean per case | Fire enabled | Fire withheld |
| --- | ---: | ---: |
| Team success | 3/3 | 3/3 |
| Operator activation health plus armor loss | 0 | 4 |
| Total team health loss | 9.33 | 21.33 |
| Total team armor loss | 4.67 | 10.67 |
| Deaths | 0 | 0 |
| Completion decisions | 107.33 | 107.67 |
| Observed ammunition expenditure | 3.67 | 0 |

Only seed 2 showed activation-time operator damage reduction (12 to 0); seeds
0/1 had zero activation damage in both conditions. Completion-time effects were
mixed. The sample supports observed protection, with no demonstrated success
advantage. No task difficulty or forced cooperation rule was added to obtain a
positive effect.

Ammunition-derived shot counts are lower bounds on spending within four-tic
intervals because clip pickups can mask expenditure. One pickup interval was
observed and recorded separately. ATTACK-command counts are also retained.

## Limits and future work

Repeated same-seed mechanics runs showed a small trajectory difference despite
matching declared setup; exact seeded trajectory repeatability is therefore not
established. The first observed divergence, matching actions/tics/task flags,
and exact input-file hashes are recorded in the cover audit. Its cause is
unassigned. Matching paired initial states does not imply identical subsequent
trajectories. Characterize repeatability before confirmatory learning runs.

The visual review verifies sampled actor views and boundary consistency; it is
not a learned image-navigation evaluation. The development-host checks do not
validate CUDA/PyTorch training or the separate H200 allocation. P1.5-P1.7 remain
deferred, including the formal larger acceptance/throughput campaign.

The original acceptance evidence, matching sources, traces and WADs are archived
under `../phase1_steps1_4/before_requested_fixes/`.

## Reproduce

From the project root in Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase1 verify --seeds 0 1 2 --output-dir artifacts/phase1_requested_fixes_rerun
```

Native ViZDoom requires access to Windows message queues and local UDP sockets;
the verified run used the required unrestricted execution. Use a new output
directory to retain the reviewed evidence. Individual check commands are in the
implementation notes linked above.
