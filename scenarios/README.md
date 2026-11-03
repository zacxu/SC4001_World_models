# Cooperative scenario source and build provenance

`world_models.env.scenario_builder.build_scenario(task_family, seed, frame_skip)`
generates original UDMF geometry, compiles the corresponding editable ACS task
source, and packages a PWAD plus a JSON manifest. Output defaults to
`artifacts/phase1/scenarios/`. The manifest includes source, PWAD, and compiler
SHA-256 hashes, coordinates, role assignment, and the minimal mechanics flags.

Compiler: official [ZDoom ACC 1.60 release](https://github.com/ZDoom/acc/releases/tag/1.60).
On Windows extract `acc-1.60-win32.zip` into `.tools/acc-1.60/`; on Linux use the
matching official release. Alternatively set `WORLD_MODELS_ACC` to the executable.
The official Windows archive SHA-256 obtained during implementation is
`d70c633fcf07da705562ce65104379e1c9ecaeaa159981acc5275840e606a15f`.
The compiler is external tooling; retain the upstream release and associated
copyright/license provenance rather than claiming ownership of it.

Base textures/actors use the separately installed Freedoom 2 IWAD bundled with
ViZDoom. The generated PWAD contains original geometry/scripts and four original
PNG extraction textures, with no copied base-game sprites or textures. The
deterministic pixel art is generated in `scenario_builder.py`; individual PNGs
and their SHA-256 hashes are retained alongside each WAD. Preserve the bundled Freedoom copyright/license
when redistributing an IWAD. See https://freedoom.github.io/about.html.

In scenario version `coop-mechanics-v2`, the manifest's single extraction
rectangle defines its complete grid-aligned checkerboard floor, eastern-wall
EXIT signs, and success boundary. Both living player centers must be within its
half-open bounds (`x1 <= x < x2`, `y1 <= y < y2`). The amber LOCKED floor/sign
turn green OPEN after the relay latch or both cover consoles complete. The
same art and status rule apply across all generated development layouts. These
are physical first-person rendering cues, supplemented by the normal HUD.

USER13-20 expose each of the two possible cover enemies' integer x/y, nonnegative
health, and alive flag for the privileged diagnostic controller only; relay
initializes them to zero. Stable TIDs are 200 and 201. USER12 is their summed
nonnegative health. This small diagnostic extension does not complete the
future P1.5 auxiliary-target registry or add privileged inputs to local actors.

Only task-mechanic state needed by P1.3/P1.4 is exported now. The broader privileged
auxiliary registry, formal dataset splits, and P1.5–P1.7 acceptance campaign are
later work. Actual scripted completions must be recorded through the coordinator.
