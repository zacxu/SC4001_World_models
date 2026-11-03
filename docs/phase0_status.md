# Phase 0 completion record

Phase 0 implements the research contract and project-wide conventions in section
0 of the detailed research implementation plan.

| Requirement | Implementation/evidence |
|---|---|
| Research question, hypothesis, claim boundaries | `docs/research_contract.md` |
| Local-only deployment / training information rules | `contracts.py`, `docs/interface_contract.md`, configuration guards |
| Shared defaults and B/D parity | Strict typed YAML composition and paired validator |
| Two task families and pilot/final run definitions | Task/experiment profiles and expanded run matrices |
| Timing, clock, actions, reward, bootstrap contracts | Dependency-free contracts and meaningful boundary tests |
| Paired initialization and independent later streams | Stable seed derivation and collision-checked matrices |
| Repository/package structure | `src/world_models/`, scripts, configs, scenarios, tests, artifacts, docs |
| Tested, pinned foundation tooling | `requirements-phase0.lock` and installed-version checks |
| Observed hardware/software inventory | Generated JSON; requested resources shown separately |
| Executable verification | `python -m scripts.phase0 check` and its acceptance JSON |

The current pass/fail result, exact tests, interpreter, inventory, and source
hashes are recorded in `artifacts/phase0_acceptance.json`. That report is generated
from execution rather than a hand-written assertion of completion.

## Phase 1 handoff

Per section 0.4, the training-runtime lock follows the two-player smoke test on
the actual training host. That runtime validation, four-H200 allocation check,
scenario compiler/assets, multiplayer synchronization, rendered countdown, and
task mechanics are Phase 1 work. Phase 0's acceptance status does not claim those
have passed. All training-run matrix entries are `not_started`.
