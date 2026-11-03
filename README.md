# Shared imagination for decentralized cooperation

The project compares direct local MAPPO learning in a shared Transformer world
model (**B**) with teacher learning, action distillation, and local refinement in
the same framework (**D**). Both deploy two actors using only their own visual
observations, previous actions, and separate recurrent memories.

Start with the [approved proposal](research_implementation_proposal.md),
[detailed plan](detailed_research_implementation_plan.md),
[research contract](docs/research_contract.md), and [runbook](docs/runbook.md).

Phase 0 implements configuration, contracts, seed derivation, inventory, and
verification; its record is [artifacts/phase0_acceptance.json](artifacts/phase0_acceptance.json).
Phase 1's complete environment interface and acceptance campaign are documented
in the [P1.5-P1.7 runbook](docs/phase1_complete.md), with a versioned
[target registry](configs/privileged_registry_v1.json),
[geometry splits](configs/environment_splits.yaml), and measured
[phase1_acceptance.json](artifacts/phase1_complete/phase1_acceptance.json).
The [readable results](artifacts/phase1_complete/README.md) include replay,
10,000-decision/100-reset checks, native mechanics videos and collector selection.

Phase 1's first step implements P1.1–P1.4: the live two-player coordinator, local
visual/action adapters, relay, and cover-and-activate mechanics. See the
[implementation and commands](docs/phase1_steps1_4.md) and its generated
[verification report](artifacts/phase1_requested_fixes/acceptance.json).
The revised mechanics include visible extraction cues, unrestricted console
roles, moving persistent enemies, paired fire/no-fire diagnostics, durable
wrapper recovery, and explicit IWAD/port handling. Earlier evidence is archived
under `artifacts/phase1_steps1_4/before_requested_fixes/`.

## Set up on Windows PowerShell

From this project directory:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase0.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m scripts.phase0 check --output artifacts/phase0_acceptance.json
```

## Set up on Linux

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-phase0.lock
.venv/bin/python -m pip install --no-build-isolation -e .
.venv/bin/python -m scripts.phase0 check --output artifacts/phase0_acceptance.json
```

For the implemented Phase 1 environment, use the tested Python 3.12 stack.
After creating the virtual environment, install its complete dependency lock and
scenario compiler:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-phase1.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m scripts.bootstrap_acc
```

See [the artifact policy and fresh-clone checks](artifacts/README.md) for which
reports are versioned and how to regenerate the excluded raw outputs.

**P2.1 is completed for the approved pilot scope:** relay and cover, seeds 0 and 1,
with four shared 50,000-transition stores. See the [measured results and acceptance](artifacts/phase21/README.md)
and [real replay runbook](docs/phase21_replay.md). World-model and policy training
remain later steps. Raw data stays outside Git; preserve a separate verified backup.

**P2.2 is completed:** the shared stochastic Transformer architecture, visual
encoder/decoder, supervised heads, and prior-only imagination interface are
implemented and verified. See the [architecture and API](docs/phase22_architecture.md)
and [verification report](artifacts/phase22/phase22_acceptance.json).
The model is untrained; P2.4 fitting and P2.5 model-quality acceptance remain pending.

**P2.3 is completed:** the reduced model objective, fitting-only normalization,
and loss/gradient/event diagnostics are implemented and verified. See the
[objectives and usage](docs/phase23_objectives.md) and
[verification report](artifacts/phase23/phase23_acceptance.json).
The four frozen pilot normalization files are shared by B/D within each
task/seed pair. Next is **P2.4: train, validate, and refresh the model**.

Edit shared defaults in `configs/base.yaml`. Task and experiment files own their
respective settings; variant overlays can only enable the teacher/distillation
procedure. Configuration validation rejects unrecognized fields, local-input
leaks, inconsistent budgets, and unequal common B/D settings.

The importable code uses the standard `src/world_models/` package layout; the
submodules follow the detailed plan's environment/data/model/policy/training/
evaluation/infrastructure separation. The tests use Python's built-in unittest.
