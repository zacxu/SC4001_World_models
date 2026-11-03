# Foundation runbook

Run commands from the project root after the README's environment setup. On
Windows use `.\.venv\Scripts\python.exe`; on Linux replace that executable with
`.venv/bin/python`.

```powershell
.\.venv\Scripts\python.exe -m unittest tests.test_config tests.test_contracts tests.test_inventory tests.test_phase0_cli tests.test_reproducibility -v
.\.venv\Scripts\python.exe -m scripts.phase0 validate --experiment final --output artifacts/phase0/final_configs.json
.\.venv\Scripts\python.exe -m scripts.phase0 validate --experiment pilot --output artifacts/phase0/pilot_configs.json
.\.venv\Scripts\python.exe -m scripts.phase0 matrix --experiment final --output artifacts/phase0/final_run_matrix.json
.\.venv\Scripts\python.exe -m scripts.phase0 matrix --experiment pilot --output artifacts/phase0/pilot_run_matrix.json
.\.venv\Scripts\python.exe -m scripts.phase0 inventory --output artifacts/phase0/inventory.json
.\.venv\Scripts\python.exe -m scripts.phase0 check --output artifacts/phase0_acceptance.json
```

`check` validates every B/D/task/profile combination, checks common-setting
parity, expands and checks RNG namespaces, runs the unit tests, verifies installed
bootstrap pins, records measured host inventory, and hashes source/config/docs.
It exits nonzero on failure. Regenerate this report after changing Phase 0 code,
configs, tests, or contracts so its recorded hashes describe current files.
It selects the Phase 0 tests explicitly, so the foundation check does not require
the optional simulator/image dependencies introduced in Phase 1. The Phase 1
verification command runs the complete installed test suite.

Configuration composition is `base.yaml` plus one task, one variant, and one
experiment profile. Unknown keys, duplicate YAML keys, wrong types, invalid
checkpoint grids, and overlays that change another section fail validation.
Change common settings in the common file so B and D remain comparable. Numerical
pilot changes must be documented; substantive post-freeze changes require a new
protocol and consistent reruns.

The Phase 0 lock pins only the tooling actually exercised now. Phase 1 will choose
and pin Python/PyTorch/CUDA/ViZDoom/scenario-compiler versions after a two-player
smoke test on the actual training host. Do not infer that the promised four H200s
are present from this development machine's report. Record base assets, maps,
compiler version, and driver/runtime inventory in the Phase 1 manifest.

Phase 0 does not start a simulator or run training. The placeholder packages and
scenario directories mark the next implementation destinations without providing
fake simulator/training entry points.

P1.1–P1.4 now provide the actual simulator and scenario implementation. Use the
[step-specific runbook](phase1_steps1_4.md) for its runtime lock, compiler setup,
build/smoke/mechanics commands, and verification evidence. The remaining Phase 1
steps are not included in that report.
Use the [P1.5-P1.7 completion runbook](phase1_complete.md) for the full privileged
registry, geometry holdouts, boundary/mask interfaces, native replay and measured
acceptance campaign. The main current environment report is
`artifacts/phase1_complete/phase1_acceptance.json`; it does not begin model training.
