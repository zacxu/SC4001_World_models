# Foundation runbook

The active four-day fixed-data protocol is described in the
[detailed plan](../detailed_research_implementation_plan.md). Standalone
[teacher](phase41_teacher.md), [distillation](phase42_distillation.md) and
[local refinement](phase43_refinement.md) stages are implemented; their training
is operator-controlled. The [P4.4 integrated D guide](phase44_integration.md)
connects the stages and can register a completed P4.3 history without retraining.
The [P4.5 guide](phase45_acceptance.md) verifies on CPU during training and audits
the completed campaign afterward, without optimization or native evaluation.
The [P5.2 guide](phase52_training.md) supplies the matching reduced B controller,
fresh D adapter and paired candidate matrix. The
[P5.1 guide](phase51_protocol.md) freezes the static endpoint protocol and binds
its inputs on CPU. Its candidate registration does not make the comparison
execution-ready. The [complete Phase 5 guide](phase5_study.md) now supplies the
combined verification and separate prospective freeze, all four fresh learners,
ledger, final actor-only test evaluator and paired analysis. Start there after
the P5.2 candidate registration; no previous registration is rewritten.
After all four final evaluations, use the [server completion guide](phase5_completion.md)
for P5.6-P5.8. Its separate reporting package keeps the 296 frozen sources
unchanged and runs saved-record accounting, figures and acceptance on server CPU.
It neither reruns scientific episodes nor starts a learner.
After that gate passes, follow the [P6.1-P6.2 server guide](phase6_release.md)
for fixed native-trace diagnosis, a sealed release and portable six-table/two-figure
reproduction. These commands use CPU and saved records. Keep the new
`study_release` namespace separate from both prior source freezes; leave
P6.3's manuscript and claim acceptance for the subsequent step.
The [P3.5 guide](phase35_training.md) with `--study-profile relay_week_v1` retains the
earlier online development schedule. The Phase 0 commands below validate the
original foundation profiles; their pilot/final matrices are historical.
Preserve those configuration files for P2.4 provenance.

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
For new policy work, use the explicit `configs/studies/relay_week_v1.yaml`
amendment instead of editing the common files bound to saved P2.4 checkpoints.
It keeps common B/D settings matched. Substantive post-freeze changes require a
new protocol and consistent reruns.

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
