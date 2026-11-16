# P2.4: use the last saved checkpoints

**The shortened exploratory pilot no longer requires 5,000 updates.** Use the
existing frozen selection after transferring the project to your GPU server.
Create a selection only if you have not already frozen the four models.
Counts may differ between task/seed cases; B and D must use the same selected
checkpoint within each case. Original targets and checkpoint contents stay intact.

The existing selection contains Relay 4,500/4,500 and Cover 5,000/5,000 updates.
Its latest P2.5 screen is complete; the active `relay_week_v1` study reuses the
two Relay cases. Preserve the entire four-case selection. The later reduction
to 250 model updates per online block is a separate Phase 3 study amendment,
not a change to these initial checkpoints.

Complete the [new GPU server setup](../linux_gpu_server_setup.md) first. Keep the
existing run folder names, even when they contain `aspire2a`; paths inside the
project must retain their relative layout.

## Windows PowerShell: freeze locally

After the training job has stopped and its latest files are synchronized, run
these commands from your Windows project **only if the selection does not exist**.
No GPU or environment activation is
needed when using the project's Python executable directly. Each Python command
below is **one line**; Bash's trailing `\` does not continue a PowerShell command.

```powershell
Set-Location "C:\Users\willi\Desktop\SC4001_World_models"
.\.venv\Scripts\python.exe -m scripts.phase24_freeze --run-root artifacts/phase24_runs/aspire2a_2gpu_v1 --output-dir artifacts/phase24_selection/shortened_v1 --reason "Time-limited exploratory pilot; use latest committed checkpoints"
```

After creation succeeds, verify the selection and display the saved counts:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase24_freeze --inspect artifacts/phase24_selection/shortened_v1/selection.json
```

`--inspect` reads an existing selection; it does not create one. If the selection
already exists, inspect it instead of rerunning creation. Transfer its folder
and the referenced source run, replay and normalization together to your GPU
server, keeping the same project-relative layout. Continue with P2.5 there.

The remaining commands use **Bash on your Linux GPU server**.

## 1. Transfer and inspect the existing selection

Stop any remaining old P2.4 training before the final synchronization. Use
WinSCP in binary mode to transfer the project into
`/home/zixuan/projects/WM1/`, including the selection and referenced run,
checkpoint/receipt pairs, replay and normalization. Recreate Python using the
setup guide; do not copy an old environment.

No GPU is needed to inspect or freeze checkpoints. In your server terminal:

```bash
export WM_ROOT=/home/zixuan/projects/WM1
cd "$WM_ROOT"
source scripts/gpu_server_env.sh

python -m scripts.phase24_freeze \
  --inspect artifacts/phase24_selection/shortened_v1/selection.json
```

If this succeeds, **skip creation** and continue to P2.5. The selection is
immutable; newer source checkpoints cannot silently replace its chosen weights.

## 2. Create a selection only if missing

In the same activated terminal, run this only when no selection exists:

```bash
python -m scripts.phase24_freeze \
  --run-root artifacts/phase24_runs/aspire2a_2gpu_v1 \
  --output-dir artifacts/phase24_selection/shortened_v1 \
  --reason "Time-limited exploratory pilot; use latest committed checkpoints"
```

Use your actual latest stopped run for `--run-root` if different.
The output directory must be new. The command verifies all four checkpoint
receipts, configuration, source, replay and normalization before writing
`selection.json` and its SHA receipt. It performs no optimizer updates.

## 3. Confirm the saved counts and proceed to P2.5

```bash
python -m scripts.phase24_freeze \
  --inspect artifacts/phase24_selection/shortened_v1/selection.json
```

This shows the **saved optimizer updates** and original target for each model.
A live `status.json` count can be ahead of the last saved checkpoint.

Review the existing [P2.5 validation](phase25_validation.md) for this exact selection;
rerun it only when new evidence is needed. Then follow the
[bounded Relay policy check](phase36_acceptance.md).
Phase 3 also uses it through `--initial-selection`; no checkpoint renaming,
configuration edits or fabricated 5,000-update completion are needed.
Freezing records the shortened budget; model-quality and real-policy checks
still apply.

Keep the selection, its source run and checkpoint/receipt pairs together when
backing up or transferring the project. It references the weights without
duplicating them. Later saves cannot change this selection.
No further initial P2.4 fitting or cluster submission is required.
