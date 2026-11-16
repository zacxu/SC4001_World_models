# P6.1-P6.2: diagnose and package the completed study

Use **Bash on the original Linux server**, with the same project environment.
These steps use CPU and saved evidence. They do not train, open the simulator,
collect episodes or load actor/model weights for inference.

Sync the new **`study_release/` folder** and updated documentation to
`/home/zixuan/projects/WM1`. Preserve `src/`, `scripts/`, `tests/`, configurations
and `study_reporting/`: their original source receipts still authenticate Phase 5.
Each step restores its environment and required paths, so a new server terminal
is supported. Stop if a command fails; Step 3 requires Step 2 to finish first.

## 1. Activate and verify

```bash
cd /home/zixuan/projects/WM1 &&
source scripts/gpu_server_env.sh &&
python -m pip install -r study_release/requirements.txt &&
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt) &&
P5_ROOT=$(dirname "$P5_COHORT") &&
P5_FINISH="$P5_ROOT/completion_20261113_091048" &&
P6_CHECK="artifacts/phase6162/check_$(date +%Y%m%d_%H%M%S)" &&
python -B -m study_release.phase6 verify --output-dir "$P6_CHECK"
```

Expected: `complete_p6162_implementation`, zero failures/errors/skips.
The completion path above is your synchronized, passed Phase 5 report.
Verification uses synthetic fixtures, including mocked graphics, rather than
your scientific records.

## 2. Generate P6.1 diagnosis and P6.2 release

This block also works in a new terminal. It reuses your existing passed
93-test server check (`check_20261113_102212`) if `P6_CHECK` is unset. When Step 1
just succeeded in the same terminal, its new check directory is used instead.

```bash
cd /home/zixuan/projects/WM1 &&
source scripts/gpu_server_env.sh &&
P5_COHORT=$(cat artifacts/phase5/latest_cohort.txt) &&
P5_FINISH="$(dirname "$P5_COHORT")/completion_20261113_091048" &&
P6_CHECK="${P6_CHECK:-artifacts/phase6162/check_20261113_102212}" &&
P6_RUN="artifacts/phase6_runs/completion_$(date +%Y%m%d_%H%M%S)" &&
python -B -m study_release.phase6 complete \
  --phase5-acceptance "$P5_FINISH/phase5_acceptance.json" \
  --verification-report "$P6_CHECK/implementation_checks.json" \
  --output-dir "$P6_RUN"
```

Expected: `status: complete`, `P6.1_complete: true`, `P6.2_complete: true`.
`phase6_acceptance: false` is intentional: **P6.3's manuscript and claim review
remain outstanding**. The command leaves the original Phase 5 files unchanged.
Wait for this command to exit successfully before Step 3. It authenticates
Phase 5 evidence before creating its output directory, then reads/copies archive
files. If it reports an error, retain the full terminal error or the `reasons`
in `phase61_62_completion.json`; do not continue to reproduction.

| Output within `$P6_RUN` | What to use |
|---|---|
| `diagnosis/diagnosis.md` | Read the fixed trace review and limits on causal explanations |
| `diagnosis/diagnosis.json` | Machine-readable observations, evidence pointers and missing categories |
| `diagnosis/selected_traces.csv` | Selected episodes and exact terminal trace lines |
| `release/README.md` | Contents, reproduction commands and limitations |
| `release/release_manifest.json`, `SHA256SUMS` | Complete packaged byte inventory |
| `release/archive_locations.json` | Original server locations, hashes and sizes of omitted large binaries/tools |
| `release/command_provenance.json` | Recorded provenance versus reproduction recipes |
| `phase61_62_completion.json` | Stage completion and remaining work |

P6.1 selects the **first success and first failure in bank order for each arm**,
at most eight traces. If a category does not exist, it is explicitly absent.
Your current zero-success results therefore select four failed traces and record
four missing successful categories. Native traces contain actions, reward/events,
entropy and timing; they do not contain RGB/positions or matched model forecasts.
They can show missing milestones, but cannot establish whether local ambiguity,
a mechanics problem or model error caused that behavior. No new ablation is run.
Available Cover screening and long B development records stay separate.

The release includes frozen source/configuration/locks, four local actors,
episode records, ledgers, deviations, numerical figure inputs and plots. Large
training/replay payloads and engine/compiler assets remain in the documented
server archive; **keep those original files**. Original JSON paths remain intact.
Portable mappings support saved-record reproduction without restoring those
paths. Exact historical training shell commands and physical GPU IDs were not
logged; the package labels recommended recipes separately. It does not upload
anything or claim independent retraining without the original archive.

## 3. Verify reproduction from the sealed release

This rebuilds **six tables and two SVG/PNG figures** from included saved records.
It checks numerical equality and exact table bytes against the archived outputs.
Use the block below from any server terminal. It finds the latest completed
release for the current frozen cohort, validates its completion/manifest
receipts, and uses a fresh output directory outside the sealed release.
It does not depend on the earlier shell variable `P6_RUN`:

```bash
cd /home/zixuan/projects/WM1 &&
source scripts/gpu_server_env.sh &&
P6_RELEASE=$(python -B - <<'PY'
from pathlib import Path
import sys
from study_release.evidence import file_sha256, read_report

cohort_path = Path(Path("artifacts/phase5/latest_cohort.txt").read_text().strip())
cohort_id = read_report(cohort_path)["cohort_identity"]
for path in sorted(Path("artifacts/phase6_runs").glob("completion_*/phase61_62_completion.json"), reverse=True):
    report = read_report(path)
    if not (report.get("schema_version") == "p6162-completion-v1"
            and report.get("cohort_identity") == cohort_id
            and report.get("status") == "complete"
            and report.get("P6.1_complete") is True
            and report.get("P6.2_complete") is True):
        continue
    release = (path.parent / "release").resolve()
    manifest_path = release / "release_manifest.json"
    manifest = read_report(manifest_path)
    ref = report["release_manifest"]
    if (Path(ref["path"]).resolve() != manifest_path
            or file_sha256(manifest_path) != ref["sha256"]
            or manifest_path.stat().st_size != ref["bytes"]
            or manifest.get("schema_version") != "p62-minimal-release-v1"
            or manifest.get("status") != "complete"
            or manifest.get("cohort_identity") != cohort_id
            or not (release / "source/study_release/saved.py").is_file()):
        raise SystemExit(f"Release validation failed: {release}")
    print(f"Using completed release: {release}", file=sys.stderr)
    print(release)
    break
else:
    raise SystemExit("No completed P6 release for this cohort. Inspect the Step 2 output before proceeding.")
PY
) &&
P6_REPRO="$(pwd)/artifacts/phase6_runs/reproduction_$(date +%Y%m%d_%H%M%S)" &&
(
  cd "$P6_RELEASE/source" &&
  python -B -m study_release.saved regenerate \
    --release-dir "$P6_RELEASE" --output-dir "$P6_REPRO"
)
```

If you saw `/home/zixuan/projects/WM1//release/source`, the previous terminal's
`P6_RUN` variable was empty. Rerun this updated Step 3; restoring the path does
not require repeating Step 2. If no completed release is found, inspect Step 2's
report instead of guessing a directory.
If the diagnostic shows `P6 directories: 0`, return to the self-contained
Step 2 above. A passed implementation check in Step 1 does not generate a
scientific release. Existing current-source checks and Phase 5 results can be
reused; no training or native evaluation is required.

Expected: `status: complete`, `numerical_reproduction: true`,
`six_tables_byte_identical: true`, `two_figures_rendered: true`.
Keep **`-B`**: it prevents Python bytecode files from changing the sealed package's
file inventory. Matplotlib rendering can vary across platforms; the plotted
numbers and table bytes are checked, without claiming universal image-byte equality.

Then retain the release and reproduction report, sync them back for writing,
and continue with **P6.3**. Review the diagnosis's undetermined categories and all
zero/negative results before updating manuscript claims. A stage completion
marker is evidence completeness, not proof of successful teacher transfer.


## Completed handoff to P6.3

The synchronized `completion_20261113_104421` and `reproduction_20261113_104832`
reports are complete. Continue on the laptop with the [P6.3 paper guide](phase63_paper.md).
Its new acceptance report covers the final manuscript and evidence review;
historical P6.1/P6.2 reports remain unchanged. Repeating server generation is
unnecessary for writing or verifying the completed paper.
