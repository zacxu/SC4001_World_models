# Versioned reports and local research artifacts

Git keeps the selected top-level acceptance reports, manifests, receipts, review
summaries, small validation-interaction ledgers, and audit helper scripts allowed
by the root .gitignore. Source code, tests, configs, dependency locks, plans, and
editable scenario ACS sources are versioned in their normal project directories.

Raw simulator traces, per-worker journals, generated maps/textures, videos,
screenshots, source snapshots, scratch investigations, and future collection or
training outputs are excluded. Ignoring them does not delete local files.
Back up any raw evidence or datasets you need separately from Git.

The saved reports describe historical measured runs. They contain original
absolute paths and hashes of evidence files, including files not shipped in this
repository. Preserve report bytes and receipts; do not replace their old paths.
A clone contains the reports, not a self-contained copy of the original raw
evidence bundle, and cannot validate all referenced evidence without that bundle.
Old acceptance caches are not portable to a new checkout or virtual environment.

## Recreate the environment and generated assets

On Windows, use Python 3.12 and run from the project root:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-phase2-cpu.lock
.\.venv\Scripts\python.exe -m pip install --no-build-isolation -e .
.\.venv\Scripts\python.exe -m scripts.bootstrap_acc
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -m unittest discover -s tests -q
.\.venv\Scripts\python.exe -m scripts.phase1 smoke --task both --output-dir artifacts/fresh_clone_smoke
```

The bootstrap verifies/downloads the pinned ACC compiler. ViZDoom supplies the
bundled Freedoom IWAD; scenario geometry and extraction textures are generated
from the checked-in builder and ACS sources. Downloaded tooling and installed
dependencies do not need to be committed.

For Linux setup and compiler requirements, see ../scenarios/README.md and the
Phase 1 runbooks under ../docs/.

To produce a new complete native acceptance bundle, use a fresh output directory:

```powershell
.\.venv\Scripts\python.exe -m scripts.phase1_acceptance verify --output-dir artifacts/fresh_clone_acceptance
```

This runs the full campaign, including the 10,000-decision and 100-reset checks;
it is separate from the short clone smoke check. Native execution needs local UDP
and Windows message-queue access. Use a new directory for subsequent campaigns
and retain historical reports rather than overwriting them.

P2.2 adds small architecture-verification reports under `artifacts/phase22/`.
They cover the untrained model, explicit random streams, and read-only replay
interfaces. No model checkpoint or prediction-quality result is included.
Install `requirements-phase2-cpu.lock` for the full current unit suite, and see
the [architecture runbook](../docs/phase22_architecture.md) for a fresh check.

P2.3 adds objective-verification reports and frozen fitting-only normalization
metadata under `artifacts/phase23/`. Each B/D task/seed pair references the same
statistics file. These metadata contain counts, field moments, episode IDs,
and hashes; replay RGB and model checkpoints remain excluded. See the
[objective runbook](../docs/phase23_objectives.md) to regenerate or recheck them.

P2.5 keeps compact engineering implementation and frozen replay-cohort evidence
under `artifacts/phase25/`. These are CPU/source checks, separate from trained
checkpoint quality and formal Phase 2 acceptance. Server evaluation reports,
model-selection receipts and weights remain local; preserve a separate backup.
See the [compact validation guide](../docs/phase25_validation.md).
