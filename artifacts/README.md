# Model artifacts in Git

Source code, tests, configs, dependency locks and ordinary project documentation
remain versioned. Papers and research reports stay local: the root `.gitignore`
excludes `paper/`, `papers/`, `reports/`, `research_reports/`, `results/`, and
artifact outputs except for model files and their required operational metadata.

Model weights/checkpoints use Git LFS through `.gitattributes`. The model
exceptions include adjacent checkpoint/export manifests and SHA receipts,
run/launch configuration, selected-checkpoint metadata and normalization
statistics. Generated acceptance reports, metrics, native evaluation records,
analysis tables/figures, paper PDFs and saved research releases are excluded.

Datasets are also excluded, including root `data/`, nested real replay, NumPy
archives and synthetic distillation buffers. Buffer trajectory `state.pt` files
are training data and remain ignored despite the model-file exceptions.

Ignoring files does not delete local copies. Previously tracked papers and
research reports must also be removed from the Git index to stop including them
in subsequent commits. Historical commits are separate and are not rewritten by
these rules.

Before committing model files, run `git lfs install` in the project directory.
The existing model/checkpoint history totals about 66.97 GB before deduplication;
check the hosting account's LFS storage and bandwidth before uploading it.
A Git/LFS clone excludes datasets and reports, so preserve separate local
backups if you need the complete research archive or training inputs.
