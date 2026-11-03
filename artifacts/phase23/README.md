# P2.3 measured results

P2.3 is complete. The six approved objective terms, mask/indexing adapters,
initial fitting-only normalization, and diagnostic reports are implemented.
The model remains untrained; optimizer updates and predictive-quality
acceptance belong to P2.4 and P2.5.

- **62 new objective tests passed.** The full suite ran **389 tests:
  388 passed, 1 skipped** (Windows symlink privilege unavailable).
- Full-fitting statistics use **1,088 episodes, 182,835 transitions, and
  183,923 unique states**. The 104 model-diagnostic episodes are excluded.
- **20 real loss/gradient windows from 14 fitting episodes** cover success,
  deadline, death, collection cutoff, and six positive latch/console milestones.
  Every check has finite loss/gradients; pre-clipping global gradient L2 ranges
  from 19.6302 to 29.0026 for these untrained fixtures.
- **Zero optimizer updates and zero new simulator interactions.** Parameter
  values and the four replay manifests remain unchanged. No checkpoint saved.
- CPU float32/bfloat16 tests passed; no CUDA/H200 execution measured.

| Task / seed | Fit episodes | Fit transitions | Unique fitting states |
|---|---:|---:|---:|
| relay / 0 | 314 | 43,777 | 44,091 |
| relay / 1 | 324 | 46,729 | 47,053 |
| cover / 0 | 228 | 44,725 | 44,953 |
| cover / 1 | 222 | 47,604 | 47,826 |

Statistics are shared physically within each B/D task/seed pair. Each JSON has
field moments/counts, full fitting cohort provenance, label profiles, an
identity hash, and a SHA receipt. All continuous fields have support. Constant
fields use scale one; no class weighting or approved loss weight was changed.

The fitting profile contains only **five non-deadline failure transitions**.
Relay latch events occur in roughly 0.43–0.45% of fitting transitions; cover
console completion events occur in roughly 0.19–0.29%. This is a data-imbalance
finding, not a claim that an untrained model learns or ignores those events.
P2.4/P2.5 must assess rare events separately before any pilot adjustment.

The weighted objective is `100*RGB + bonus_Huber + outcome_CE + auxiliary_mean
+ 0.5*dynamics_KL + 0.1*representation_KL`. KL sums all 32 variables before its
per-sample one-nat floor. Each auxiliary field is reduced over its valid labels
before equal averaging across supported fields. Full-batch and uneven 1+3
microbatch totals and six gradient paths agree in the independent test.

Missing required terminal views skip joint RGB and posterior KL. Available
terminal reward/outcome/state labels still supervise the transition prior.
Reward diagnostics include deadline resolution of continue probability.

See [the runbook](../../docs/phase23_objectives.md),
[acceptance](phase23_acceptance.json), [full tests](regression_tests.json),
and [shared normalization bindings](normalization_bindings.json). These are
implementation checks; no held-out dynamics or policy-performance claim is made.
