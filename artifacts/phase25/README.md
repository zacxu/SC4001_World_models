# Compact P2.5 implementation evidence

The evaluator implements the reduced engineering screen for an exploratory
Phase 3 integration pilot. See the [Linux guide](../../docs/phase25_validation.md)
for server commands. It does not start training or simulator collection.

Local CPU verification: **55 tests passed, zero failures/errors/skips**. The
four existing replay stores each supply 32 uniform and 32 event windows;
required milestone support ranges from 8 to 16 distinct positives per selected
event stratum. No trained checkpoint or server GPU run was evaluated here.
The user reports P2.4 training is ongoing; local files may not reflect its
current optimizer counts or saved checkpoints.

- `implementation_v1/implementation_checks.json` and its SHA receipt record
  the targeted structural and P2.5 implementation checks on local CPU fixtures.
- `cohort_review_v1/replay_cohorts.json` and its receipt audit the four existing
  replay stores and bounded diagnostic coordinates. This contains no learned
  predictions and makes no checkpoint-quality or GPU-runtime claim.

Server screens write fresh `report.json` files with receipts, exact checkpoint
updates and source identities, separate uniform/event metrics, support and
timing. A four-model campaign also writes `pilot_model_selection.json`;
it preserves readiness and all four shared checkpoints without finalizing a
P2.4 run or replacing its B/D bindings. Server runs and weights remain excluded
from Git. Retain those artifacts separately.

For campaign handoff, require the final report and selection with both SHA
receipts. The report is published last. The selection's
`campaign_identity_sha256` is the canonical hash of the campaign report with
the `pilot_model_selection` field excluded; the final report binds the actual
selection file SHA. A selection alone is not a completed campaign.

`P2.5_quality_acceptance` remains false. Diagnostic episodes use training
geometry, and failure/death classes without positive examples remain
unvalidated. These implementation checks do not establish trained-model
quality, policy learning, or a B-versus-D benefit.
