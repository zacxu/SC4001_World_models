# Pilot backup and final-campaign capacity

The measured **200,000-transition** pilot occupies **7.014 GiB of replay**
and **7.905 GiB across complete case directories**, including runtime
archives, interaction ledgers, bindings, and recovery evidence. The
[storage inventory](../storage_inventory.json) recorded **791.6 GiB
free on D:** at its measurement time. These are measured bytes; they are not a
claim about current free space after later copies or other disk use.

At the observed pilot bytes per transition, **40 million transitions** project to
approximately **1.698 TB decimal** for the complete-case retention format.
That is an extrapolation, not a storage guarantee. Checkpoints, final evaluation
outputs, independent backups, temporary working copies, filesystem overhead, and
headroom require separate budgets. Image entropy, learned behavior, episode
lengths, and recovery frequency can change the result. Shared-initialization
savings should be calculated explicitly rather than assumed to cover those
additional costs. Nominal raw RGB alone is approximately **2.949 TB decimal** at
40 million transitions before extra episode-start observation slots.

The measured D: free space is below the projected full retention footprint.
Before starting confirmatory/final seeds, name and verify an archive destination
and a capacity policy that covers replay, runtime/lineage evidence, checkpoints,
evaluation, backups, and working margin. The final-campaign gate remains closed
until that storage and restore procedure is provisioned. **No external archive or
backup destination has been authorized or verified in this task.**

Keep each sealed case directory together: `replay/`, `bindings/`, the exact
training ledger and collection specification/report, runtime archives/receipts,
and all recovery lineage plus preserved original-prefix evidence. Retain the
source snapshots, dependency locks, task/split/target registries, and acceptance
reports alongside a project version that can decode the data. B/D bindings point
to one shared store; do not create an unnecessary second physical replay copy.
Raw data is ignored by Git, so pushing source and summaries is not a backup of
these case directories.

Once an approved target exists, copy sealed data, compare file hashes and
manifests, and perform a restore to a fresh location. Verify both B/D bindings,
chunk identities, full production replay integrity, and archived runtime evidence
using the restored copy. Do not delete the only local case copy before the
restore succeeds. A second directory on the same physical drive is a working
copy, not independent protection against drive loss.

C: hot journals are a separate cleanup concern. Their removal is permitted only
under the finalized exact-path/hash cleanup plan after every ledger has its
verified D: counterpart. The original interrupted relay seed-1 hot ledger matches
its preserved **32,700-transition before-snapshot**, not the later 50,000-transition
ledger. Never delete or regenerate valid replay to resolve the telemetry issue.

Keep only a bounded replay cache in RAM. The production reader measurements
support contiguous sequence access more strongly than scattered single states:
approximately **752-807 joint observations/s** for the measured 50-state windows
versus **21-25/s** for shuffled single states on that contended, prefetched-cache
host. They do not validate the eventual four-H200 training loader. Benchmark the
actual recurrent batching, preprocessing, transfers, and training demand before
scaling the final campaign; no chunk-format change or recollection follows from
these measurements alone.

The exact measured inputs are retained in [the capacity assessment](backup_capacity_assessment.json).
