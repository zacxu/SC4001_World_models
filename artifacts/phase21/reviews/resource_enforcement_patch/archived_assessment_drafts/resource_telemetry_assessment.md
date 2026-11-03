# Historical resource telemetry assessment

The original [main report](../phase21/phase21_acceptance.json) remains **failed**.
Its retained exception is `psutil.AccessDenied` for PID **25004**, originating from
Windows `OpenProcess` with **WinError 5** while reading CPU affinity. The fault's
time, process creation time/name/executable, and liveness classification were not
persisted. The evidence does **not** establish that this was an exiting-process
race, a live protected process, or the worker from the separate ViZDoom startup
failure.

The frozen monitor source latched the background exception and stopped sampling.
The collection loop did not check the latched failure before every operation;
context exit later raised the resource error. That exit exception can mask the
collection exception already propagating from a future. The [relay seed-1
startup failure](../../data/initial_replay/pilot/relay/seed_1/collection_error.json)
is preserved separately. These two failures must not be described as one proven
cause.

There is positive, narrower resource evidence. Frozen code configured the parent
for CPUs **0?3** and numeric thread limits of **1** before child creation. The
[point observation](../phase21/reviews/ongoing_resource_observation.json) at
**2026-11-02 22:40:10 SGT** recorded the actual campaign root and its observed descendants within
CPUs 0?3. It is one observation. Passed preflight, codec/reader, and continuation
windows have their own measurements. None provides the missing original main
collection's continuous descendant coverage, exact monitor-failure time, total
CPU usage, or complete peak-RAM series. A later inheritance probe or monitor fix
cannot reconstruct that history.

The [generation-end verification](../phase21/generation_end_verification.json)
records four sealed **50,000-transition** cases, **200,000** transitions total,
unchanged original generation sources/runtime, and completed explicit
continuation. Replay integrity, rewards, event accounting, masks, and native
execution are independently auditable. Retain all valid data. Resource telemetry
loss does not by itself establish corrupt transitions or justify recollection.
Final data acceptance still requires its separate complete audits.

The [production reader benchmark](../phase21/reviews/production_reader_benchmark.json)
is a separate, bounded window under live collection contention, using the
unchanged generation-era reader. OS cache was prefetched without controlled
residency/eviction. Its measured sequence/scattered-state rates do not certify
full-collection resource usage or eventual GPU-loader throughput.

The frozen finalizer's generic limitation remains accurate; its SHA need not
change. Link this assessment from the final README/evidence index. Interpret
?source-enforced limits? as evidence of configured parent pinning and the
observed windows, **not** continuous retrospective certification of every child.
Keep the raw failed report and receipt unchanged. Future collector hardening and
its tests should be recorded separately from this historical assessment.

The [JSON assessment](resource_telemetry_assessment.json) contains exact evidence
hashes, the original traceback, supported facts, and unsupported claims. No new
native action, process probe, source edit, or dataset change was performed for
this assessment.
