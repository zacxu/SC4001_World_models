# P6.3: final paper and local evidence review

Run on the Windows laptop in PowerShell. The completed four-run Relay cohort,
P6.1 diagnosis, P6.2 release and saved-record reproduction are the inputs.
No GPU, simulator, new episodes or training are needed.

## 1. Read the completed report

- [Research paper](../paper/research_paper_draft.md)
- [Claim-to-evidence checklist](../paper/claim_evidence.md)
- [Substantive manuscript review](../paper/manuscript_review.json)
- [Final acceptance](../artifacts/phase6_runs/paper_20261114_v1/phase6_acceptance.json)

All four actors have 0/20 successes and no latch progress. Final acceptance
means the reduced study and report are complete; successful transfer remains
unestablished. The original P6.1/P6.2 report retains its earlier false full-phase
flag because it predates this paper review.

## 2. Check the accepted local files

```powershell
cd C:\Users\willi\Desktop\SC4001_World_models
.\.venv\Scripts\python.exe -B -m study_paper.phase63 inspect --report artifacts/phase6_runs/paper_20261114_v1/phase6_acceptance.json
```

Expected: `status: passed`, `phase6_acceptance: true`, `P6.3_complete: true`.
Inspection verifies the evidence and document bytes bound by that report.
Keep the sealed release unchanged and preserve `.sha256` receipts.

## 3. Only when revising the manuscript

Regenerate the eight marked paper tables and restore the four authenticated
SVG/PNG files from the already completed reproduction:

```powershell
.\.venv\Scripts\python.exe -B -m study_paper.phase63 prepare
```

This reaggregates all 80 saved endpoint records and checks the six archived CSVs
and two numeric figure inputs. The three development tables use their original
separate report readers. It copies existing rendered figures; Matplotlib is not
required. PyYAML 6.0.3 is the only nonstandard dependency for this path and is
already installed in the laptop environment.

Review any changed prose, claims and figures, then update the findings and exact
paper/checklist hashes in `paper/manuscript_review.json`. This is a substantive
review step: a hash refresh alone is not a scientific review. The command does
not automatically mark that review complete.

After the review, create a fresh acceptance record:

```powershell
$P63Run = "artifacts/phase6_runs/paper_$(Get-Date -Format yyyyMMdd_HHmmss)"
.\.venv\Scripts\python.exe -B -m study_paper.phase63 complete --output-dir $P63Run
```

Completion checks the immutable source sets, release and reproduction receipts,
table values, copied figures, exact claim assertions, local links and the review's
document hashes. It runs synthetic rejection tests and saves their log, a paper
snapshot and `phase6_acceptance.json`. Changed evidence, an outdated review or
a failed check prevents acceptance. Existing completion directories are preserved.

The checks authenticate saved evidence; semantic claims are covered by the
explicit Codex-assisted review. They do not imply independent human peer review,
positive model quality, effective teacher transfer, or a retrainable public
archive. Venue formatting and submission remain separate presentation work.
