# EdgeCloud: Evidence Budgeting for Edge AI under Degraded Connectivity

Evidence selection across concurrent disaster incidents under a shared byte budget and
forecast delay, compared against simple rules that receive the same information, on a
synthetic benchmark (nine conditions, twenty seeds) and a reduced CrisisMMD workload.
Revised for ICCA after two reviews.

## The manuscript

The submission is the six-page version:

| File | Pages | Role |
|---|---|---|
| `manuscript/EdgeCloud_ICCA_v2_6pp_IEEEA4.pdf` | 6 | **Submission** (IEEE conference template, A4, references included in the count) |
| `manuscript/EdgeCloud_ICCA_v2_6pp_IEEEA4.docx` | | editable copy of the submission |
| `manuscript/EdgeCloud_ICCA_v2_IEEEA4.pdf` / `.docx` | 12 | full-length version of the same study |
| `manuscript/EdgeCloud_master_v2.pdf` / `.docx` | 16 | master record with every table (notation, methods, conditions, ablations, all exploratory grids) |
| `manuscript/EdgeCloud_Response_to_Reviewers_v2.docx` | | point-by-point response to both reviews, with the location of every change in the six-page and the twelve-page version |

All three versions are built from one source, `manuscript/latex/main.tex`
(`six_body.tex` holds the six-page body; the tables and figures are shared macros, so
the three versions print identical values):

```
bash manuscript/latex/build.sh            # six-page, twelve-page and master PDFs
python scripts/check_submission.py        # pre-submission gate (page counts, punctuation,
                                          # references, package digests, camera-ready slots)
```

The camera-ready slots (copyright line, archival DOI) live in `manuscript/latex/camera_ready.tex`.
The gate fails by design until the DOI is pasted in.

## Layout

```
manuscript/   the files above, LaTeX source, figures, original reviews and original submission
results/      every result file behind the paper (twenty-seed synthetic runs, CrisisMMD
              reruns and rotations, emulator transfers, established-task scores) and the
              editorial record (citation register, review report, six-page fidelity check)
scripts/      runners that regenerate the results, the figures and the deposit package
protocols/    frozen JSON protocols with SHA-256 sidecars; an edit stops the run
upstream/     the reference implementation (v2.3) behind the published numbers, unchanged
work/         the v2.7 working copy that adds the held-out contribution term
data/         derived CrisisMMD features, byte counts and labels (no posts or images)
prompts/      the review protocols the manuscript was prepared under
```

## Reproduction

Every script checks a published number before producing new ones and stops on a
mismatch. Synthetic results reproduce byte for byte on one CPU architecture; the
real-data pipeline depends on an unpinned pretrained encoder, so its image components
are reproduced to within 0.13 and the reported tables to within 1.0 error with every sign
preserved. CrisisMMD v2.0 is obtained from its authors under their terms and is not
redistributed here.

The release package (v2.6.18, 234 files with `SHA256SUMS`) is assembled by
`scripts/build_deposit_v2.6.18.sh`.

## Status

Editorial checks complete. Open before upload: the archival DOI and the copyright line in
`camera_ready.tex`, confirmation of the venue's page rule and review model, ORCIDs and an
institutional similarity check.
