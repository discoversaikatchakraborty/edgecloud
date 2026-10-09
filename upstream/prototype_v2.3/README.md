# EdgeCloud Emergency AI Prototype (v2.6.16)

This package implements the evidence-budgeting method described in **Evidence
budgeting for edge AI under degraded connectivity: a validation study of adaptivity,
admissibility, and adversarial metadata, on synthetic and real disaster data**.

It contains two workloads. The main benchmark is a controlled synthetic proof of
concept: it tests the method and the software path, and it is not evidence of field
performance. A reduced version of the same comparison also runs on a real workload
built from CrisisMMD, which is what Section 6.12 of the manuscript reports.

## Run

```bash
pip install -r requirements-lock.txt   # exact versions of the reported run (Python 3.12)
python test_prototype.py               # integrity tests (about 10 s)
python run_experiment.py               # 5 seeds, all methods, conditions, ablations, sweeps (about 3 min)
python run_contested_budget.py         # pre-registered contested-budget test, 10 seeds (about 1.5 min)
python run_contested_exploratory.py    # exploratory summary-floor variant (labeled as such)
python run_contested_confirmation.py   # second pre-registered test on 10 new seeds
python contested_hold_rates.py         # diagnostic: how often each allocator drops a modality
python make_figures.py                 # regenerates every manuscript figure from results/
python run_spoof_sweep.py              # adversarial metadata forgery against admissibility
python run_adversary_grid.py           # forgery crossed with a compromised text source
python run_link_realism.py             # packet loss, and replay of measured LTE capacity traces
python run_decentralized.py            # cost of removing coordination between controllers
python make_figures_adhoc.py           # submission figures into figures_submission/
python make_graphical_abstract.py      # Elsevier graphical abstract
python run_compute_cost.py             # measured edge-side decision cost
python prepare_crisismmd.py --src DIR  # derive the real workload from an extracted CrisisMMD archive
python run_real_data.py                # the policy comparison on real disaster data
python make_figures_realdata.py        # figures for the real-data sections
python representation_diagnostics.py   # per-modality classifier accuracy on the real workload
python run_split_robustness.py         # the real-data comparison over seven rotations of the event split
python escalation_diagnostics.py       # how often the severity escalation rule actually fires
python prose_diagnostics.py            # scalars the manuscript states in prose, derived
python make_notebook.py                # rebuilds the notebook: a walkthrough of the core
                                       # benchmark only, not a reproduction of the paper
python test_table_completeness.py PATH # every table shows every row of its source
python test_numbers_traceable.py PATH  # every number in the prose traces to a results file
```

`results/experiment_manifest.json` records the SHA-256 of `config.json` and
`core.py`, the runtime versions, the event splits, the calibration
temperatures, and the derived wait threshold for each seed. Rerunning with the
locked environment on the same CPU architecture reproduces every output file byte
for byte, and this also holds across Python 3.12 to 3.14, NumPy 2.4.4 to 2.5.3,
pandas 3.0.2 to 3.0.6 and scikit-learn 1.8.0 to 1.9.1. It does not hold across
architectures: `rng.beta` and `rng.dirichlet` reach platform libm, so about 10% of
source observation vectors differ in their last bits between Linux x86_64 and
macOS arm64. Every reported quantity is written through `float_format="%.6g"` or
an explicit `round` and is unaffected. Since 2.3.0 the provenance digest is taken
over the observation rounded to nine decimals so that it, too, is stable across
architectures.

## What the benchmark does and does not show

Main findings, mean of five seeds, 720 held-out windows per seed (95% t intervals
in `results/confidence_intervals.csv`). In every condition of the **synthetic**
benchmark, a simple rule informed by the formulation matched the optimizer. The real
workload does not reproduce that ranking; see "Real disaster data" below.

* **Low bandwidth (256 kbps).** A fixed rule (image summary, raw text, raw sensor)
  matched the policy: 4.2 vs 4.4 errors, 1,854 vs 3,590 bytes, p95 0.59 vs 0.92 s.
* **Delayed imagery (stochastic, forecast-only delay).** The policy cut p95 latency
  from 1,565 s (wait-if-uncertain, which ignores the forecast) to 355 s. A constant
  forecast-delay threshold derived from the closed form on calibration events
  (196 to 219 s across seeds) did at least as well: 9.4 vs 9.2 errors (paired
  difference -0.2 [-2.4, 2.0]), p95 26 s, 29% fewer bytes. Accuracy gains over
  always-wait baselines exist only because evidence is assumed to go stale while a
  decision waits (`sensitivity:*` rows in `ablation_results.csv`).
  `wait_threshold_sweep.csv` shows the constant-threshold frontier.
* **Misaligned evidence.** The fixed rule with the admissibility filter made 9.4
  errors, the proposed system 8.8, the fixed rule without the filter 20.2. The
  filter, not the optimizer, accounts for the gain, and the generator makes the
  filter's job easy (aligned metadata is exact; misaligned reports come from
  distant events).
* Novelty, reliability and severity terms produced no measurable benefit in
  their stress tests; the quality term reduced errors by about 1 to 1.4 windows.
* The combined abstention/review queue holds about 30% of windows; under low
  bandwidth 98% of queued windows are correct. Conformal abstention alone (about
  10% of windows) catches nearly all errors; the conflict rule (JS >= 0.15) adds
  11 to 43 points of queue for at most 2 extra caught errors per 720 windows
  (`review_rule_sensitivity.csv`). Prespecified settings are kept in the main results.

## Contested shared budget (option B)

The main conditions never make the shared budget bind, so the knapsack could not
matter there. `PROTOCOL_contested_budget.md` was frozen (SHA-256 in `config.json`)
before the proposed method ran on a harder condition (noise scale 1.75, chosen from
baselines on calibration events) with per-window budgets of 700 to 1,900 bytes.

* The optimizer did not beat a shared-budget greedy rule that upgrades the most
  uncertain items at any budget after Holm adjustment. At 700 bytes it was worse
  (+20.8 errors) because its value function prices a dropped modality at zero, so it
  held image or text summaries to buy raw sensor records.
* The exact solver matched a greedy solver on the same utility to within one error.
* An exploratory summary floor fixed the tight-budget failure. On ten new seeds
  (`PROTOCOL_contested_budget_confirmation.md`) its roughly 3-error advantage over the
  greedy rule at 1,200 to 1,900 bytes replicated in direction but did not survive
  Holm adjustment.

Outputs: `contested_budget_*.csv`, `contested_budget_manifest.json`, `figures/contested_budget.png`.

## Methods compared

Image only, text only, transmit all (unconstrained reference that ignores the
budget), budget-aware confidence gate, fixed compression, fixed rule, random
action within budget, proposed; under delayed imagery only, wait if uncertain and
the derived constant wait threshold; under misaligned evidence only, the fixed rule
with admissibility.

## Conditions

Normal, low bandwidth, delayed imagery, intermittent outage, image missing,
text missing, noisy text, contradictory evidence, misaligned evidence.

## Outputs

* `seed_level_results.csv`, `confidence_intervals.csv`, `paired_differences.csv`
* `ablation_results.csv`, `ablation_paired_differences.csv`, `latency_penalty_sweep.csv`
* `wait_threshold_sweep.csv`, `review_rule_sensitivity.csv`
* `status_summary.csv`, `error_analysis.csv`
* `spoof_sweep_*`, `adversary_grid_*` (adversarial metadata and compromised-source results)
* `link_loss_*`, `link_trace_*`, `link_realism_manifest.json` (packet loss and measured-capacity replay)
* `decentralized_*` (cost of removing controller coordination)
* `compute_cost_methods.csv`, `compute_cost_scaling.csv` (measured edge-side decision cost)
* `real_data_summary.csv`, `real_data_paired.csv`, `real_data_seed_level.csv`,
  `real_data_manifest.json` (the comparison on CrisisMMD)
* `real_data_representation.csv` (per-modality classifier accuracy on the real workload)
* `real_data_split_robustness.csv` (the same comparison over seven rotations of the event split)
* `escalation_rates.csv`, `prose_diagnostics.csv` (scalars the manuscript states in prose)
* `record_level_results.csv.gz` (every method, condition and seed)
* `evidence_envelopes_proposed.jsonl.gz` (full provenance envelope for every proposed decision;
  each envelope carries its own SHA-256, and each evidence item carries a digest of its source vector)

## Network model and traces

Version 2.4 adds two optional link parameters whose defaults reproduce every earlier
result exactly. `packet_loss` inflates bytes on the wire by 1/(1 - p) and charges a
200 ms timeout per lost packet, so the penalty grows with payload. `link_trace`
replays a measured cellular capacity trace from `data/traces/` instead of a constant
rate; `data/traces/provenance.json` records the source, per-file SHA-256 and summary
statistics of each. Traces are third-party measurements, replayed unmodified, and
each is compared against a constant link of the same mean capacity so that the
comparison isolates variability rather than speed. `controllers` partitions a slot's
windows across independent allocators. There is still no MAC layer, routing,
inter-node contention, congestion control or bidirectional traffic.

## Real disaster data

`prepare_crisismmd.py` derives a workload from an extracted CrisisMMD v2.0 archive
(3,526 labelled report-image pairs from seven 2017 disasters, split by event) and
`run_real_data.py` runs the same policies on it, unchanged. The archive itself is
not redistributed; `data/crisismmd/provenance.json` records the source URL, the
split, the encoder, the reduction and a digest of the derived records.

The comparison is reduced, not repeated: two modalities rather than three, three
conditions rather than nine, and three of the eleven methods absent. What it shows:

* **The synthetic ranking does not carry over.** Against the fixed rule the
  optimizer is ahead in 15 of 21 comparisons across seven rotations of the event
  split, and in every rotation when the link is not the constraint
  (`real_data_split_robustness.csv`). Which way any single condition falls depends
  on which disasters are held out.
* **The exact solver and a greedy heuristic over the same utility are identical** on
  this workload, in every condition and at every seed.
* **One modality carries no signal for the label** and no term in the value function
  can detect it, which is the finding the manuscript argues is most transferable.
* **Conformal coverage does not survive the move**, falling from 0.883-0.898 on
  synthetic data to 0.744-0.816, because exchangeability breaks across events.
* **The assumed byte model is contradicted by measurement**: real images are about
  four times smaller than assumed, and real text is smaller than the summary the
  method defines as its cheap alternative.

## Licence and citation

The code is under the **Apache License 2.0** (`LICENSE`), which is permissive and
carries an explicit patent grant. The data, result files and figures are under
**CC BY 4.0** (`LICENSE-DATA`). `CITATION.cff` records both and carries the citation
for the software and the paper; its author fields use the same placeholders as the
manuscript.

The CrisisMMD corpus itself is **not** redistributed here and is not covered by
either licence. What ships is derived from it — reduced CLIP embeddings, byte counts,
labels, hazard tags and recovered arrival times — and onward use of anything derived
from it remains subject to the corpus authors' terms as well as to CC BY 4.0. Where
the two conflict for some use, the corpus authors' terms govern. The capacity traces
are likewise third-party material under their own terms, recorded in
`data/traces/provenance.json`.

Both licence files carry a `[year] [copyright holder]` line to complete before
deposit, alongside the author placeholders elsewhere.

## Important limitations

The synthetic generator has been supplemented with real disaster data, not replaced
by it: the link, the budgets, the loss model, the slot structure, the sensor
modality and severity all remain synthetic in Section 6.12. Latency is modeled from bytes, link rate, a shared
queue and injected delays; it is not a packet-level emulation. Policy
coefficients and thresholds are fixed by hand in `config.json`; they were not
selected on a validation split, and the prespecification is not externally
timestamped. The derived wait threshold is the only quantity selected from data,
and it uses calibration events only. The manuscripts are in `../Option3_Merged`
(the submission) and `../Option1_PaperA_AdHocNetworks` and
`../Option1_PaperB_RealDataCompanion` (retained as analysis).
