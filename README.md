# ICCA Revision Package: Edge evidence budgeting

Everything needed to close the two reviews, in one place.

```
manuscript/   revised paper (PDF, DOCX, LaTeX source + IEEEtran.cls + figures),
              response letter, original review text
Rerun_Spec.md the experiment plan behind scripts/ (read first)
protocols/    frozen JSON protocols, one per item; a .sha256 sidecar is written
              on first load and any later edit stops the run
scripts/      runners for items A, C, D, E plus common.py and adapters.py
results/      empty; every script writes here with a _rev1 suffix
```

## What needs you before anything runs

`scripts/adapters.py` is the only file that touches the Edge repository. Its
five functions raise `NotImplementedError` until wired to your loaders:

| function | used by | needs |
|---|---|---|
| `load_crisismmd_rotation(rot)` | A, C, D | full-dimension CLIP text and image embeddings, labels, events, roles for rotations 0..6; rotation 0 must give n_test = 968 |
| `edge_text_model(class_weight, seed)` | A | your edge text model factory (may return None to skip) |
| `run_synthetic_condition(cond, seeds, methods)` | E | the benchmark runner that produced the paper's tables |
| `run_real_budgeting(split, cond, seed, methods, ...)` | C, D | the real-data runner, with optional injected delays and class weighting |
| `derive_wait_threshold(split, forecast)` | D | the package's T* derivation on calibration windows |

Every script checks reproduction of a published number before producing new
ones and stops on a mismatch, so a wrong adapter cannot silently produce a
plausible table.

## Order

```
pip install numpy scipy scikit-learn
python scripts/item_a_representation_diagnostics.py   # writes results/itemA_decision.json
python scripts/item_c_real_class_handling.py          # needs itemA_decision.json
# item B: follow protocols/itemB_protocol.json; no runner supplied (see note inside)
python scripts/item_d_real_delay_threshold.py
python scripts/item_e_more_seeds.py
```

Each script prints a `MANUSCRIPT:` line naming the sections and tables that
its output changes. Do not edit the manuscript from a script that stopped at a
reproduction gate.

## Two author actions that do not need the reruns

1. Replace the red `AUTHOR CHECK` marker in Section III-B of the manuscript
   with the value-function formula as implemented (`\authorcheck{...}` in
   `manuscript/latex/main.tex`).
2. Deposit the package and replace the DOI marker in Section XI.

## Known assumptions written into the protocols

- `itemD_protocol.json`: `sigma_delay` is set to 0.4, equal to the forecast
  error, because the manuscript did not record the synthetic delay spread
  separately. If the package configuration differs, create a new protocol file
  rather than editing this one.
- `itemA_protocol.json`: the TF-IDF control is omitted because raw text is not
  shipped with the package.
- Reference values in the protocols are copied from the revised manuscript's
  tables; they are the reproduction gates.
