"""Item A: text-pathway diagnostic (Reviewer 2.3, Reviewer 1.2).

Question: is the majority-class text classifier a property of the task, of the
8-component PCA reduction or of class imbalance?

Grid (frozen in protocols/itemA_protocol_rev1.json):
  representation x class_handling x classifier, on rotation 0 first, then the
  best cell on all seven rotations. Representation runs at 8 PCA components,
  at 32, at full CLIP dimension and at TF-IDF as a control. PCA and the TF-IDF
  vocabulary are fitted on the training rows of the rotation only.

The full-dimension arm needs the 512-dimension CLIP text embedding, which the
package does not ship: text_x0..text_x7 is a projection to eight components and
is not invertible. scripts/build_full_text_embeddings.py recomputes it with the
encoder named in provenance.json and must be run first.

Outputs:
  results/representation_diagnostics_rev1.csv            every cell, rotation 0
  results/representation_diagnostics_rev1_rotations.csv   best cell, rotations 0..6
  results/itemA_decision_rev1.json                        decision-rule outcome
  results/itemA_manifest_rev1.json                        frozen protocol and digest

Run:  python scripts/item_a_representation_diagnostics.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, recall_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import adapters  # noqa: E402
from common import RESULTS, load_protocol, rng, run_manifest, write_csv  # noqa: E402

PROTOCOL = "itemA_protocol_rev1.json"


def balanced_resample(X, y, seed):
    """Oversample minority classes to the majority count. Works on sparse rows too."""
    r = rng(seed)
    classes, counts = np.unique(y, return_counts=True)
    target = counts.max()
    idx = []
    for c in classes:
        ci = np.flatnonzero(y == c)
        idx.append(r.choice(ci, size=target, replace=len(ci) < target))
    idx = np.concatenate(idx)
    r.shuffle(idx)
    return X[idx], y[idx]


def build_features(split, rep_kind: str, proto: dict, seed: int):
    """Train and test matrices for one representation, fitted on training rows only.

    'pca8' and 'pca32' fit PCA on the training rows. 'full' is the untouched 512
    dimension CLIP text embedding. 'tfidf' fits the vocabulary and the document
    frequencies on the training text only. 'pca8_local4' is the reference cell and
    returns the eight shipped components; the truncation to four lives in the
    estimator, as it does in the package.
    """
    tr, te = split.mask("train"), split.mask("test")
    if rep_kind == "tfidf":
        if split.text_raw is None:
            raise RuntimeError("raw text unavailable; the TF-IDF control cannot run")
        params = dict(proto["tfidf"]["params"])
        params["ngram_range"] = tuple(params["ngram_range"])
        v = TfidfVectorizer(**params)
        Xtr = v.fit_transform(split.text_raw[tr])
        Xte = v.transform(split.text_raw[te])
        return Xtr, Xte
    if rep_kind == "pca8_local4":
        return split.X_text_pca8[tr], split.X_text_pca8[te]
    X = split.X_text_full
    if rep_kind == "full":
        return X[tr], X[te]
    if rep_kind.startswith("pca"):
        n = int(rep_kind[3:])
        p = PCA(n_components=n, random_state=seed).fit(X[tr])
        return p.transform(X[tr]), p.transform(X[te])
    raise ValueError(rep_kind)


def fit_predict(classifier: str, class_handling: str, Xtr, ytr, Xte, seed: int):
    cw = "balanced" if class_handling == "class_weight" else None
    if class_handling == "resample":
        Xtr, ytr = balanced_resample(Xtr, ytr, seed)
    sparse = sp.issparse(Xtr)
    if classifier == "logreg":
        clf = make_pipeline(
            StandardScaler(with_mean=not sparse),
            LogisticRegression(max_iter=2000, class_weight=cw, random_state=seed),
        )
    elif classifier == "package_edge":
        clf = adapters.edge_text_model(class_weight=cw, seed=seed)
    elif classifier == "package_edge_local4":
        clf = adapters.edge_text_model(class_weight=cw, seed=seed, local_dims=4)
    else:
        raise ValueError(classifier)
    if clf is None:
        return None
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def metrics(y_true, y_pred, classes):
    maj = np.bincount(y_true).argmax()
    y_maj = np.full_like(y_true, maj)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "majority_accuracy": accuracy_score(y_true, y_maj),
        "majority_macro_f1": f1_score(y_true, y_maj, average="macro"),
        "majority_class": int(maj),
        "n_test": int(len(y_true)),
        "n_classes_predicted": int(len(np.unique(y_pred))),
        **{f"recall_class{c}": r for c, r in
           zip(classes, recall_score(y_true, y_pred, labels=classes, average=None,
                                     zero_division=0))},
    }


def run_cell(split, rep_kind, class_handling, classifier, proto, seed):
    tr, te = split.mask("train"), split.mask("test")
    Xtr, Xte = build_features(split, rep_kind, proto, seed)
    y_pred = fit_predict(classifier, class_handling, Xtr, split.y[tr], Xte, seed)
    if y_pred is None:
        return None
    m = metrics(split.y[te], y_pred, np.unique(split.y))
    m["n_features"] = int(Xtr.shape[1])
    return m


def main():
    proto = load_protocol(PROTOCOL)
    os.makedirs(RESULTS, exist_ok=True)

    split0 = adapters.load_crisismmd_rotation(0)
    counts = {r: int(split0.mask(r).sum()) for r in ("train", "cal", "test")}
    exp = proto["expected_n_rotation0"]
    assert counts == exp, f"rotation 0 sizes {counts} != {exp}; the split is not the paper's"
    assert split0.text_features_are_full, (
        "X_text_full is still the 8-component projection. Run "
        "scripts/build_full_text_embeddings.py before item A; the shipped features "
        "cannot stand in for the full embedding.")
    assert split0.X_text_full.shape[1] == proto["representation_source"]["full_dimension"]
    run_manifest(os.path.join(RESULTS, "itemA_manifest_rev1.json"), proto, extra={
        "rotation0_sizes": counts,
        "text_full_dimension": int(split0.X_text_full.shape[1]),
        "text_embedding_cache_manifest": json.load(open(os.path.join(
            os.path.dirname(RESULTS), "data", "crisismmd_text_full_manifest.json"))),
        "outputs": proto["outputs"],
    })
    print(f"[split] rotation 0 train {counts['train']} cal {counts['cal']} test {counts['test']}")
    print(f"[text]  full embedding {split0.X_text_full.shape}")

    rows = []
    for rep_kind in proto["representations"]:
        for ch in proto["class_handling"]:
            for clf in proto["classifiers"]:
                for seed in proto["seeds"]:
                    m = run_cell(split0, rep_kind, ch, clf, proto, seed)
                    if m is None:
                        continue
                    rows.append({"rotation": 0, "representation": rep_kind,
                                 "class_handling": ch, "classifier": clf,
                                 "reference_cell": False, "seed": seed, **m})
                print(f"rot0 {rep_kind:6s} {ch:12s} {clf:18s} "
                      f"acc={np.mean([r['accuracy'] for r in rows[-len(proto['seeds']):]]):.4f} "
                      f"(maj {rows[-1]['majority_accuracy']:.4f}) "
                      f"macroF1={np.mean([r['macro_f1'] for r in rows[-len(proto['seeds']):]]):.4f} "
                      f"classes_pred={rows[-1]['n_classes_predicted']} d={rows[-1]['n_features']}")

    # The deployed edge text pathway, reported for comparison and kept out of selection.
    ref = proto["reference_cell"]
    for seed in proto["seeds"]:
        m = run_cell(split0, ref["representation"], ref["class_handling"],
                     ref["classifier"], proto, seed)
        rows.append({"rotation": 0, "representation": ref["representation"],
                     "class_handling": ref["class_handling"],
                     "classifier": ref["classifier"], "reference_cell": True,
                     "seed": seed, **m})
    print(f"rot0 {ref['representation']:6s} {ref['class_handling']:12s} "
          f"{ref['classifier']:18s} "
          f"acc={np.mean([r['accuracy'] for r in rows[-len(proto['seeds']):]]):.4f} "
          f"macroF1={np.mean([r['macro_f1'] for r in rows[-len(proto['seeds']):]]):.4f} "
          f"classes_pred={rows[-1]['n_classes_predicted']}   [paper's own configuration]")

    write_csv(os.path.join(RESULTS, "representation_diagnostics_rev1.csv"), rows)

    # Best cell by mean macro-F1 over seeds on rotation 0, among the frozen grid cells.
    agg = collections.defaultdict(list)
    for r in rows:
        if r["reference_cell"]:
            continue
        agg[(r["representation"], r["class_handling"], r["classifier"])].append(r["macro_f1"])
    best = max(agg, key=lambda k: np.mean(agg[k]))
    print(f"[best cell on rotation 0] {best} mean macro-F1={np.mean(agg[best]):.4f}")

    # Rerun_Spec section A words the rule over ANY cell, not only the macro-F1 best one.
    # The protocol records that as decision_rule_secondary and requires it to be reported
    # alongside the primary rule, so every grid cell that satisfies the accuracy condition
    # on rotation 0 is carried to all seven rotations as well. Selecting on macro-F1 alone
    # would otherwise leave the cells that beat the baseline on accuracy untested.
    acc_gain0 = {}
    for k in agg:
        sel = [r for r in rows if not r["reference_cell"]
               and (r["representation"], r["class_handling"], r["classifier"]) == k]
        acc_gain0[k] = float(np.mean([r["accuracy"] - r["majority_accuracy"] for r in sel]))
    candidates = [k for k, g in acc_gain0.items() if g > proto["decision_rule"]["min_accuracy_gain"]]
    carried = [best] + [k for k in candidates if k != best]
    print(f"[carried to all rotations] best cell {best}; "
          f"accuracy-condition candidates {candidates}")

    rot_rows = []
    for cell in carried:
        role = "best_cell" if cell == best else "accuracy_candidate"
        for r in rows:
            if not r["reference_cell"] and (r["representation"], r["class_handling"],
                                            r["classifier"]) == cell:
                rot_rows.append({**{k: v for k, v in r.items()}, "role": role})
        for rot in range(1, 7):
            split = adapters.load_crisismmd_rotation(rot)
            for seed in proto["seeds"]:
                m = run_cell(split, cell[0], cell[1], cell[2], proto, seed)
                rot_rows.append({"rotation": rot, "representation": cell[0],
                                 "class_handling": cell[1], "classifier": cell[2],
                                 "reference_cell": False, "seed": seed, "role": role, **m})
        for rot in range(0, 7):
            sel = [r for r in rot_rows if r["rotation"] == rot
                   and (r["representation"], r["class_handling"], r["classifier"]) == cell]
            print(f"rot{rot} {cell[0]:6s} {cell[1]:12s} {cell[2]:18s} "
                  f"acc={np.mean([r['accuracy'] for r in sel]):.4f} "
                  f"(maj {sel[0]['majority_accuracy']:.4f}) "
                  f"macroF1={np.mean([r['macro_f1'] for r in sel]):.4f} "
                  f"(maj {sel[0]['majority_macro_f1']:.4f}) n_test={sel[0]['n_test']}")
    write_csv(os.path.join(RESULTS, "representation_diagnostics_rev1_rotations.csv"), rot_rows)

    # Decision rule, frozen before the run.
    d = proto["decision_rule"]

    def tally(cell):
        detail, passing, f1_passing = {}, 0, 0
        for rot in range(0, 7):
            rs = [r for r in rot_rows if r["rotation"] == rot
                  and (r["representation"], r["class_handling"], r["classifier"]) == cell]
            acc_gain = float(np.mean([r["accuracy"] - r["majority_accuracy"] for r in rs]))
            f1_gain = float(np.mean([r["macro_f1"] - r["majority_macro_f1"] for r in rs]))
            ok = acc_gain > d["min_accuracy_gain"] and f1_gain > 0
            passing += int(ok)
            f1_passing += int(f1_gain > 0)
            detail[int(rot)] = {
                "accuracy": float(np.mean([r["accuracy"] for r in rs])),
                "majority_accuracy": float(rs[0]["majority_accuracy"]),
                "accuracy_gain": acc_gain,
                "macro_f1": float(np.mean([r["macro_f1"] for r in rs])),
                "majority_macro_f1": float(rs[0]["majority_macro_f1"]),
                "macro_f1_gain": f1_gain,
                "recall_class0": float(np.mean([r["recall_class0"] for r in rs])),
                "recall_class1": float(np.mean([r["recall_class1"] for r in rs])),
                "recall_class2": float(np.mean([r["recall_class2"] for r in rs])),
                "n_test": int(rs[0]["n_test"]),
                "n_classes_predicted": int(rs[0]["n_classes_predicted"]),
                "pass": bool(ok)}
        return detail, passing, f1_passing

    detail, passing, f1_passing = tally(best)
    informative = passing >= d["min_rotations_passing"]

    per_cell = {}
    secondary_pass = []
    for cell in carried:
        cd, cp, cf = tally(cell)
        name = f"{cell[0]}|{cell[1]}|{cell[2]}"
        per_cell[name] = {
            "role": "best_cell" if cell == best else "accuracy_candidate",
            "rotation0_accuracy_gain": cd[0]["accuracy_gain"],
            "rotation0_macro_f1": cd[0]["macro_f1"],
            "rotations_passing_both_conditions": int(cp),
            "rotations_with_positive_macro_f1_gain": int(cf),
            "per_rotation": cd}
        if (cd[0]["accuracy_gain"] > d["min_accuracy_gain"]
                and cf >= d["min_rotations_passing"]):
            secondary_pass.append(name)
    secondary = bool(secondary_pass)

    out = {
        "protocol": proto["_name"], "protocol_sha256": proto["_sha256"],
        "best_cell": {"representation": best[0], "class_handling": best[1],
                      "classifier": best[2],
                      "rotation0_mean_macro_f1": float(np.mean(agg[best]))},
        "rotations_passing": int(passing), "required": int(d["min_rotations_passing"]),
        "text_informative_under_tested_representation": bool(informative),
        "primary_rule": d["as_implemented"],
        "secondary_reading": {
            "note": proto["decision_rule_secondary"]["statement"],
            "cells_satisfying": secondary_pass,
            "outcome": secondary,
            "best_cell_rotation0_accuracy_gain": detail[0]["accuracy_gain"],
            "best_cell_rotations_with_positive_macro_f1_gain": int(f1_passing),
        },
        "carried_cells": per_cell,
        "per_rotation": detail,
    }
    with open(os.path.join(RESULTS, "itemA_decision_rev1.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))
    print("\nMANUSCRIPT: update Section VIII-A, the abstract sentence on predictive signal, "
          "Section IX check 6 and the Conclusion from itemA_decision_rev1.json.")


if __name__ == "__main__":
    main()
