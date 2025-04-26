# tune_leaf_purity.py
"""Hyper‑parameter tuner for a *single‑tree* XGBoost leaf‑partitioner
that balances **leaf‑purity hit‑rate** and **leaf size evenness**.

Score maximised:
    S = hit_rate − λ * coeff_of_variation(leaf_sizes)

* hit_rate  – fraction of validation rows whose ``category_id`` exists among
              the training ``category_id`` set of the same leaf.
* cv        – std(counts) / mean(counts) on the *training* leaf‑size counts.
* λ (lambda_balance) tunes the trade‑off (default 0.2).

Outputs
-------
* best parameters + score to stdout
* ``xgboost_leaf_partitioner.json``   – trained single‑tree model with best params
* ``train_leaf_ids.npy`` / ``val_leaf_ids.npy`` – NumPy arrays of leaf indices
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Configuration – feature / target columns (same as trainXGRe_leaves.py)
# ---------------------------------------------------------------------------
FEATURE_COLS = [
    "habitat",
    "countryCode",
    "metaSubstrate",
    "landcover",
    "biogeographicalRegion",
    "latitude",
    "longitude",
    "region",
    "district",
    "elevation",
]
NUMERIC_COLS = ["latitude", "longitude", "elevation"]
TARGET_COL = "category_id"
MODEL_PATH = "xgboost_leaf_partitioner.json"
TRAIN_LEAF_PATH = "train_leaf_ids.npy"
VAL_LEAF_PATH = "val_leaf_ids.npy"
TEST_LEAF_PATH = "test_leaf_ids.npy"

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.resolve())
    return pd.read_csv(p)


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    keep_cols = FEATURE_COLS + [TARGET_COL]
    df = df[keep_cols].copy()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(-999)
    cat_cols = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]
    df[cat_cols] = df[cat_cols].fillna("missing").astype(str)
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    return df

# ---------------------------------------------------------------------------
# Main tuning routine
# ---------------------------------------------------------------------------

def main(args):
    print("➡️  Loading metadata …")
    train_df = load_csv(args.train_csv)
    val_df   = load_csv(args.val_csv)
    test_df  = load_csv(args.test_csv)

    print("➡️  Pre‑processing …")
    combined = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    processed = preprocess(combined)

    n_train = len(train_df)
    n_val   = len(val_df)
    n_test  = len(test_df)
    train_proc = processed.iloc[:n_train].copy()
    val_proc   = processed.iloc[n_train:n_train+n_val].copy()
    test_proc  = processed.iloc[n_train+n_val:].copy()

    # Encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])
    y_val   = le.transform(val_df[TARGET_COL])
    # y_test = le.transform(test_df[TARGET_COL])

    drop_cols = [TARGET_COL]
    X_train = train_proc.drop(columns=drop_cols)
    X_val   = val_proc.drop(columns=drop_cols)
    X_test  = test_proc.drop(columns=drop_cols)

    dtrain = xgb.DMatrix(X_train)
    dval   = xgb.DMatrix(X_val)
    dtest  = xgb.DMatrix(X_val)

    lambda_balance = args.lambda_balance

    # ------------------------------------------------------
    # Optuna objective
    # ------------------------------------------------------
    def objective(trial: optuna.Trial):
        params = {
            "n_estimators": 1,
            "objective": "multi:softmax",
            "num_class": len(le.classes_),
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "learning_rate": 1.0,  # single tree, lr irrelevant
            "max_depth": trial.suggest_int("max_depth", 10, 18, step=2),
            "max_leaves": trial.suggest_categorical("max_leaves", [128, 256, 512, 1024]),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "verbosity": 0,
        }

        clf = xgb.XGBClassifier(**params)
        clf.fit(X_train, y_train)
        booster = clf.get_booster()

        # Flatten leaf arrays (shape: n_rows × 1) → (n_rows,)
        leaf_train = booster.predict(dtrain, pred_leaf=True).astype(int).squeeze()
        leaf_val   = booster.predict(dval,   pred_leaf=True).astype(int).squeeze()

        leaf_train_single = leaf_train[np.arange(len(y_train)), y_train]     # if multi-class
        bag = defaultdict(set)
        for leaf, cid in zip(leaf_train_single, y_train):
            bag[leaf].add(cid)

        leaf_val_single = leaf_val[np.arange(len(y_val)), y_val]         # if multi-class

        # metrics
        hits = sum(1 for l, cid in zip(leaf_val_single, y_val) if cid in bag[l])
        hit_rate = hits / len(y_val)

        counts = np.bincount(leaf_train_single)
        cv = counts.std() / counts.mean() if counts.mean() else 0.0

        score = hit_rate - lambda_balance * cv
        trial.set_user_attr("hit_rate", hit_rate)
        trial.set_user_attr("cv", cv)
        # CV: smaller = more even
        # hit_rate: larger = more hits, more ground truth in leaf
        # score: larger = better
        print(f"Trial {trial.number}: hit_rate = {hit_rate:.4f},  CV = {cv:.4f},  score = {score:.4f}")
        return -score  # Optuna minimises

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    best_hit = best.user_attrs["hit_rate"]
    best_cv  = best.user_attrs["cv"]
    print("\n✅ Best score:", -best.value)
    print(f"   hit_rate = {best_hit:.4f},  CV = {best_cv:.4f}")
    print("   params   =", best.params)

    # --------------------------------------------------
    # Retrain best model on train set & save artefacts
    # --------------------------------------------------
    best_params = best.params | {
        "n_estimators": 1,
        "objective": "multi:softmax",
        "num_class": len(le.classes_),
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "learning_rate": 1.0,
        "verbosity": 0,
    }
    best_clf = xgb.XGBClassifier(**best_params)
    best_clf.fit(X_train, y_train)
    booster = best_clf.get_booster()

    leaf_train = booster.predict(dtrain, pred_leaf=True).astype(int)
    leaf_val   = booster.predict(dval,   pred_leaf=True).astype(int)
    leaf_test  = booster.predict(dtest,  pred_leaf=True).astype(int)
    leaf_train_single = leaf_train[np.arange(len(y_train)), y_train]     # if multi-class
    leaf_val_single   = leaf_val[np.arange(len(y_val)), y_val]         # if multi-class
    leaf_test_single  = leaf_test[np.arange(len(y_test)), y_test]       # if multi-class

    best_clf.save_model(MODEL_PATH)
    np.save(TRAIN_LEAF_PATH, leaf_train_single)
    np.save(VAL_LEAF_PATH,   leaf_val_single)
    np.save(TEST_LEAF_PATH,  leaf_test_single)

    print(f"\n💾 Model saved to {MODEL_PATH}")
    print(f"💾 Leaf ids (train) -> {TRAIN_LEAF_PATH}")
    print(f"💾 Leaf ids (val)   -> {VAL_LEAF_PATH}")
    print(f"💾 Leaf ids (test)  -> {TEST_LEAF_PATH}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune XGBoost leaf‑purity vs. balance.")
    parser.add_argument("--train_csv", default="/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv")
    parser.add_argument("--val_csv",   default="/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv")
    parser.add_argument("--test_csv",   default="/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv")
    parser.add_argument("--n_trials",  type=int, default=50, help="Optuna trials")
    parser.add_argument("--lambda_balance", type=float, default=0.2,
                        help="Penalty weight for imbalance (CV)")
    args = parser.parse_args()

    main(args)
