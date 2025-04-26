# tune_leaf_purity_regressor.py
"""Single‑tree **XGBRegressor** tuner for leaf‑partitioning

This version trains exactly **one regression tree** (objective="reg:squarederror")
and optimises the trade‑off between

    • **hit‑rate** – fraction of validation rows whose `category_id` appears in
      the training set of the same leaf;
    • **CV** – coefficient of variation of training leaf sizes (smaller ⇒ more
      even buckets).

Score maximised:  `S = hit_rate − λ·CV`.

Outputs
-------
* `xgboost_leaf_partitioner.json`  – trained single‑tree regressor
* `train_leaf_ids.npy`, `val_leaf_ids.npy`, `test_leaf_ids.npy` – 1‑D arrays of
  leaf indices (shape = n_rows).
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

# ----------------------------------------------------------------------------
# Columns / paths (same 10‑feature subset)
# ----------------------------------------------------------------------------
FEATURE_COLS = [
    "habitat", "countryCode", "metaSubstrate", "landcover",
    "biogeographicalRegion", "latitude", "longitude", "region",
    "district", "elevation",
]
NUMERIC_COLS = ["latitude", "longitude", "elevation"]
TARGET_COL = "category_id"
TARGET_COL2 = "scientificName"
MODEL_PATH = "xgboost_leaf_partitioner.json"
TRAIN_LEAF_PATH = "train_leaf_ids.npy"
VAL_LEAF_PATH   = "val_leaf_ids.npy"
TEST_LEAF_PATH  = "test_leaf_ids.npy"

# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def load_csv(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p.resolve())
    return pd.read_csv(p)

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    keep = FEATURE_COLS + [TARGET_COL, TARGET_COL2]
    keep_existing = [col for col in keep if col in df.columns]
    df = df[keep_existing].copy()
    df[NUMERIC_COLS] = df[NUMERIC_COLS].fillna(-999)
    cat_cols = [c for c in FEATURE_COLS if c not in NUMERIC_COLS]
    df[cat_cols] = df[cat_cols].fillna("missing").astype(str)
    df = pd.get_dummies(df, columns=cat_cols, dummy_na=False)
    return df

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main(args):
    # 0  Load & preprocess ----------------------------------------------------------------
    train_df = load_csv(args.train_csv)
    val_df   = load_csv(args.val_csv)
    test_df  = load_csv(args.test_csv)

    combined = pd.concat([train_df, val_df, test_df], ignore_index=True)
    processed = preprocess(combined)

    n_train, n_val = len(train_df), len(val_df)
    train_proc = processed.iloc[:n_train]
    val_proc   = processed.iloc[n_train:n_train + n_val]
    test_proc  = processed.iloc[n_train + n_val:]

    # label encode target for regression (numeric already, but ensure 0‑..)
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[TARGET_COL])
    y_val   = le.transform(val_df[TARGET_COL])

    X_train = train_proc.drop(columns=[TARGET_COL, TARGET_COL2])
    X_val   = val_proc.drop(columns=[TARGET_COL, TARGET_COL2])
    X_test  = test_proc.drop(columns=[TARGET_COL, TARGET_COL2])

    dtrain, dval, dtest = map(xgb.DMatrix, (X_train, X_val, X_test))

    λ = args.lambda_balance

    # 1  Optuna objective -----------------------------------------------------
    def objective(trial):
        params = {
            "n_estimators": 1,
            "objective": "reg:squarederror",
            "tree_method": "hist",
            "grow_policy": "lossguide",
            "learning_rate": 1.0,
            "max_depth": trial.suggest_int("max_depth", 10, 18, 2),
            "max_leaves": trial.suggest_categorical("max_leaves", [128, 256, 512, 1024]),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 50, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 1.0),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "verbosity": 0,
        }
        reg = xgb.XGBRegressor(**params)
        reg.fit(X_train, y_train)
        booster = reg.get_booster()

        leaf_train = booster.predict(dtrain, pred_leaf=True).astype(int).ravel()
        leaf_val   = booster.predict(dval,   pred_leaf=True).astype(int).ravel()

        # map leaf -> species set using original category ids
        bag: defaultdict[int, set[int]] = defaultdict(set)
        for l, cid in zip(leaf_train, y_train):
            bag[l].add(int(cid))

        hits = sum(1 for l, cid in zip(leaf_val, y_val) if cid in bag[l])
        hit_rate = hits / len(y_val)

        counts = np.bincount(leaf_train)
        cv = counts.std() / counts.mean() if counts.mean() else 0.0

        trial.set_user_attr("hit_rate", hit_rate)
        trial.set_user_attr("cv", cv)
        score = hit_rate - λ * cv
        # CV: smaller = more even
        # hit_rate: larger = more hits, more ground truth in leaf
        # score: larger = better
        overcrowd = (counts > 5 * counts.mean()).mean()   # fraction of leaves >5×mean

        # ---------- weighted score ----------------------
        W_HIT      = 4.0        # make this large
        W_CV       = 0.2
        W_CROWD    = 0.2

        score = ( -W_HIT   * hit_rate
                + W_CV   * cv
                + W_CROWD* overcrowd )
        print(f"Trial {trial.number}: hit_rate = {hit_rate*100:.4f}%,  CV = {cv:.4f},  score = {score:.4f}")
        return score            # Optuna minimises

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = study.best_trial
    print(f"\n✅ Best hit   = {best.user_attrs['hit_rate']:.4f}")
    print(f"✅ Best CV    = {best.user_attrs['cv']:.4f}")
    print("✅ Params    =", best.params)

    # 2  Retrain best model & save artefacts ----------------------------------
    best_params = {**best.params, 
        "n_estimators": 1,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "grow_policy": "lossguide",
        "learning_rate": 1.0,
        "verbosity": 0,
    }
    reg_best = xgb.XGBRegressor(**best_params).fit(X_train, y_train)
    booster  = reg_best.get_booster()

    leaf_train = booster.predict(dtrain, pred_leaf=True).astype(int).ravel()
    leaf_val   = booster.predict(dval,   pred_leaf=True).astype(int).ravel()
    leaf_test  = booster.predict(dtest,  pred_leaf=True).astype(int).ravel()

    reg_best.save_model(MODEL_PATH)
    np.save(TRAIN_LEAF_PATH, leaf_train)
    np.save(VAL_LEAF_PATH,   leaf_val)
    np.save(TEST_LEAF_PATH,  leaf_test)

    print(f"\n💾 Saved model        -> {MODEL_PATH}")
    print(f"💾 train_leaf_ids.npy -> {TRAIN_LEAF_PATH}   (shape {leaf_train.shape})")
    print(f"💾 val_leaf_ids.npy   -> {VAL_LEAF_PATH}     (shape {leaf_val.shape})")
    print(f"💾 test_leaf_ids.npy  -> {TEST_LEAF_PATH}    (shape {leaf_test.shape})")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune single-tree XGBRegressor for leaf partitioning.")
    parser.add_argument("--train_csv", default="/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv")
    parser.add_argument("--val_csv",   default="/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv")
    parser.add_argument("--test_csv",   default="/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv")
    parser.add_argument("--n_trials",  type=int, default=50)
    parser.add_argument("--lambda_balance", type=float, default=0.2,
                   help="Penalty weight for imbalance (CV)")
    main(parser.parse_args())
