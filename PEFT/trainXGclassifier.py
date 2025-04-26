import pandas as pd
import numpy as np
from math import pi
from collections import defaultdict
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes

# --- Feature engineering helpers ---
def geo_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return pd.DataFrame({'x': x, 'y': y, 'z': z})

def month_to_cyclical(month):
    radians = 2 * pi * (month % 12) / 12
    return pd.DataFrame({'month_sin': np.sin(radians), 'month_cos': np.cos(radians)})

# --- Load datasets ---
train_df = pd.read_csv(
    "/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv"
)
val_df = pd.read_csv(
    "/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv"
)
target = 'category_id'
save_model_path = 'xg/xgb_classifier_partition.json'

# --- Combine for preprocessing ---
combined_df = pd.concat([train_df, val_df], axis=0, ignore_index=True)

# --- Smart substrate grouping ---
substrate_counts = combined_df['substrate'].value_counts()
def map_substrate(sub):
    if pd.isna(sub):
        return 'missing'
    elif substrate_counts.get(sub, 0) >= 100:
        return sub
    elif substrate_counts.get(sub, 0) >= 20:
        return 'other_common'
    else:
        return 'rare'
combined_df['substrate_grouped'] = combined_df['substrate'].apply(map_substrate)

# --- Spatial & temporal ---
combined_df[['x', 'y', 'z']] = geo_to_cartesian(
    combined_df['latitude'], combined_df['longitude']
)
combined_df[['month_sin', 'month_cos']] = month_to_cyclical(combined_df['month'])

# --- Categorical one-hot for low-card features ---
categorical_cols = [
    'habitat', 'metaSubstrate', 'landcover', 'biogeographicalRegion', 'substrate_grouped'
]
low_card = [c for c in categorical_cols if combined_df[c].nunique() <= 32]
combined_df = pd.get_dummies(combined_df, columns=low_card, dummy_na=True)
combined_df = combined_df.fillna(-999)

# --- Encode target labels ---
species_encoder = LabelEncoder()
combined_df[target] = species_encoder.fit_transform(combined_df[target])

# --- Split back into train/val ---
train_encoded = combined_df.iloc[:len(train_df)].reset_index(drop=True)
val_encoded   = combined_df.iloc[len(train_df):].reset_index(drop=True)

# --- Prepare feature matrices ---
feature_cols = [
    c for c in train_encoded.columns
    if c != target and ptypes.is_numeric_dtype(train_encoded[c])
]
X_train = train_encoded[feature_cols]
X_val   = val_encoded[feature_cols]
y_train = train_encoded[target].astype(int)
y_val   = val_encoded[target].astype(int)

# --- DMatrix for leaf extraction ---
dtrain = xgb.DMatrix(X_train)
dval   = xgb.DMatrix(X_val)

# --- Score function per leaf ---
def compute_leaf_score(leaf_id, true_label, leaf_to_labels):
    labels = leaf_to_labels.get(leaf_id, [])
    if len(labels) == 0 or true_label not in labels:
        return 0.0
    return labels.count(true_label) / len(labels)

# --- Hyperopt objective ---
def objective(params):
    clf = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=2427,
        n_estimators=1,
        learning_rate=1.0,
        tree_method='hist',
        grow_policy='lossguide',
        max_depth=int(params['max_depth']),
        min_child_weight=int(params['min_child_weight']),
        gamma=params['gamma'],
        subsample=params['subsample'],
        colsample_bytree=params['colsample_bytree'],
        max_leaves=int(params['max_leaves']),
        reg_lambda=params['lambda'],
        reg_alpha=params['alpha'],
        # use_label_encoder=False,
        eval_metric='mlogloss'
    )
    clf.fit(X_train, y_train)
    booster = clf.get_booster()

    train_leaves = booster.predict(dtrain, pred_leaf=True).flatten()
    val_leaves   = booster.predict(dval, pred_leaf=True).flatten()

    # Map each leaf to list of training labels
    leaf_to_labels = defaultdict(list)
    for leaf_id, lbl in zip(train_leaves, y_train):
        leaf_to_labels[leaf_id].append(int(lbl))

    # Compute per-sample leaf score on validation
    scores = [
        compute_leaf_score(leaf, lbl, leaf_to_labels)
        for leaf, lbl in zip(val_leaves, y_val)
    ]
    avg_score = float(np.mean(scores))

    # Debug info
    zero_count = sum(1 for s in scores if s == 0.0)
    print(f"Zero-score val samples: {zero_count}/{len(scores)}")

    return {
        'loss': -avg_score,
        'status': STATUS_OK,
        'avg_score': avg_score,
        'params': params
    }

# --- Hyperparameter search space ---
space = {
    'max_depth':       hp.choice('max_depth',       [6, 8, 10]),
    'min_child_weight':hp.choice('min_child_weight',[1, 5, 10]),
    'gamma':           hp.uniform('gamma',           0.0, 1.0),
    'subsample':       hp.uniform('subsample',       0.6, 1.0),
    'colsample_bytree':hp.uniform('colsample_bytree',0.6, 1.0),
    'max_leaves':      hp.choice('max_leaves',      [64, 128, 256]),
    'lambda':          hp.uniform('lambda',          1.0, 10.0),
    'alpha':           hp.uniform('alpha',           0.0, 1.0)
}

# --- Run optimization ---
trials = Trials()
best = fmin(
    fn=objective,
    space=space,
    algo=tpe.suggest,
    max_evals=30,
    trials=trials
)

# --- Select best trial ---
best_trial = sorted(trials.results, key=lambda r: -r['avg_score'])[0]
print("Best params:", best_trial['params'])
print(f"Best avg leaf score: {best_trial['avg_score']:.4f}")

# --- Train final classifier on train+val ---
bp = best_trial['params']
final_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(species_encoder.classes_),
    n_estimators=1,
    learning_rate=1.0,
    tree_method='hist',
    grow_policy='lossguide',
    max_depth=int(bp['max_depth']),
    min_child_weight=int(bp['min_child_weight']),
    gamma=bp['gamma'],
    subsample=bp['subsample'],
    colsample_bytree=bp['colsample_bytree'],
    max_leaves=int(bp['max_leaves']),
    reg_lambda=bp['lambda'],
    reg_alpha=bp['alpha'],
    use_label_encoder=False,
    eval_metric='mlogloss'
)
final_clf.fit(
    pd.concat([X_train, X_val], axis=0),
    np.concatenate([y_train, y_val], axis=0)
)
final_clf.save_model(save_model_path)
print(f"Model saved to {save_model_path}")
