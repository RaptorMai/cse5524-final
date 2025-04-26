import pandas as pd
import numpy as np
from math import pi
from collections import defaultdict
import xgboost as xgb
from hyperopt import fmin, tpe, hp, Trials, STATUS_OK
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes

def geo_to_cartesian(lat, lon):
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = np.cos(lat_rad) * np.cos(lon_rad)
    y = np.cos(lat_rad) * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return pd.DataFrame({'x': x, 'y': y, 'z': z})

def month_to_cyclical(month):
    radians = 2 * pi * (month % 12) / 12
    return pd.DataFrame({
        'month_sin': np.sin(radians),
        'month_cos': np.cos(radians)
    })

# Load datasets
train_df = pd.read_csv("/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv")
val_df = pd.read_csv("/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv")
test_df = pd.read_csv("/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv")
target = 'category_id'
save_path = 'xg/xgb_regressor_partition.json'

combined_df = pd.concat([train_df, val_df], axis=0)

# --- Smart substrate mapping ---
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

# Spatial + temporal features
combined_df[['x', 'y', 'z']] = geo_to_cartesian(combined_df['latitude'], combined_df['longitude'])
combined_df[['month_sin', 'month_cos']] = month_to_cyclical(combined_df['month'])

# Final categorical handling
categorical_cols = ['habitat', 'metaSubstrate', 'landcover', 'biogeographicalRegion', 'substrate_grouped', '']
low_card = [col for col in categorical_cols if combined_df[col].nunique() <= 32]
combined_df = pd.get_dummies(combined_df, columns=low_card)
combined_df = combined_df.fillna(-999)

train_encoded = combined_df.iloc[:len(train_df)].copy()
val_encoded = combined_df.iloc[len(train_df):len(train_df) + len(val_df)].copy()
# test_encoded = combined_df.iloc[len(train_df) + len(val_df):].copy()

species_encoder = LabelEncoder()
train_encoded[target] = species_encoder.fit_transform(train_encoded[target])
val_encoded[target] = species_encoder.transform(val_encoded[target])

feature_cols = [col for col in test_encoded.columns if col not in ['category_id', 'scientificName'] and ptypes.is_numeric_dtype(test_encoded[col])]
X_train = train_encoded[feature_cols]
X_val = val_encoded[feature_cols]
y_train = train_encoded[target].astype(float)
y_val = val_encoded[target].astype(float)

dtrain = xgb.DMatrix(X_train)
dval = xgb.DMatrix(X_val)

def compute_leaf_score(leaf_id, species, leaf_species_group):
    species_list = leaf_species_group.get(leaf_id, [])
    if not species_list or species not in species_list:
        return 0.0
    return species_list.count(species) / len(species_list)

def objective(params):
    model = xgb.XGBRegressor(
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
        reg_alpha=params['alpha']
    )

    model.fit(X_train, y_train)
    booster = model.get_booster()

    train_leaves = booster.predict(dtrain, pred_leaf=True)
    val_leaves = booster.predict(dval, pred_leaf=True)

    leaf_species_group = defaultdict(list)
    for leaf, species in zip(train_leaves, y_train.astype(int)):
        leaf_species_group[leaf].append(species)

    val_leaf_counts = defaultdict(int)
    val_scores = []

    for leaf, species in zip(val_leaves, y_val.astype(int)):
        val_leaf_counts[leaf] += 1
        val_scores.append(compute_leaf_score(leaf, species, leaf_species_group))

    avg_score = np.mean(val_scores)
    zero_score_count = val_scores.count(0.0)
    print(f"\n⚠️ Number of validation samples with score 0.0 (best tree): {zero_score_count} out of {len(val_scores)}")

    print(f"\nLeaf Node Distribution (Val Set):")
    for leaf_id, count in sorted(val_leaf_counts.items()):
        print(f"  Leaf {leaf_id}: {count} validation samples")

    leaf_species_diversity = {leaf: len(set(species_list)) for leaf, species_list in leaf_species_group.items()}
    print("\n✅ Training Leaf Species Diversity:")
    for leaf_id, num_species in sorted(leaf_species_diversity.items()):
        print(f"  Leaf {leaf_id}: {num_species} unique species")

    return {
        'loss': -avg_score,
        'status': STATUS_OK,
        'avg_score': avg_score,
        'params': params,
        'val_leaf_counts': dict(val_leaf_counts),
        'train_leaf_diversity': leaf_species_diversity
    }

space = {
    'max_depth': hp.choice('max_depth', [6, 10, 14]),
    'min_child_weight': hp.choice('min_child_weight', [10, 20, 30]),
    'gamma': hp.uniform('gamma', 0.5, 2.0),
    'subsample': hp.uniform('subsample', 0.6, 0.9),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 0.9),
    'max_leaves': hp.choice('max_leaves', [32, 64, 128]),
    'lambda': hp.uniform('lambda', 1.0, 5.0),
    'alpha': hp.uniform('alpha', 0.0, 1.0)
}

trials = Trials()
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=50, trials=trials)

best_trial = sorted(trials.results, key=lambda x: -x['avg_score'])[0]
print("\n✅ Best Hyperopt Params (Regressor):", best_trial['params'])
print(f"✅ Best Avg Leaf Score: {best_trial['avg_score']:.4f}")

print("\n✅ Final Leaf Counts (Best Trial):")
for leaf_id, count in sorted(best_trial['val_leaf_counts'].items()):
    print(f"  Leaf {leaf_id}: {count} validation samples")

print("\n✅ Final Training Leaf Species Diversity:")
for leaf_id, count in sorted(best_trial['train_leaf_diversity'].items()):
    print(f"  Leaf {leaf_id}: {count} unique species in training")

best_params  = best_trial['params']

model = xgb.XGBRegressor(
    n_estimators=1,
    learning_rate=1.0,
    tree_method='hist',
    grow_policy='lossguide',
    max_depth=best_params['max_depth'],
    min_child_weight=best_params['min_child_weight'],
    gamma=best_params['gamma'],
    subsample=best_params['subsample'],
    colsample_bytree=best_params['colsample_bytree'],
    max_leaves=best_params['max_leaves'],
    reg_lambda=best_params['lambda'],
    reg_alpha=best_params['alpha']
)

model.fit(X_train, y_train)
model.fit(X_val, y_val)
model.save_model(save_path)
print(f'Model saved to {save_path}')
