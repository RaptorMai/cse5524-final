import pandas as pd
import numpy as np
from math import pi
from collections import defaultdict
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
import pandas.api.types as ptypes
import joblib

# === Preprocessing helpers ===
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

# === Load CSVs ===
train_df = pd.read_csv("/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Train.csv")
val_df = pd.read_csv("/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Val.csv")
test_df = pd.read_csv("/research/nfs_chao_209/fungi-clef-2025/metadata/FungiTastic-FewShot/FungiTastic-FewShot-Test.csv")
target = 'category_id'

combined_df = pd.concat([train_df, val_df, test_df], axis=0)

# === Smart substrate grouping ===
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

# === Add features ===
combined_df[['x', 'y', 'z']] = geo_to_cartesian(combined_df['latitude'], combined_df['longitude'])
combined_df[['month_sin', 'month_cos']] = month_to_cyclical(combined_df['month'])

# === One-hot encode low-cardinality categories ===
categorical_cols = ['habitat', 'metaSubstrate', 'landcover', 'biogeographicalRegion', 'substrate_grouped']
low_card = [col for col in categorical_cols if combined_df[col].nunique() <= 32]
combined_df = pd.get_dummies(combined_df, columns=low_card)
combined_df = combined_df.fillna(-999)

# === Split ===
train_encoded = combined_df.iloc[:len(train_df)].copy()
val_encoded = combined_df.iloc[len(train_df):len(train_df) + len(val_df)].copy()
test_encoded = combined_df.iloc[len(train_df) + len(val_df):].copy()

# === Encode labels ===
# Ideally, load the exact label encoder used during training
# But if you don't have it saved, recreate it here
species_encoder = LabelEncoder()
train_encoded[target] = species_encoder.fit_transform(train_encoded[target])
val_encoded[target] = species_encoder.transform(val_encoded[target])

# === Select numeric features ===
feature_cols = [col for col in test_encoded.columns if col not in ['scientificName', 'category_id'] and ptypes.is_numeric_dtype(test_encoded[col])]
X_train = train_encoded[feature_cols]
X_val = val_encoded[feature_cols]
y_train = train_encoded[target].astype(float)
y_val = val_encoded[target].astype(float)

# === Load booster ===
booster = xgb.Booster()
booster.load_model("xgb_regressor_partition.json")

dtrain = xgb.DMatrix(X_train)
dval = xgb.DMatrix(X_val)

train_leaves = booster.predict(dtrain, pred_leaf=True)
val_leaves = booster.predict(dval, pred_leaf=True)

# === Evaluate matching species in same leaf ===
leaf_species_group = defaultdict(list)
for leaf_id, species in zip(train_leaves, y_train.astype(int)):
    leaf_species_group[leaf_id].append(species)

val_scores = []
val_leaf_counts = defaultdict(int)
for leaf_id, true_species in zip(val_leaves, y_val.astype(int)):
    val_leaf_counts[leaf_id] += 1
    species_list = leaf_species_group.get(leaf_id, [])
    if not species_list or true_species not in species_list:
        val_scores.append(0.0)
    else:
        match_score = species_list.count(true_species) / len(species_list)
        val_scores.append(match_score)

# === Report ===
avg_score = np.mean(val_scores)
zero_score_count = val_scores.count(0.0)
print(f"\n📊 Re-Evaluated Performance using saved tree:")
print(f"✅ Avg match ratio per sample: {avg_score:.4f}")
print(f"❌ Samples with zero matching species in leaf: {zero_score_count} / {len(y_val)}")

print(f"\n🌿 Leaf Node Distribution (Val Set):")
for leaf_id, count in sorted(val_leaf_counts.items()):
    print(f"  Leaf {leaf_id}: {count} validation samples")