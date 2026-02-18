import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

train_trx = pd.read_csv("train_transaction.csv")
train_id  = pd.read_csv("train_identity.csv")

#1 --- 1) Exploration ---
print(train_trx.shape)
print(train_id.shape)
#print(train_trx.head())
#print(train_id.head())
#print(train_trx["isFraud"].value_counts(normalize=True))
#print(train_trx["TransactionAmt"].describe())
#print(train_trx.isnull().mean().sort_values(ascending=False).head(10))

# --- 2) Fusion (LEFT join) ---
train = train_trx.merge(train_id, on="TransactionID", how="left")
print("Shape après fusion :", train.shape)
#print(train.head())

# --- 3) Drop colonnes > 90% NaN (calculé sur train) ---
missing_ratio = train.isna().mean() 
#print(missing_ratio)         # proportion NaN par colonne
cols_to_drop = missing_ratio[missing_ratio > 0.90].index.tolist()
print("Nb colonnes à supprimer (>90% NaN):", len(cols_to_drop))
train = train.drop(columns=cols_to_drop)
print("Shape après suppression colonnes >90% NaN :", train.shape)
train = train.fillna(-999)
print("NaN restants :", int(train.isna().sum().sum()))

# --- 4) Save cleaned data  step1---
obj_cols = train.select_dtypes(include=["object"]).columns
train[obj_cols] = train[obj_cols].astype(str)
train.to_parquet("train_step1_clean.parquet", index=False)

# --- 5) Normalisation temporelle (soustraction) ---
#  Temps en jours (float)
train["DT_days"] = train["TransactionDT"].astype(np.float32) / np.float32(24 * 60 * 60)
#  Colonnes D à transformer
D_cols = ["D1", "D2", "D15"]
#  Transformation par SOUSTRACTION (dtype-safe)
for col in D_cols:
    new_col = col + "n"   # ex: D1n
    
    # créer la colonne en float
    train[new_col] = np.nan
    
    valid = (train[col] != -999) & (train["DT_days"] > 0)
    
    train.loc[valid, new_col] = (
        train.loc[valid, col].astype(np.float32)
        - train.loc[valid, "DT_days"]
    )
    
    # remettre -999 pour les valeurs non calculées
    train[new_col] = train[new_col].fillna(-999)

#--- 6) transformation logarithmique de TransactionAmt ---
train["TransactionAmt_log"] = np.log1p(train["TransactionAmt"])

# --- 7) ENCODAGE DES VARIABLES CATÉGORIELLES ---
print("\n🔄 Encodage catégoriel...")

# Frequency Encoding (haute cardinalité)
freq_cols = ["addr1", "addr2", "card1", "card2", "card3", "card4", 
             "card5", "card6", "P_emaildomain", "R_emaildomain",
             "DeviceType", "DeviceInfo"]
freq_cols = [c for c in freq_cols if c in train.columns]
for col in freq_cols:
    freq_map = train[col].value_counts(normalize=True).to_dict()
    train[col + "_freq"] = train[col].map(freq_map).fillna(-999)
print(f"  ✓ Frequency encoding : {len(freq_cols)} colonnes")

# Label Encoding (faible cardinalité)
label_cols = ["ProductCD", "M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]
label_cols = [c for c in label_cols if c in train.columns]
label_maps = {}  # Stocker les mappings pour affichage
for col in label_cols:
    unique_vals = train[col].unique()
    label_map = {val: idx for idx, val in enumerate(unique_vals)}
    label_maps[col] = label_map
    train[col + "_label"] = train[col].map(label_map).fillna(-999)
print(f"  ✓ Label encoding : {len(label_cols)} colonnes")
print("✅ Encodage terminé.")

# --- 8) Save cleaned data step2 ---
train.to_parquet("train_step2_preprocessed.parquet", index=False)
print("✅ Sauvegardé : train_step2_preprocessed.parquet")
