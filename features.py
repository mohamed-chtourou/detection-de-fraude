import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MISSING_VAL = -999
TARGET = "isFraud"

# =========================
# 0) Utils
# =========================
os.makedirs("figures", exist_ok=True)

def safe_hist(series, title, fname, bins=60, logx=False):
    s = series.replace(MISSING_VAL, np.nan).dropna()
    if len(s) == 0:
        print(f"⚠️ Plot skip (empty): {title}")
        return

    plt.figure(figsize=(10, 4))
    if logx:
        # log1p pour éviter log(0)
        s = np.log1p(s)
        plt.hist(s, bins=bins)
        plt.title(title + " (log1p)")
    else:
        plt.hist(s, bins=bins)
        plt.title(title)

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=200)
    plt.close()

def safe_bar(x, y, title, xlabel, ylabel, fname):
    plt.figure(figsize=(10, 4))
    plt.bar(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=200)
    plt.close()

def plot_top_counts(series, title, fname, top=20):
    s = series.fillna("NA").astype(str)
    vc = s.value_counts().head(top)
    plt.figure(figsize=(10, 5))
    plt.bar(vc.index.astype(str), vc.values)
    plt.xticks(rotation=45, ha="right")
    plt.title(title)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"figures/{fname}", dpi=200)
    plt.close()

# =========================
# 1) Load
# =========================
train = pd.read_parquet("train_step2_preprocessed.parquet")
print("✅ Chargé :", train.shape)

# Échantillon pour visualisations (plus rapide)
visu = train.sample(min(60000, len(train)), random_state=42).copy()

# ======================================================
# 2) Heure + alertFeature + VISU  ✅ FIX IMPORTANT
# ======================================================
train["hour"] = ((train["TransactionDT"] // 3600) % 24).astype(np.int8)

# % fraude par heure (en %)
fraud_by_hour = train.groupby("hour")[TARGET].mean() * 100

# ✅ FIX: alertFeature basé sur QUANTILES (robuste, assure 0-1-2-3)
q1 = fraud_by_hour.quantile(0.25)
q2 = fraud_by_hour.quantile(0.50)
q3 = fraud_by_hour.quantile(0.75)

def get_alert_level(h):
    p = fraud_by_hour.get(h, 0.0)
    if p >= q3:
        return 3
    elif p >= q2:
        return 2
    elif p >= q1:
        return 1
    else:
        return 0

train["alertFeature"] = train["hour"].apply(get_alert_level).astype(np.int8)

# VISU 1
safe_bar(
    x=fraud_by_hour.index,
    y=fraud_by_hour.values,
    title="Fraude (%) par heure",
    xlabel="Heure",
    ylabel="Fraude (%)",
    fname="01_fraud_by_hour.png"
)

# VISU 2
alert_counts = train["alertFeature"].value_counts().sort_index()
safe_bar(
    x=alert_counts.index.astype(str),
    y=alert_counts.values,
    title="Distribution alertFeature",
    xlabel="Niveau d'alerte (0-3)",
    ylabel="Nombre",
    fname="02_alertFeature_dist.png"
)

print("✅ Step hour + alertFeature OK")

# ======================================================
# 3) UID (1 seul) + agrégations MINIMALES + VISU
# ======================================================
train["uid1"] = train["card1"].astype(str) + "_" + train["addr1"].astype(str)

tmp = train[["uid1", "TransactionAmt"]].copy()
tmp["TransactionAmt"] = tmp["TransactionAmt"].replace(MISSING_VAL, np.nan)

agg = tmp.groupby("uid1")["TransactionAmt"].agg(["mean", "std", "count"]).reset_index()
agg.columns = ["uid1", "uid_amt_mean", "uid_amt_std", "uid_amt_count"]
train = train.merge(agg, on="uid1", how="left")

train[["uid_amt_mean", "uid_amt_std"]] = train[["uid_amt_mean", "uid_amt_std"]].fillna(MISSING_VAL).astype(np.float32)
train["uid_amt_count"] = train["uid_amt_count"].fillna(0).astype(np.int16)

# ✅ FIX VISU: distributions très asymétriques -> log1p
visu_uid = train.sample(min(60000, len(train)), random_state=42)
safe_hist(visu_uid["uid_amt_count"].astype(np.float32),
          "UID: nombre de transactions (uid_amt_count)",
          "03_uid_amt_count.png",
          bins=60,
          logx=True)

print("✅ Step UID aggs OK")

# ======================================================
# 4) Features temporelles UID (rank, timedelta, first_tran) + VISU
# ======================================================
train = train.sort_values(["uid1", "TransactionDT"])
train["uid_rank"] = train.groupby("uid1").cumcount().astype(np.int16)

train["uid_timedelta"] = train.groupby("uid1")["TransactionDT"].diff().fillna(0).astype(np.float32)

first_dt = train.groupby("uid1")["TransactionDT"].transform("min")
train["first_tran"] = (train["TransactionDT"] - first_dt).astype(np.float32)

visu_t = train.sample(min(60000, len(train)), random_state=42)
# ✅ log1p indispensable (timedelta très skew)
safe_hist(visu_t["uid_timedelta"], "UID: temps depuis dernière transaction (uid_timedelta)", "04_uid_timedelta.png", bins=80, logx=True)
safe_hist(visu_t["first_tran"], "UID: temps depuis première transaction (first_tran)", "05_first_tran.png", bins=80, logx=True)

print("✅ Step temporel UID OK")

# ======================================================
# 5) Anomalies UID (TransactionAmt uniquement) + VISU
# ======================================================
train["uid_amt_dev"] = MISSING_VAL
valid = (train["TransactionAmt"] != MISSING_VAL) & (train["uid_amt_mean"] != MISSING_VAL)
train.loc[valid, "uid_amt_dev"] = (train.loc[valid, "TransactionAmt"] - train.loc[valid, "uid_amt_mean"]).astype(np.float32)

train["uid_amt_z"] = MISSING_VAL
valid2 = valid & (train["uid_amt_std"] > 0) & (train["uid_amt_std"] != MISSING_VAL)
train.loc[valid2, "uid_amt_z"] = (train.loc[valid2, "uid_amt_dev"] / train.loc[valid2, "uid_amt_std"]).astype(np.float32)

visu_a = train.sample(min(60000, len(train)), random_state=42)
safe_hist(visu_a["uid_amt_z"], "UID: z-score TransactionAmt (uid_amt_z)", "06_uid_amt_z.png", bins=80)

print("✅ Step anomalies UID OK")

# ======================================================
# 6) Outlier flags (MAX 10 colonnes) + VISU
# ======================================================
candidate_cols = ["TransactionAmt_log", "TransactionAmt", "C1", "C2", "C4", "C6", "C8", "D1n", "D2", "D15"]
candidate_cols = [c for c in candidate_cols if c in train.columns][:10]

iqr_k = 1.5
flags = {}

for c in candidate_cols:
    s = train[c].replace(MISSING_VAL, np.nan).astype(np.float32)
    q1 = np.nanpercentile(s, 25)
    q3 = np.nanpercentile(s, 75)
    iqr = q3 - q1
    low = q1 - iqr_k * iqr
    high = q3 + iqr_k * iqr
    flags[f"is_outlier_{c}"] = ((s < low) | (s > high)).fillna(False).astype(np.int8)

train = pd.concat([train, pd.DataFrame(flags, index=train.index)], axis=1)

out_counts = {k: int(train[k].sum()) for k in flags.keys()}
plt.figure(figsize=(10, 4))
plt.bar(list(out_counts.keys()), list(out_counts.values()))
plt.xticks(rotation=45, ha="right")
plt.title("Nombre d'outliers détectés (flags)")
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("figures/07_outlier_flags_counts.png", dpi=200)
plt.close()

print("✅ Step outliers OK")

# ======================================================
# 7) Device hash + uid_device_nunique + device_uid_nunique + VISU
# ======================================================
device_cols = [c for c in ["DeviceType", "DeviceInfo", "id_30", "id_31", "id_32", "id_33"] if c in train.columns]

if len(device_cols) > 0:
    dev_str = train[device_cols].fillna("NA").astype(str).agg("|".join, axis=1)

    # ✅ FIX: hash stable et compact
    train["device_hash"] = pd.util.hash_pandas_object(dev_str, index=False).astype(np.uint64)

    tmpd = train.loc[train["device_hash"].notna(), ["uid1", "device_hash"]].copy()

    uid_device = tmpd.groupby("uid1")["device_hash"].nunique().astype(np.int16)
    train["uid_device_nunique"] = train["uid1"].map(uid_device).fillna(0).astype(np.int16)

    dev_uid = tmpd.groupby("device_hash")["uid1"].nunique().astype(np.int16)
    train["device_uid_nunique"] = train["device_hash"].map(dev_uid).fillna(0).astype(np.int16)

    vdev = train.sample(min(60000, len(train)), random_state=42)
    safe_hist(vdev["uid_device_nunique"].astype(np.float32), "UID: nb devices uniques (uid_device_nunique)", "08_uid_device_nunique.png", bins=50, logx=True)
    safe_hist(vdev["device_uid_nunique"].astype(np.float32), "Device: nb UIDs uniques (device_uid_nunique)", "09_device_uid_nunique.png", bins=50, logx=True)
else:
    train["device_hash"] = MISSING_VAL
    train["uid_device_nunique"] = 0
    train["device_uid_nunique"] = 0
    print("⚠️ device_cols manquantes -> device features ignorées")

print("✅ Step device OK")

# ======================================================
# 8) Behaviour: open_card + decimal_digit + VISU
# ======================================================
if "D1n" in train.columns:
    train["open_card"] = train["D1n"].astype(np.float32)
elif "D1" in train.columns and "DT_days" in train.columns:
    valid = (train["D1"] != MISSING_VAL) & (train["DT_days"] != MISSING_VAL)
    train["open_card"] = MISSING_VAL
    train.loc[valid, "open_card"] = (train.loc[valid, "D1"].astype(np.float32) - train.loc[valid, "DT_days"].astype(np.float32))
else:
    train["open_card"] = MISSING_VAL

# ✅ FIX: decimal_digit sans casser float (méthode robuste)
amt = train["TransactionAmt"].replace(MISSING_VAL, np.nan)
amt_str = amt.fillna(0).map(lambda x: f"{x:.3f}")   # limite à 3 décimales max (suffisant)
train["decimal_digit"] = amt_str.str.split(".").str[1].str.rstrip("0").str.len().astype(np.int8)

vb = train.sample(min(60000, len(train)), random_state=42)
safe_hist(vb["open_card"], "open_card (proxy âge compte)", "10_open_card.png", bins=80, logx=True)
safe_hist(vb["decimal_digit"].astype(np.float32), "decimal_digit (nb décimales TransactionAmt)", "11_decimal_digit.png", bins=10)

print("✅ Step behaviour OK")

# ======================================================
# 9) email_domain_comp + VISU
# ======================================================
if "P_emaildomain" in train.columns and "R_emaildomain" in train.columns:
    p = train["P_emaildomain"].fillna("NA").astype(str)
    r = train["R_emaildomain"].fillna("NA").astype(str)
    train["email_domain_comp"] = (p == r).astype(np.int8)

    vc = train["email_domain_comp"].value_counts().sort_index()
    safe_bar(vc.index.astype(str), vc.values, "email_domain_comp (0/1)", "Match", "Count", "12_email_domain_comp.png")
else:
    train["email_domain_comp"] = 0
    print("⚠️ colonnes email manquantes -> email_domain_comp ignorée")

print("✅ Step email OK")

# ======================================================
# 10) Feature Cross + Label Encoding (léger) + VISU
# ======================================================
cross_pairs = [("card1", "card5"), ("addr1", "P_emaildomain")]
cross_cols = []

for a, b in cross_pairs:
    if a in train.columns and b in train.columns:
        newc = f"{a}__{b}"
        train[newc] = train[a].fillna("NA").astype(str) + "_" + train[b].fillna("NA").astype(str)
        cross_cols.append(newc)

# label encoding (simple, stable)
for col in cross_cols:
    vals = train[col].fillna("NA").astype(str)
    uniq = vals.unique()
    mapping = {v: i for i, v in enumerate(uniq)}
    train[col + "_label"] = vals.map(mapping).astype(np.int32)

    v = train.sample(min(50000, len(train)), random_state=42)[col]
    plot_top_counts(v, f"Top catégories {col} (feature cross)", f"13_top_{col}.png", top=20)

print(f"✅ Step feature cross OK: {cross_cols}")

# ======================================================
# 11) CountEncoding (léger) + VISU
# ======================================================
count_cols = [c for c in ["ProductCD", "card1", "addr1", "P_emaildomain", "device_hash"] if c in train.columns]
count_feats = {}

for col in count_cols:
    vc = train[col].value_counts(dropna=False)
    newc = f"{col}_count_full"
    count_feats[newc] = train[col].map(vc).fillna(0).astype(np.int32)

train = pd.concat([train, pd.DataFrame(count_feats, index=train.index)], axis=1)

# ✅ log1p conseillé (count encoding très skew)
for col in list(count_feats.keys())[:2]:
    v = train.sample(min(60000, len(train)), random_state=42)[col].astype(np.float32)
    safe_hist(v, f"{col} (CountEncoding)", f"14_{col}.png", bins=80, logx=True)

print("✅ Step count encoding OK")

# ======================================================
# 12) Agrégations groupées (mean/std) + VISU
# ======================================================
group_aggs = [("C1", "card1"), ("TransactionAmt", "card1"), ("TransactionAmt", "addr1")]
new_aggs = []

for num_col, grp_col in group_aggs:
    if num_col in train.columns and grp_col in train.columns:
        base = train[[grp_col, num_col]].copy()
        base[num_col] = base[num_col].replace(MISSING_VAL, np.nan).astype(np.float32)

        g = base.groupby(grp_col)[num_col].agg(["mean", "std"]).reset_index()
        g.columns = [grp_col, f"{num_col}_{grp_col}_mean", f"{num_col}_{grp_col}_std"]
        train = train.merge(g, on=grp_col, how="left")

        new_aggs += [f"{num_col}_{grp_col}_mean", f"{num_col}_{grp_col}_std"]

train[new_aggs] = train[new_aggs].fillna(MISSING_VAL).astype(np.float32)

if "TransactionAmt_card1_mean" in train.columns and "TransactionAmt" in train.columns:
    vs = train.sample(min(30000, len(train)), random_state=42)
    x = vs["TransactionAmt"].replace(MISSING_VAL, np.nan)
    y = vs["TransactionAmt_card1_mean"].replace(MISSING_VAL, np.nan)
    m = x.notna() & y.notna()

    plt.figure(figsize=(6, 6))
    plt.scatter(x[m], y[m], s=5, alpha=0.25)
    plt.xlabel("TransactionAmt")
    plt.ylabel("TransactionAmt_card1_mean")
    plt.title("Group agg: montant vs moyenne (par card1)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("figures/15_groupagg_scatter.png", dpi=200)
    plt.close()

print("✅ Step grouped aggs OK")

# ======================================================
# 13) Nettoyage + sauvegarde
# ======================================================
train = train.sort_index()
train = train.drop(columns=["uid1"])  # UID pas nécessaire pour le modèle

# Convert object -> string avant parquet (ok)
obj_cols = train.select_dtypes(include=["object"]).columns
if len(obj_cols) > 0:
    train[obj_cols] = train[obj_cols].astype(str)

train.to_parquet("train_step3_features_light_plus.parquet", index=False)

print("✅ Sauvegardé : train_step3_features_light_plus.parquet")
print("Shape:", train.shape)
print("✅ Figures sauvegardées dans ./figures/")
