import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder


MISSING_VAL = -999
TARGET = "isFraud"

train = pd.read_parquet("train_step2_preprocessed.parquet")
print("✅ Chargé :", train.shape)

# ======================================================
# 1) FRAUDE PAR HEURE + alertFeature
# ======================================================
print("\n⏰ Feature temporelle : fraude par heure...")

train["hour"] = ((train["TransactionDT"] // 3600) % 24).astype(np.int8)

fraud_hour = train.groupby("hour")[TARGET].mean() * 100

plt.figure(figsize=(10,4))
plt.bar(fraud_hour.index, fraud_hour.values)
plt.title("Fraude (%) par heure")
plt.xlabel("Heure")
plt.ylabel("Fraude (%)")
plt.grid(axis="y", alpha=0.3)
plt.savefig("01_fraud_by_hour.png")
plt.show()

# alertFeature simple (3 niveaux)
train["alertFeature"] = pd.cut(
    fraud_hour[train["hour"]].values,
    bins=[0, 3, 6, 20],
    labels=[0, 1, 2]
).astype(np.int8)

plt.figure(figsize=(6,4))
train["alertFeature"].value_counts().sort_index().plot(kind="bar")
plt.title("Distribution alertFeature")
plt.xlabel("Niveau d'alerte")
plt.ylabel("Nombre")
plt.grid(axis="y", alpha=0.3)
plt.savefig("02_alertFeature_dist.png")
plt.show()


# ======================================================
# 2) UID UTILISATEUR + AGRÉGATION SIMPLE
# ======================================================
print("\n👤 Création UID + stats TransactionAmt...")

train["UID"] = train["card1"].astype(str) + "_" + train["addr1"].astype(str)

uid_stats = train.groupby("UID")["TransactionAmt"].agg(["mean", "count"]).reset_index()
uid_stats.columns = ["UID", "uid_amt_mean", "uid_amt_count"]

train = train.merge(uid_stats, on="UID", how="left")

plt.figure(figsize=(8,4))
plt.hist(train["uid_amt_count"], bins=50)
plt.title("UID : nombre de transactions")
plt.xlabel("Transactions par UID")
plt.ylabel("Fréquence")
plt.savefig("03_uid_amt_count.png")
plt.show()


# ======================================================
# 3) TEMPS ENTRE TRANSACTIONS (uid_timedelta)
# ======================================================
print("\n⏱️ Temps depuis dernière transaction...")

train = train.sort_values(["UID", "TransactionDT"])
train["uid_timedelta"] = train.groupby("UID")["TransactionDT"].diff().fillna(0)

plt.figure(figsize=(8,4))
plt.hist(train["uid_timedelta"], bins=60)
plt.title("UID : temps depuis dernière transaction")
plt.xlabel("Secondes")
plt.ylabel("Nombre")
plt.savefig("04_uid_timedelta.png")
plt.show()


# ======================================================
# 4) DEVICE HASH + DEVICE FEATURES
# ======================================================
print("\n📱 Identification des appareils...")

device_cols = ["id_30", "id_31", "DeviceType"]
device_cols = [c for c in device_cols if c in train.columns]

train["device_hash"] = train[device_cols].astype(str).agg("_".join, axis=1)

# nb devices par UID
train["uid_device_nunique"] = train.groupby("UID")["device_hash"].transform("nunique")

# nb UID par device
train["device_uid_nunique"] = train.groupby("device_hash")["UID"].transform("nunique")

plt.figure(figsize=(8,4))
plt.hist(train["uid_device_nunique"], bins=50)
plt.title("UID : nb devices uniques")
plt.savefig("05_uid_device_nunique.png")
plt.show()

plt.figure(figsize=(8,4))
plt.hist(train["device_uid_nunique"], bins=50)
plt.title("Device : nb UID uniques")
plt.savefig("06_device_uid_nunique.png")
plt.show()


# ======================================================
# 5) FEATURES COMPORTEMENTALES
# ======================================================
print("\n🧠 Features comportementales...")

# open_card : proxy âge compte
if "D1" in train.columns:
    train["open_card"] = train["TransactionDT"] / 86400 - train["D1"]

plt.figure(figsize=(8,4))
plt.hist(train["open_card"].replace(MISSING_VAL, np.nan).dropna(), bins=60)
plt.title("open_card (proxy âge compte)")
plt.savefig("07_open_card.png")
plt.show()

# decimal_digit
train["decimal_digit"] = train["TransactionAmt"].apply(
    lambda x: len(str(x).split(".")[1]) if "." in str(x) else 0
)

plt.figure(figsize=(6,4))
train["decimal_digit"].value_counts().plot(kind="bar")
plt.title("decimal_digit")
plt.savefig("08_decimal_digit.png")
plt.show()

# email_domain_comp
if "P_emaildomain" in train.columns and "R_emaildomain" in train.columns:
    train["email_domain_comp"] = (
        train["P_emaildomain"] == train["R_emaildomain"]
    ).astype(np.int8)

    train["email_domain_comp"].value_counts().plot(kind="bar")
    plt.title("email_domain_comp (0/1)")
    plt.savefig("09_email_domain_comp.png")
    plt.show()


# ======================================================
# 6) FEATURE CROSS + LABEL ENCODING
# ======================================================
print("\n🔗 Feature Cross...")

cross_cols = [
    ("card1", "card5"),
    ("addr1", "P_emaildomain")
]

for a, b in cross_cols:
    if a in train.columns and b in train.columns:
        name = f"{a}__{b}"
        train[name] = train[a].astype(str) + "_" + train[b].astype(str)

        le = LabelEncoder()
        train[name] = le.fit_transform(train[name])

        top_vals = train[name].value_counts().head(15)

        plt.figure(figsize=(8,4))
        top_vals.plot(kind="bar")
        plt.title(f"Top catégories {name}")
        plt.savefig(f"10_top_{name}.png")
        plt.show()


# ======================================================
# 7) COUNT ENCODING
# ======================================================
print("\n📊 CountEncoding...")

count_cols = ["card1", "ProductCD"]

for c in count_cols:
    if c in train.columns:
        train[f"{c}_count_full"] = train[c].map(train[c].value_counts())

        plt.figure(figsize=(8,4))
        plt.hist(train[f"{c}_count_full"], bins=60)
        plt.title(f"{c}_count_full (CountEncoding)")
        plt.savefig(f"11_{c}_count_full.png")
        plt.show()


# ======================================================
# 8) GROUP AGGREGATION SIMPLE
# ======================================================
print("\n📌 Agrégation groupée par card1...")

if "card1" in train.columns:
    train["TransactionAmt_card1_mean"] = train.groupby("card1")["TransactionAmt"].transform("mean")

    sample = train.sample(3000)

    plt.figure(figsize=(6,6))
    plt.scatter(sample["TransactionAmt"], sample["TransactionAmt_card1_mean"], alpha=0.3)
    plt.title("TransactionAmt vs moyenne (par card1)")
    plt.xlabel("TransactionAmt")
    plt.ylabel("Mean card1")
    plt.savefig("12_groupagg_scatter.png")
    plt.show()


# ======================================================
# 9) SAUVEGARDE
# ======================================================
print("\n💾 Sauvegarde finale...")

drop_cols = ["UID", "device_hash"]
train = train.drop(columns=[c for c in drop_cols if c in train.columns])

train.to_parquet("train_step3_final_features.parquet", index=False)

print("✅ Dataset final prêt pour le modèle !")
print("Shape :", train.shape)
