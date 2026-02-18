import numpy as np
import pandas as pd
import time

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings("ignore")

start_time = time.time()

# =====================================================
# CONFIG
# =====================================================
TARGET = "isFraud"
RANDOM_STATE = 42

print("🚀 Fraud Detection – Kaggle Pipeline (8GB Safe)")

# =====================================================
# 1️⃣ Load
# =====================================================
print("Loading data...")
t0 = time.time()
df = pd.read_parquet("train_step3_features_light_plus.parquet")
print(f"✓ Loaded in {time.time()-t0:.1f}s")

X = df.drop([TARGET, "TransactionID", "TransactionDT"], axis=1, errors="ignore")
y = df[TARGET].astype(int)

print("Shape:", X.shape)
print("Fraud rate:", y.mean()*100)

# =====================================================
# 2️⃣ Encode object → int (Solution B SAFE)
# =====================================================
print("\nEncoding categoricals...")
t0 = time.time()
obj_cols = X.select_dtypes(include=["object"]).columns.tolist()

for col in obj_cols:
    X[col] = X[col].fillna("missing").astype(str)
    uniq = X[col].unique()
    mapping = {v: i for i, v in enumerate(uniq)}
    X[col] = X[col].map(mapping).astype(np.int32)

print(f"✓ Encoded {len(obj_cols)} columns in {time.time()-t0:.1f}s")

# =====================================================
# 3️⃣ Train / Test Split (Holdout Test)
# =====================================================
X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print("Trainval:", X_trainval.shape)
print("Test:", X_test.shape)

# =====================================================
# 4️⃣ OOF Stacking (4-Fold) with Early Stopping
# =====================================================
print("\n🔁 OOF stacking with 4 folds...")
t0 = time.time()
N_SPLITS = 4
EARLY_STOPPING = 50

skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
oof_xgb = np.zeros(len(X_trainval), dtype=np.float32)
oof_cat = np.zeros(len(X_trainval), dtype=np.float32)
xgb_best_iters = []
cat_best_iters = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval), 1):
    print(f"\nFold {fold}/{N_SPLITS}")
    X_tr = X_trainval.iloc[tr_idx]
    X_va = X_trainval.iloc[va_idx]
    y_tr = y_trainval.iloc[tr_idx]
    y_va = y_trainval.iloc[va_idx]

    scale_pos_weight = (y_tr == 0).sum() / (y_tr == 1).sum()

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        tree_method="hist",
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    xgb.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False
    )

    oof_xgb[va_idx] = xgb.predict_proba(X_va)[:, 1]
    xgb_best_iters.append(xgb.n_estimators)
    print(f"XGB fold AUC: {roc_auc_score(y_va, oof_xgb[va_idx]):.4f}")

    cat = CatBoostClassifier(
        iterations=600,
        depth=6,
        learning_rate=0.05,
        auto_class_weights="Balanced",
        eval_metric="AUC",
        verbose=False,
        random_state=RANDOM_STATE,
        thread_count=-1
    )

    cat.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=EARLY_STOPPING,
        verbose=False
    )

    oof_cat[va_idx] = cat.predict_proba(X_va)[:, 1]
    cat_best = cat.get_best_iteration()
    if not cat_best or cat_best < 1:
        cat_best = cat.get_params()["iterations"]
    cat_best_iters.append(cat_best)
    print(f"CAT fold AUC: {roc_auc_score(y_va, oof_cat[va_idx]):.4f}")

print(f"\nOOF XGB AUC: {roc_auc_score(y_trainval, oof_xgb):.4f}")
print(f"OOF CAT AUC: {roc_auc_score(y_trainval, oof_cat):.4f}")
print(f"✓ OOF stacking base models done ({time.time()-t0:.1f}s)")

# =====================================================
# 5️⃣ Train Stacking Model on OOF
# =====================================================
print("\n🔗 Training Stacking Model...")
t0 = time.time()
X_trainval_stack = np.column_stack([oof_xgb, oof_cat])

lgbm = LGBMClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    is_unbalance=True,
    random_state=RANDOM_STATE,
    verbose=-1
)

lgbm.fit(X_trainval_stack, y_trainval)

val_stack = lgbm.predict_proba(X_trainval_stack)[:, 1]
print(f"✓ STACK Trainval AUC (optimistic): {roc_auc_score(y_trainval, val_stack):.4f} ({time.time()-t0:.1f}s)")

# =====================================================
# 6️⃣ Threshold tuning (maximize F1 on Trainval)
# =====================================================
print("\n🎯 Tuning threshold...")
t0 = time.time()
thresholds = np.linspace(0.05, 0.5, 50)
best_t = 0.5
best_f1 = 0

for t in thresholds:
    pred = (val_stack >= t).astype(int)
    f1 = f1_score(y_trainval, pred)
    if f1 > best_f1:
        best_f1 = f1
        best_t = t

print(f"✓ Best threshold: {round(best_t,3)} | F1: {round(best_f1,4)} ({time.time()-t0:.1f}s)")

# =====================================================
# 7️⃣ Refit Base Models on Full Trainval
# =====================================================
print("\n🔄 Refitting base models on full trainval...")
t0 = time.time()
best_xgb_iter = int(np.median(xgb_best_iters))
best_cat_iter = int(np.median(cat_best_iters))
scale_pos_weight = (y_trainval == 0).sum() / (y_trainval == 1).sum()

xgb_final = XGBClassifier(
    n_estimators=best_xgb_iter,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    eval_metric="auc",
    tree_method="hist",
    n_jobs=-1,
    random_state=RANDOM_STATE
)

cat_final = CatBoostClassifier(
    iterations=best_cat_iter,
    depth=6,
    learning_rate=0.05,
    auto_class_weights="Balanced",
    eval_metric="AUC",
    verbose=False,
    random_state=RANDOM_STATE,
    thread_count=-1
)

xgb_final.fit(X_trainval, y_trainval, verbose=False)
cat_final.fit(X_trainval, y_trainval, verbose=False)
print(f"✓ Refit done ({time.time()-t0:.1f}s)")

# =====================================================
# 8️⃣ TEST FINAL SCORE
# =====================================================
test_stack = np.column_stack([
    xgb_final.predict_proba(X_test)[:, 1],
    cat_final.predict_proba(X_test)[:, 1]
])

test_proba = lgbm.predict_proba(test_stack)[:, 1]
test_pred = (test_proba >= best_t).astype(int)

print("\n🏁 FINAL TEST RESULTS")
print("AUC :", roc_auc_score(y_test, test_proba))
print("F1  :", f1_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred))
print("Recall   :", recall_score(y_test, test_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

print(f"\n⏱️  Total execution time: {time.time()-start_time:.1f}s ({(time.time()-start_time)/60:.1f}min)")
