import os
import time
import joblib
import warnings
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    RandomizedSearchCV,
    learning_curve,
    RepeatedStratifiedKFold,
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    ConfusionMatrixDisplay,
    roc_curve,
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb

import tensorflow as tf
tf.get_logger().setLevel("ERROR")
from tensorflow import keras
from tensorflow.keras import layers, callbacks


# ─────────────────────────────────────────────────────────────
# 1. LOAD
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("STEP 1: Loading Data...")
print("=" * 55)

ml = pd.read_csv(r"C:\Users\chara\OneDrive\Desktop\Churn\bank_ml_features.csv")
ml.columns = ml.columns.str.strip().str.upper()

print(f"✅ Rows: {len(ml):,} | Cols: {ml.shape[1]}")
print(f"✅ Churn dist:\n{ml['IS_CHURNED'].value_counts()}")
print(f"   Churn rate: {ml['IS_CHURNED'].mean():.2%}")

TARGET = "IS_CHURNED"


# ─────────────────────────────────────────────────────────────
# 2. FEATURES
# ─────────────────────────────────────────────────────────────
print("\nSTEP 2: Feature Engineering...")

df = ml.drop(columns=["CUSTOMER_ID", "CUSTOMERID"], errors="ignore")

CATS = ["GENDER", "GEOGRAPHY", "RFM_SEGMENT"]
for c in CATS:
    if c in df.columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

NUMERIC = [
    c for c in [
        "AGE", "CREDITSCORE", "TENURE", "BALANCE", "NUMOFPRODUCTS",
        "HASCRCARD", "ISACTIVEMEMBER", "ESTIMATEDSALARY",
        "R_SCORE", "F_SCORE", "M_SCORE", "RFM_TOTAL",
        "RECENCY", "FREQUENCY", "MONETARY",
        "BALANCE_SALARY_RATIO", "CREDIT_AGE_RATIO", "BALANCE_PER_PRODUCT",
        "TENURE_AGE_RATIO", "ACTIVE_X_PRODUCTS", "CREDIT_X_ACTIVE",
        "BALANCE_X_ACTIVE", "HAS_BALANCE", "SENIOR_CUSTOMER", "LONG_TENURE",
        "RFM_X_CREDIT", "RFM_X_BALANCE",
        "COHORT_CHURN_RATE", "COHORT_AVG_TENURE", "COHORT_SIZE"
    ]
    if c in df.columns
]

# Additional engineered features
if all(col in df.columns for col in ["BALANCE", "CREDITSCORE"]):
    df["BALANCE_X_CREDIT"] = df["BALANCE"] * df["CREDITSCORE"]

if all(col in df.columns for col in ["ESTIMATEDSALARY", "ISACTIVEMEMBER"]):
    df["SALARY_X_ACTIVE"] = df["ESTIMATEDSALARY"] * df["ISACTIVEMEMBER"]

if all(col in df.columns for col in ["AGE", "NUMOFPRODUCTS"]):
    df["AGE_X_PRODUCTS"] = df["AGE"] * df["NUMOFPRODUCTS"]

if all(col in df.columns for col in ["RFM_TOTAL", "COHORT_CHURN_RATE"]):
    df["RFM_X_COHORT"] = df["RFM_TOTAL"] * df["COHORT_CHURN_RATE"]

if all(col in df.columns for col in ["CREDITSCORE", "BALANCE"]):
    df["CREDIT_X_BALANCE"] = df["CREDITSCORE"] / (df["BALANCE"] + 1)

if all(col in df.columns for col in ["NUMOFPRODUCTS", "ISACTIVEMEMBER"]):
    df["PRODUCTS_X_ACTIVE"] = df["NUMOFPRODUCTS"] * df["ISACTIVEMEMBER"]

if all(col in df.columns for col in ["SENIOR_CUSTOMER", "ISACTIVEMEMBER"]):
    df["SENIOR_X_INACTIVE"] = df["SENIOR_CUSTOMER"] * (1 - df["ISACTIVEMEMBER"])

if "RFM_TOTAL" in df.columns:
    df["LOW_RFM_FLAG"] = (df["RFM_TOTAL"] <= 6).astype(int)

ENGINEERED = [
    "BALANCE_X_CREDIT",
    "SALARY_X_ACTIVE",
    "AGE_X_PRODUCTS",
    "RFM_X_COHORT",
    "CREDIT_X_BALANCE",
    "PRODUCTS_X_ACTIVE",
    "SENIOR_X_INACTIVE",
    "LOW_RFM_FLAG",
]

FEATURES = NUMERIC + CATS + ENGINEERED
FEATURES = [f for f in FEATURES if f in df.columns]

X = df[FEATURES]
y = df[TARGET]

print(f"✅ Total features: {len(FEATURES)}")


# ─────────────────────────────────────────────────────────────
# 3. SPLIT
# ─────────────────────────────────────────────────────────────
print("\nSTEP 3: Splitting...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.15, stratify=y_train, random_state=42
)

print(f"✅ Train:{len(X_tr):,} Val:{len(X_val):,} Test:{len(X_test):,}")


# ─────────────────────────────────────────────────────────────
# 4. PREPROCESS + SMOTE
# ─────────────────────────────────────────────────────────────
print("\nSTEP 4: Preprocessing + SMOTE...")

imp = SimpleImputer(strategy="median")
scl = StandardScaler()

X_tr_p = scl.fit_transform(imp.fit_transform(X_tr))
X_val_p = scl.transform(imp.transform(X_val))
X_train_p = scl.transform(imp.transform(X_train))
X_test_p = scl.transform(imp.transform(X_test))

print(f"Before SMOTE — 0:{(y_tr == 0).sum():,} | 1:{(y_tr == 1).sum():,}")

smote = SMOTE(random_state=42, sampling_strategy=1.0)
X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_p, y_tr)

print(f"After SMOTE  — 0:{(y_tr_sm == 0).sum():,} | 1:{(y_tr_sm == 1).sum():,}")

os.makedirs("outputs", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# HELPER: EVALUATE ANY MODEL
# ─────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_prob, threshold=None):
    if threshold is None:
        best_t, best_f1 = 0.50, 0.0
        for t_ in np.arange(0.25, 0.70, 0.01):
            p = (y_prob >= t_).astype(int)
            f = f1_score(y_true, p, pos_label=1, zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t_
        threshold = best_t

    y_pred = (y_prob >= threshold).astype(int)

    print(f"\n{'=' * 55}")
    print(f"MODEL: {name} | Threshold: {threshold:.2f}")
    print(f"{'=' * 55}")
    print(classification_report(
        y_true, y_pred, target_names=["Retained", "Churned"]
    ))

    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1)
    f1 = f1_score(y_true, y_pred, pos_label=1)

    print(f"Accuracy       : {acc:.4f} ({acc * 100:.2f}%)")
    print(f"ROC-AUC        : {auc:.4f}")
    print(f"Avg Precision  : {ap:.4f}")
    print(f"Churned Prec   : {prec:.4f}")
    print(f"Churned Recall : {rec:.4f}")
    print(f"Churned F1     : {f1:.4f}")

    return {
        "name": name,
        "acc": acc,
        "auc": auc,
        "ap": ap,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "threshold": threshold,
        "y_prob": y_prob,
    }


results = {}


# ─────────────────────────────────────────────────────────────
# 5. MODEL A: XGBOOST
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MODEL A: XGBoost + SMOTE")
print("=" * 55)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

xgb_pipe = ImbPipeline([
    ("smote", SMOTE(random_state=42, sampling_strategy=1.0)),
    ("xgb", xgb.XGBClassifier(
        n_estimators=500,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric="logloss",
        random_state=42,
        verbosity=0
    ))
])

t = time.time()
xgb_pipe.fit(X_train_p, y_train)
s = cross_val_score(xgb_pipe, X_train_p, y_train, cv=skf, scoring="roc_auc")
print(f"\nCV1 Stratified K-Fold (5):")
print(f"   Scores: {[f'{x:.4f}' for x in s]}")
print(f"   Mean  : {s.mean():.4f} ± {s.std():.4f} | ⏱️{time.time() - t:.1f}s")

t = time.time()
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=42)
rs = cross_val_score(xgb_pipe, X_train_p, y_train, cv=rskf, scoring="roc_auc")
print(f"\nCV2 Repeated K-Fold (3x5=15):")
print(
    f"   Mean:{rs.mean():.4f} Std:{rs.std():.4f} "
    f"Min:{rs.min():.4f} Max:{rs.max():.4f} | ⏱️{time.time() - t:.1f}s"
)

t = time.time()
rs_search = RandomizedSearchCV(
    ImbPipeline([
        ("smote", SMOTE(random_state=42, sampling_strategy=1.0)),
        ("xgb", xgb.XGBClassifier(
            eval_metric="logloss",
            random_state=42,
            verbosity=0
        ))
    ]),
    param_distributions={
        "xgb__max_depth": [6, 8, 10],
        "xgb__learning_rate": [0.01, 0.05, 0.1],
        "xgb__n_estimators": [300, 500, 700],
        "xgb__min_child_weight": [1, 3, 5],
        "xgb__subsample": [0.7, 0.8, 0.9],
        "xgb__colsample_bytree": [0.7, 0.8, 0.9],
        "xgb__gamma": [0, 0.1, 0.2],
    },
    n_iter=20,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    scoring="roc_auc",
    n_jobs=-1,
    random_state=42,
    verbose=0
)
rs_search.fit(X_train_p, y_train)
best_xgb_pipe = rs_search.best_estimator_

print(f"\nCV3 RandomizedSearchCV:")
print(f"   Best Params: {rs_search.best_params_}")
print(f"   Best AUC   : {rs_search.best_score_:.4f} | ⏱️{time.time() - t:.1f}s")

t = time.time()
ts, tr_s, v_s = learning_curve(
    best_xgb_pipe,
    X_train_p,
    y_train,
    cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
    train_sizes=np.linspace(0.1, 1.0, 8),
    scoring="roc_auc",
    n_jobs=-1,
)
plt.figure(figsize=(8, 5))
plt.plot(ts, tr_s.mean(1), label="Train")
plt.plot(ts, v_s.mean(1), label="Val")
plt.fill_between(
    ts, tr_s.mean(1) - tr_s.std(1), tr_s.mean(1) + tr_s.std(1), alpha=0.1
)
plt.fill_between(
    ts, v_s.mean(1) - v_s.std(1), v_s.mean(1) + v_s.std(1), alpha=0.1
)
plt.xlabel("Training Size")
plt.ylabel("AUC")
plt.title("CV4: Learning Curve — XGBoost")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/learning_curve_xgb.png")
plt.close()
print(f"CV4 Learning Curve saved | ⏱️{time.time() - t:.1f}s")

xgb_proba = best_xgb_pipe.predict_proba(X_test_p)[:, 1]
results["XGBoost"] = evaluate("XGBoost", y_test, xgb_proba)


# ─────────────────────────────────────────────────────────────
# 6. MODEL B: ANN
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MODEL B: ANN (25 epochs)")
print("=" * 55)

n = X_tr_sm.shape[1]

ann = keras.Sequential([
    layers.Input(shape=(n,)),
    layers.Dense(256, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(128, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.3),

    layers.Dense(64, activation="relu"),
    layers.BatchNormalization(),
    layers.Dropout(0.2),

    layers.Dense(32, activation="relu"),
    layers.Dropout(0.15),

    layers.Dense(1, activation="sigmoid")
])

ann.compile(
    optimizer=keras.optimizers.Adam(0.001),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        keras.metrics.AUC(name="auc"),
        keras.metrics.Precision(name="precision"),
        keras.metrics.Recall(name="recall"),
    ],
)

t = time.time()
history = ann.fit(
    X_tr_sm,
    y_tr_sm,
    validation_data=(X_val_p, y_val),
    epochs=25,
    batch_size=256,
    verbose=1,
    callbacks=[
        callbacks.EarlyStopping(
            monitor="val_auc",
            patience=5,
            restore_best_weights=True,
            mode="max",
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=0
        ),
        callbacks.ModelCheckpoint(
            "outputs/best_ann.h5",
            monitor="val_auc",
            save_best_only=True,
            mode="max",
            verbose=0
        ),
    ],
)

print(f"✅ ANN done — {len(history.history['loss'])} epochs | ⏱️{time.time() - t:.1f}s")
print(f"✅ Best Val AUC: {max(history.history['val_auc']):.4f}")

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (m, vm, title) in enumerate([
    ("loss", "val_loss", "Loss"),
    ("auc", "val_auc", "AUC"),
    ("accuracy", "val_accuracy", "Accuracy"),
]):
    axes[i].plot(history.history[m], label="Train")
    axes[i].plot(history.history[vm], label="Val")
    axes[i].set_title(title)
    axes[i].legend()

plt.tight_layout()
plt.savefig("outputs/ann_history.png")
plt.close()

ann_proba = ann.predict(X_test_p, verbose=0).flatten()
results["ANN"] = evaluate("ANN", y_test, ann_proba)


# ─────────────────────────────────────────────────────────────
# 7. MODEL C: LIGHTGBM (FOR ENSEMBLE ONLY)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("TRAINING LIGHTGBM FOR ENSEMBLE ONLY")
print("=" * 55)

lgb_model = lgb.LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    objective="binary",
    verbosity=-1
)

t = time.time()
lgb_model.fit(X_tr_sm, y_tr_sm)
print(f"✅ LightGBM trained for ensemble | ⏱️{time.time() - t:.1f}s")

lgb_proba = lgb_model.predict_proba(X_test_p)[:, 1]


# ─────────────────────────────────────────────────────────────
# 8. MODEL D: ENSEMBLE (XGBoost + ANN + LightGBM)
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("MODEL C: Ensemble (XGBoost + ANN + LightGBM)")
print("=" * 55)

# Weighted ensemble
# You can tune these weights later if needed
final_prob = (4 * xgb_proba + 2 * ann_proba + 3 * lgb_proba) / 9
results["Ensemble"] = evaluate("Ensemble", y_test, final_prob)


# ─────────────────────────────────────────────────────────────
# 9. FINAL COMPARISON TABLE
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("FINAL MODEL COMPARISON")
print("=" * 55)

print(f"{'Model':<12} {'Accuracy':>10} {'AUC':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
print("-" * 55)

for name, r in results.items():
    print(
        f"{name:<12} {r['acc']:>10.4f} {r['auc']:>8.4f} "
        f"{r['prec']:>8.4f} {r['rec']:>8.4f} {r['f1']:>8.4f}"
    )

best_model = max(results.items(), key=lambda x: x[1]["f1"])
print(f"\n🏆 Best model by Churned F1: {best_model[0]}")


# ─────────────────────────────────────────────────────────────
# 10. PLOTS FOR BEST MODEL
# ─────────────────────────────────────────────────────────────
best = best_model[1]
y_pred = (best["y_prob"] >= best["threshold"]).astype(int)

fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred,
    ax=ax,
    display_labels=["Retained", "Churned"],
    cmap="Blues"
)
plt.title(f"Confusion Matrix — {best['name']}")
plt.tight_layout()
plt.savefig("outputs/confusion_matrix.png")
plt.close()

plt.figure(figsize=(8, 6))
for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
    plt.plot(fpr, tpr, label=f"{name} AUC={r['auc']:.4f}")
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve — All Models")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/roc_curve.png")
plt.close()

p_arr, r_arr, _ = precision_recall_curve(y_test, best["y_prob"])
plt.figure(figsize=(7, 5))
plt.plot(r_arr, p_arr, label=f"AP={best['ap']:.4f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-Recall — {best['name']}")
plt.legend()
plt.tight_layout()
plt.savefig("outputs/precision_recall.png")
plt.close()

print("✅ All plots saved")


# ─────────────────────────────────────────────────────────────
# 11. SCORE ALL CUSTOMERS
# ─────────────────────────────────────────────────────────────
print("\nScoring all customers...")

X_all = scl.transform(imp.transform(df[FEATURES]))

p_xgb_all = best_xgb_pipe.predict_proba(X_all)[:, 1]
p_ann_all = ann.predict(X_all, verbose=0).flatten()
p_lgb_all = lgb_model.predict_proba(X_all)[:, 1]

# Use winning model for full scoring
if best["name"] == "XGBoost":
    p_all = p_xgb_all
elif best["name"] == "ANN":
    p_all = p_ann_all
else:
    p_all = (4 * p_xgb_all + 2 * p_ann_all + 3 * p_lgb_all) / 9

customer_id_col = "CUSTOMER_ID" if "CUSTOMER_ID" in ml.columns else None

if customer_id_col is not None:
    output = pd.DataFrame({
        "CUSTOMER_ID": ml[customer_id_col].values,
        "CHURN_PROBABILITY": p_all,
        "CHURN_LABEL": (p_all >= best["threshold"]).astype(int),
        "RISK_TIER": pd.cut(
            p_all,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True
        ),
        "MODEL_VERSION": f"ann_xgb_lgb_{best['name'].lower()}_v1"
    })
else:
    output = pd.DataFrame({
        "CHURN_PROBABILITY": p_all,
        "CHURN_LABEL": (p_all >= best["threshold"]).astype(int),
        "RISK_TIER": pd.cut(
            p_all,
            bins=[0, 0.3, 0.6, 1.0],
            labels=["LOW", "MEDIUM", "HIGH"],
            include_lowest=True
        ),
        "MODEL_VERSION": f"ann_xgb_lgb_{best['name'].lower()}_v1"
    })

output.to_csv("outputs/churn_predictions.csv", index=False)

print(f"📊 Risk Distribution:\n{output['RISK_TIER'].value_counts()}")


# ─────────────────────────────────────────────────────────────
# 12. SAVE ARTIFACTS
# ─────────────────────────────────────────────────────────────
ann.save("outputs/ann_model.h5")
joblib.dump(best_xgb_pipe, "outputs/best_xgb_pipe.pkl")
joblib.dump(lgb_model, "outputs/lgb_model.pkl")
joblib.dump(imp, "outputs/imputer.pkl")
joblib.dump(scl, "outputs/scaler.pkl")
joblib.dump(best["threshold"], "outputs/best_threshold.pkl")
joblib.dump(FEATURES, "outputs/feature_names.pkl")

print("\n" + "=" * 55)
print(f"✅ ALL DONE! Best model: {best['name']}")
print("   outputs/ folder:")
print("   📄 churn_predictions.csv")
print("   📊 confusion_matrix.png | roc_curve.png")
print("   📊 precision_recall.png | ann_history.png")
print("   📊 learning_curve_xgb.png")
print("   💾 ann_model.h5 | best_xgb_pipe.pkl | lgb_model.pkl")
print("=" * 55)