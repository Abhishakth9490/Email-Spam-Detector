"""
Spambase Email Spam Classification — Project Pipeline
========================================================
Loads the UCI Spambase dataset, performs EDA, trains and compares
several classification models, evaluates them, and saves the best
model plus supporting plots and a results table.

Run:  python3 src/train_and_evaluate.py
"""

import json
import warnings
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "spambase.csv"
PLOTS_DIR = ROOT / "outputs" / "plots"
MODEL_DIR = ROOT / "outputs" / "model"
RESULTS_PATH = ROOT / "outputs" / "results.json"

RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Missing values: {df.isna().sum().sum()}")

target_col = "class"
X = df.drop(columns=[target_col])
y = df[target_col]

class_counts = y.value_counts().to_dict()
print(f"Class balance -> Not spam(0): {class_counts.get(0)}, Spam(1): {class_counts.get(1)}")

# ---------------------------------------------------------------------------
# 2. EDA plots
# ---------------------------------------------------------------------------
# 2a. Class distribution
plt.figure(figsize=(5, 4))
sns.countplot(x=y, hue=y, palette=["#4C72B0", "#DD8452"], legend=False)
plt.xticks([0, 1], ["Not Spam", "Spam"])
plt.title("Class Distribution")
plt.ylabel("Number of Emails")
plt.xlabel("")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "class_distribution.png", dpi=150)
plt.close()

# 2b. Top features correlated with spam
corr_with_target = X.corrwith(y).sort_values(key=lambda s: s.abs(), ascending=False)
top_corr = corr_with_target.head(15)
plt.figure(figsize=(7, 6))
sns.barplot(x=top_corr.values, y=top_corr.index, hue=top_corr.index, palette="coolwarm", legend=False)
plt.title("Top 15 Features Correlated with Spam Label")
plt.xlabel("Correlation with 'class'")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "top_correlated_features.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 3. Train / test split + scaling
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 4. Train & compare multiple models
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
}

results = {}
fitted_models = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    proba = model.predict_proba(X_test_scaled)[:, 1]

    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="accuracy")

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, proba),
        "cv_accuracy_mean": cv_scores.mean(),
        "cv_accuracy_std": cv_scores.std(),
    }
    results[name] = metrics
    fitted_models[name] = model
    print(f"\n{name}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

# ---------------------------------------------------------------------------
# 5. Model comparison plot
# ---------------------------------------------------------------------------
results_df = pd.DataFrame(results).T.sort_values("f1_score", ascending=False)
results_df.to_csv(ROOT / "outputs" / "model_comparison.csv")

plt.figure(figsize=(9, 5))
plot_metrics = results_df[["accuracy", "precision", "recall", "f1_score", "roc_auc"]]
plot_metrics.plot(kind="bar", ax=plt.gca(), colormap="viridis")
plt.title("Model Comparison Across Metrics")
plt.ylabel("Score")
plt.ylim(0.8, 1.0)
plt.xticks(rotation=20, ha="right")
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "model_comparison.png", dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# 6. Best model — confusion matrix, ROC curve, feature importance
# ---------------------------------------------------------------------------
best_model_name = results_df.index[0]
best_model = fitted_models[best_model_name]
print(f"\nBest model: {best_model_name}")

preds_best = best_model.predict(X_test_scaled)
cm = confusion_matrix(y_test, preds_best)
plt.figure(figsize=(5, 4))
ConfusionMatrixDisplay(cm, display_labels=["Not Spam", "Spam"]).plot(
    cmap="Blues", colorbar=False, values_format="d"
)
plt.title(f"Confusion Matrix — {best_model_name}")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "confusion_matrix_best_model.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
ax = plt.gca()
for name, model in fitted_models.items():
    RocCurveDisplay.from_estimator(model, X_test_scaled, y_test, ax=ax, name=name)
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
plt.title("ROC Curves — All Models")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "roc_curves_all_models.png", dpi=150)
plt.close()

if hasattr(best_model, "feature_importances_"):
    importances = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(
        ascending=False
    ).head(15)
    plt.figure(figsize=(7, 6))
    sns.barplot(x=importances.values, y=importances.index, hue=importances.index,
                palette="mako", legend=False)
    plt.title(f"Top 15 Feature Importances — {best_model_name}")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_best_model.png", dpi=150)
    plt.close()

# ---------------------------------------------------------------------------
# 7. Save best model + scaler + results summary
# ---------------------------------------------------------------------------
joblib.dump(best_model, MODEL_DIR / "best_model.pkl")
joblib.dump(scaler, MODEL_DIR / "scaler.pkl")

summary = {
    "best_model": best_model_name,
    "n_samples": int(df.shape[0]),
    "n_features": int(X.shape[1]),
    "class_balance": {str(k): int(v) for k, v in class_counts.items()},
    "test_set_size": int(len(y_test)),
    "all_model_results": results,
}
with open(RESULTS_PATH, "w") as f:
    json.dump(summary, f, indent=2)

print("\nDone. Artifacts saved to outputs/.")
