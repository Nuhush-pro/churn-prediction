"""
First end-to-end pass: load -> clean -> engineer features -> EDA plots ->
baseline (logistic regression) vs stronger model (random forest) ->
evaluation. Run with: python src/train_baseline.py
"""
import matplotlib
matplotlib.use("Agg")  # no display needed, just save PNGs

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from data_prep import clean_data, engineer_features, load_raw_data

DATA_PATH = "data/raw_telco_churn.csv"
FIGURES_DIR = "reports/figures"


def make_eda_plots(df: pd.DataFrame) -> None:
    """Save the EDA charts the project plan asks for, with the raw numbers
    printed alongside so the takeaways can be written up from real output."""

    # 1. Class balance
    churn_rate = df["Churn"].mean()
    plt.figure(figsize=(5, 4))
    df["Churn"].value_counts().plot(kind="bar")
    plt.title(f"Churn class balance (churn rate = {churn_rate:.1%})")
    plt.xlabel("Churn (0 = stayed, 1 = left)")
    plt.ylabel("Customers")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/01_class_balance.png")
    plt.close()

    # 2. Tenure vs churn
    plt.figure(figsize=(6, 4))
    df.boxplot(column="tenure", by="Churn")
    plt.title("Tenure (months) by churn outcome")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/02_tenure_vs_churn.png")
    plt.close()

    # 3. Monthly charges distribution by churn
    plt.figure(figsize=(6, 4))
    for churned, label in [(0, "stayed"), (1, "churned")]:
        df[df["Churn"] == churned]["MonthlyCharges"].plot(
            kind="hist", alpha=0.5, bins=30, label=label
        )
    plt.legend()
    plt.title("Monthly charges distribution by churn outcome")
    plt.xlabel("MonthlyCharges")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/03_monthly_charges_by_churn.png")
    plt.close()

    # 4. Contract type vs churn rate
    plt.figure(figsize=(6, 4))
    df.groupby("Contract")["Churn"].mean().plot(kind="bar")
    plt.title("Churn rate by contract type")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/04_churn_rate_by_contract.png")
    plt.close()

    # 5. num_active_services vs churn rate (our engineered feature)
    plt.figure(figsize=(6, 4))
    df.groupby("num_active_services")["Churn"].mean().plot(kind="bar")
    plt.title("Churn rate by number of active add-on services")
    plt.ylabel("Churn rate")
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/05_churn_rate_by_num_services.png")
    plt.close()

    print(f"Overall churn rate: {churn_rate:.1%}")
    print(
        "Churn rate by contract type:\n",
        df.groupby("Contract")["Churn"].mean().round(3).to_string(),
    )
    print(
        "Churn rate by number of active services:\n",
        df.groupby("num_active_services")["Churn"].mean().round(3).to_string(),
    )


def build_pipeline(model, numeric_cols, categorical_cols) -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(name, pipeline, X_test, y_test) -> None:
    preds = pipeline.predict(X_test)
    probs = pipeline.predict_proba(X_test)[:, 1]

    print(f"\n--- {name} ---")
    print(f"Precision: {precision_score(y_test, preds):.3f}")
    print(f"Recall:    {recall_score(y_test, preds):.3f}")
    print(f"F1:        {f1_score(y_test, preds):.3f}")
    print(f"ROC-AUC:   {roc_auc_score(y_test, probs):.3f}")
    print("Confusion matrix [[TN FP] [FN TP]]:")
    print(confusion_matrix(y_test, preds))


def main():
    df = load_raw_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    make_eda_plots(df)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges", "num_active_services"]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    baseline = build_pipeline(
        LogisticRegression(max_iter=1000), numeric_cols, categorical_cols
    )
    baseline_cv_auc = cross_val_score(baseline, X_train, y_train, cv=cv, scoring="roc_auc")
    baseline.fit(X_train, y_train)

    strong = build_pipeline(
        RandomForestClassifier(n_estimators=200, random_state=42),
        numeric_cols,
        categorical_cols,
    )
    strong_cv_auc = cross_val_score(strong, X_train, y_train, cv=cv, scoring="roc_auc")
    strong.fit(X_train, y_train)

    print(f"\nBaseline (Logistic Regression) 5-fold CV ROC-AUC: {baseline_cv_auc.mean():.3f} (+/- {baseline_cv_auc.std():.3f})")
    print(f"Random Forest 5-fold CV ROC-AUC:                  {strong_cv_auc.mean():.3f} (+/- {strong_cv_auc.std():.3f})")

    evaluate("Logistic Regression (baseline)", baseline, X_test, y_test)
    evaluate("Random Forest (stronger model)", strong, X_test, y_test)


if __name__ == "__main__":
    main()
