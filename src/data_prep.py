"""
Data loading, cleaning, and feature engineering for the Telco Customer Churn
dataset. Kept separate from the training script so these functions are
unit-testable on their own (tests come in Project 2).
"""
import pandas as pd

# Columns that use "Yes" / "No" / "No internet service" / "No phone service"
# to describe an add-on service. We treat anything other than a literal
# "Yes" as "doesn't have this service" when counting active services.
SERVICE_COLUMNS = [
    "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
]


def load_raw_data(path: str) -> pd.DataFrame:
    """Load the raw Telco Customer Churn CSV exactly as downloaded."""
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Fix known data-quality issues in the raw export.

    Cleaning decisions (documented, not silent):
    - `customerID` is a unique identifier, not a feature -> dropped.
    - `TotalCharges` is loaded as text because 11 rows (all brand-new
      customers with tenure == 0) have a blank string instead of a number.
      We coerce to numeric and fill those blanks with 0, since a customer
      with 0 months of tenure genuinely has been charged $0 so far.
    - `Churn` (the target) is text Yes/No -> mapped to 1/0 so it can be
      used directly by scikit-learn.
    """
    df = df.copy()

    df = df.drop(columns=["customerID"])

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that aren't present in the raw export.

    - `tenure_group`: bucket tenure (months) into ranges. A customer's
      churn risk isn't linear in tenure -- the first year behaves very
      differently from year five -- so a categorical bucket lets tree
      models and linear models alike pick up on that shape.
    - `num_active_services`: count of add-on services the customer has
      actually subscribed to. A customer with zero add-ons and one with
      every add-on are very different churn risks even if their raw
      MonthlyCharges happen to be similar.
    """
    df = df.copy()

    df["tenure_group"] = pd.cut(
        df["tenure"],
        bins=[-1, 12, 24, 48, 60, 72],
        labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
    )

    df["num_active_services"] = (df[SERVICE_COLUMNS] == "Yes").sum(axis=1)

    return df
