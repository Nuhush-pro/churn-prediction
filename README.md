# Customer Churn Prediction

Predicting which customers of a subscription business will cancel, using the
raw (not pre-cleaned) IBM Telco Customer Churn dataset — 7,043 customers,
26.5% of whom churned.

## Data source
IBM Telco Customer Churn dataset, via IBM's public sample-data mirror:
https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
(not committed to this repo — see "Reproduce" below)

## Approach
1. `src/data_prep.py` — cleaning (fix `TotalCharges`, drop the ID column,
   encode the target) and feature engineering (`tenure_group`,
   `num_active_services`), each decision documented in the docstrings.
2. `src/train_baseline.py` — EDA charts (`reports/figures/`), then a
   preprocessing + model pipeline comparing a logistic regression baseline
   against a random forest.

## Results (first pass, untuned)

| Model | 5-fold CV ROC-AUC | Test Precision | Test Recall | Test F1 | Test ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression (baseline) | 0.846 | 0.652 | 0.521 | 0.579 | 0.843 |
| Random Forest | 0.823 | 0.598 | 0.465 | 0.523 | 0.820 |

The baseline beat the "stronger" model on every metric. That's a real result,
not a bug — worth digging into rather than smoothing over.

Recall matters more than precision for this problem: missing a customer who
was about to churn (a false negative) costs a lost customer and all their
future revenue, while a false positive just costs a probably-unnecessary
retention offer. Both models' recall (46-52%) says we're still missing
roughly half of churners — the next pass should look at class-weighting or
a lower decision threshold rather than reaching for a fancier model.

## Status
First working pass complete, not yet tuned or interrogated.

## Reproduce
```
pip install -r requirements.txt
python src/train_baseline.py
```
Dataset: download the CSV from the source link above into `data/raw_telco_churn.csv`
(the `data/` folder is gitignored — never commit raw datasets).
