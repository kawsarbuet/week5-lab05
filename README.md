**AIGC 5005 — Lab 05: Predict Late Deliveries**  
**Project:** QuickBite late-order prediction  


QuickBite wants to predict, **when a customer places an order**, whether delivery will be more than ten minutes late. This project uses a reproducible scikit-learn classification pipeline and chooses a decision threshold based on the company's voucher and missed-delay costs rather than accuracy alone.

## Setup

From the repository root:

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install scikit-learn pandas matplotlib pytest ruff ipykernel
pip install -r requirements.txt
pip install -e .
```

The lab environment uses Python 3.11 and the `conda-forge` channel. The project uses NumPy, pandas, scikit-learn, matplotlib, joblib, pytest, Ruff, and ipykernel. Use the pinned versions in `requirements.txt` when reproducing results.

## Run

```bash
python -m latedeliveries.train
```

This command is intended to regenerate the trained pipeline and reports. The required outputs are:

- `models/late_orders.joblib` — serialized complete preprocessing and classification pipeline.
- `models/model_info.json` — environment/model metadata, features, threshold, hyperparameters, and test results.
- `reports/cv_results.csv` — five-fold cross-validation model comparison.
- `reports/metrics.json` — held-out test metrics, confusion matrix, and policy costs.
- `reports/threshold_costs.json` — threshold-cost data used by the analysis notebook.

Open `notebooks/analysis.ipynb` in Jupyter or VS Code and run its cells after generating the reports. The notebook displays the prevalence, model comparison, cost-versus-threshold plot, confusion matrix, and business recommendation without retraining on the test set.

## Check it

Run these from the repository root:

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

Ruff checks Python files **and** notebook code. The project configuration should select `E`, `F`, `I`, `B`, and `UP` lint rules with an 88-character line limit. To apply formatting changes before rerunning the checks:

```bash
python -m ruff format .
python -m ruff check --fix .
```

**Validation status:** During development, the notebook initially failed JSON/UTF-8 validation and later reported a long-line/formatting issue. Those issues were diagnosed and corrective steps were applied. A final successful run of all three commands has **not** been shown in the available activity log; rerun them and confirm they pass before submission.

## Decisions

### Data and split

The provided `make_orders()` function generates **3,000 synthetic orders** using a fixed seed. The notebook recorded **468 late orders (15.6%)**. The target is `late`, where 1 means the order arrived late.

The lab requires a **stratified 75% training / 25% test split** with a single fixed random-state constant. Model comparison, hyperparameter tuning, and threshold selection must use the training data only; the held-out test set is evaluated once after those choices are fixed.

### Feature selection and leakage prevention

The model is designed to make predictions at order placement, so features must be available at that time.

| Column | Decision | Reason |
| --- | --- | --- |
| `distance_km` | Use — numeric | Order-to-customer distance is known at placement. |
| `items` | Use — numeric | Number of ordered items is known at placement. |
| `hour` | Use — numeric | Placement hour is immediately known. |
| `weather` | Use — categorical | Current weather can be known when the order is placed. |
| `day_type` | Use — categorical | Weekday/weekend status is known. |
| `order_id` | Exclude | An identifier, not a reliable predictor of delay. |
| `prep_minutes` | Exclude | Actual kitchen preparation time is observed after placement. |
| `customer_rating` | Exclude | Customer feedback is collected after delivery; direct target leakage risk. |
| `vehicle` | Exclude | The assigned courier's vehicle may not be known at order placement. |
| `courier_months` | Exclude | Courier experience may not be known before courier assignment. |
| `late` | Exclude from features | This is the outcome being predicted. |

The preprocessing pipeline uses a `ColumnTransformer` to impute missing numeric values with the median, scale numeric features, and one-hot encode categorical values with `handle_unknown="ignore"`. Preprocessing is fitted **inside** each cross-validation fold, not on the full dataset before splitting. The pipeline is also intended to handle missing values and previously unseen weather categories at prediction time.

### Five-fold model comparison

The following are the **training-only, five-fold stratified cross-validation results** recorded in the saved notebook. Each metric is shown as **mean ± standard deviation**.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: |
| Dummy | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 |
| Logistic regression | 0.857 ± 0.007 | 0.627 ± 0.072 | 0.213 ± 0.049 | 0.315 ± 0.059 | 0.775 ± 0.038 |
| Balanced logistic regression | 0.739 ± 0.022 | 0.332 ± 0.032 | 0.661 ± 0.046 | 0.442 ± 0.038 | 0.775 ± 0.038 |
| Decision tree | 0.848 ± 0.008 | 0.540 ± 0.064 | 0.248 ± 0.027 | 0.336 ± 0.017 | 0.724 ± 0.021 |
| Random forest | 0.860 ± 0.002 | 0.705 ± 0.062 | 0.179 ± 0.021 | 0.284 ± 0.021 | 0.768 ± 0.042 |

The Dummy baseline has high accuracy because late orders are uncommon, but it never detects a late order. This shows why accuracy alone is inadequate. Logistic regression and balanced logistic regression achieved the highest mean ROC-AUC (0.775). Balanced logistic regression recovered more late orders at the default classification threshold, but also produced more false positives. The business decision therefore depends on **cost at a separately selected threshold**, not solely on default-threshold F1 or accuracy.

**Model choice:** The recorded recommendation uses a logistic-regression-based model and a cost-selected threshold. The precise final estimator and its tuning settings should be confirmed against `models/model_info.json` before submission.

### Hyperparameter tuning

The lab requires `GridSearchCV` on the chosen pipeline with **at least two hyperparameters and six combinations**. ROC-AUC is a defensible search metric because it evaluates probability ranking across thresholds, while the final action threshold is optimized separately for monetary cost.

**Verify and complete from `models/model_info.json` / training output before submission:**

- Best hyperparameter settings: **not available in the notebook outputs reviewed**.
- Untuned versus tuned cross-validation score: **not available in the notebook outputs reviewed**.
- Whether any improvement exceeded the fold-to-fold standard deviation: **not yet verifiable**.

Do not claim that tuning materially improved performance until these figures have been checked.

### Cost rule and threshold

QuickBite's policy assigns a **$4 voucher cost to every flagged order** and a **$15 cost to each late order that was not flagged**:

```text
Total cost = 4 × (number of flagged orders)
           + 15 × (number of late orders not flagged)
```

The required method uses out-of-fold predicted probabilities from `cross_val_predict` on the training set and searches thresholds **0.05, 0.10, …, 0.95**. The saved notebook reports a chosen threshold of **0.25**. This threshold balances the expense of offering unnecessary vouchers against the larger cost of missing an actual late delivery. It is chosen **without consulting the held-out test set**.

## Result

### Held-out test performance

The recorded test set contains **750 orders** (25% of 3,000). The notebook's reported metrics and policy costs imply the following confusion matrix:

| Actual / Predicted | Predicted on time | Predicted late |
| --- | ---: | ---: |
| Actual on time | 561 (TN) | 72 (FP) |
| Actual late | 71 (FN) | 46 (TP) |

*These counts are reconstructed from the recorded precision, recall, accuracy, and policy costs; compare them with the exact `confusion_matrix` field in `reports/metrics.json` before submitting.*

| Metric | Held-out result |
| --- | ---: |
| Accuracy | 0.8093 |
| Precision | 0.3898 |
| Recall | 0.3932 |
| F1 | 0.3915 |
| ROC-AUC | 0.7477 |
| Selected threshold | 0.25 |

### Policy cost comparison

| Policy | Calculation | Total cost |
| --- | --- | ---: |
| Flag nobody | 117 missed late orders × $15 | **$1,755** |
| Flag everybody | 750 vouchers × $4 | **$3,000** |
| Model at threshold 0.25 | 118 vouchers × $4 + 71 missed late orders × $15 | **$1,537** |

The model saved **$218** relative to flagging nobody, or approximately **12.4%** of that baseline's cost, on this simulated held-out test set. This supports a **limited pilot**, not a production rollout: the savings are based on invented orders and simplified business assumptions, not observed customer behavior.

## What the tests cover

The required test suite is intended to protect these behaviors:

1. The feature lists exclude identifiers, the target, and information unavailable at order placement.
2. The stratified train/test split is reproducible and preserves the approximate late-order rate.
3. The pipeline returns valid probabilities between 0 and 1.
4. Preprocessing handles missing values and previously unseen categories.
5. The cost function correctly prices vouchers and missed late deliveries using hand-calculated examples.
6. Threshold selection works on small examples, including cases where flagging nobody or everybody is cheapest.
7. A saved pipeline produces consistent predictions after reloading with joblib.
8. At least one test is parametrized to cover multiple inputs.

**Test status:** These are the lab-required behaviors and planned coverage, not a claim that each individual test was independently inspected. Run `python -m pytest` and confirm at least eight tests pass. One area not directly validated by synthetic unit tests is whether voucher offers actually prevent dissatisfaction in a real-world delivery operation; that would require a live pilot or observational study.

## Known limitations

- **Synthetic data:** The 3,000 orders are generated, not real QuickBite deliveries; results may not transfer to production.
- **Simplified costs:** The scenario assumes a voucher always resolves dissatisfaction. Real voucher effectiveness and missed-delay costs may vary.
- **Moderate recall:** At the chosen threshold, the model identifies approximately 39% of late orders and misses the remainder.
- **Feature availability:** Weather, distance, and courier assignment timing must be validated in an operational system before use.
- **Unverified tuning details:** Exact best parameters and tuned-versus-untuned performance were not present in the notebook outputs reviewed and must be added from training artifacts.
- **Leakage demonstration not documented:** The lab suggests temporarily adding an excluded feature and comparing ROC-AUC, then removing it. No measured leakage-experiment result was available to report here.
- **Final checks not independently confirmed:** Formatting, linting, pytest, and fresh-clone reproducibility should all be rerun and recorded before submission.
