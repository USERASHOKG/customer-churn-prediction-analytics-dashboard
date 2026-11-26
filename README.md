# Customer Churn Prediction & Analytics Dashboard

An end-to-end machine-learning project for predicting telecom customer churn and turning model output into actionable retention insights. It uses the public IBM Telco Customer Churn dataset, compares three classifiers, handles class imbalance, and exposes the results through a Streamlit dashboard.

## Highlights

- Cleans mixed-type customer data and converts `TotalCharges` safely.
- Encodes categorical variables with a production-style `ColumnTransformer` pipeline.
- Compares logistic regression, random forest, and gradient boosting with stratified 5-fold cross-validation.
- Reports precision, recall, F1, ROC-AUC, and holdout performance.
- Uses balanced class weights or sample weights so the minority churn class is not ignored.
- Includes contract and tenure segment analysis, model comparison, feature importance, and live customer scoring.

## Quick start

```bash
git clone https://github.com/USERASHOKG/customer-churn-prediction-analytics-dashboard.git
cd customer-churn-prediction-analytics-dashboard
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
streamlit run app.py
```

The training script downloads the public dataset into `data/` on first run and saves reproducible artifacts into `models/`. Generated model binaries and metadata are intentionally ignored by Git; they can always be recreated with `python train.py`.

## Project Technical structure

```text
├── app.py                 # Streamlit dashboard
├── train.py               # Dataset download and training entry point
├── src/churn_pipeline.py  # Cleaning, preprocessing, evaluation, inference
├── tests/                 # Unit tests for the pipeline
├── data/                  # Cached public dataset (created by train.py)
└── models/                # Generated artifacts (created by train.py)
```

## Modeling notes

The primary selection criterion is cross-validated ROC-AUC, with precision, recall, and F1 shown alongside it to make the business trade-off visible. Logistic regression and random forest use balanced class weights. Gradient boosting uses balanced sample weights during fitting. The live scorer uses the saved best model and an explicit 0.5 probability threshold, which can be adjusted for a retention team's preferred intervention capacity.

## Tests

```bash
pytest -q
```

## Data source and license

Dataset: [IBM Telco Customer Churn](https://github.com/IBM/telco-customer-churn-on-icp4d). Review the source repository's terms before redistributing the data.
