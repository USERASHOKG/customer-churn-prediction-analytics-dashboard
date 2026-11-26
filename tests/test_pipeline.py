import pandas as pd
from src.churn_pipeline import clean_data, predict_customer, train_best_model


def sample_data(rows=80):
    alternating = lambda first, second: [first if i % 2 == 0 else second for i in range(rows)]
    base = {
        "customerID": [f"id-{i}" for i in range(rows)],
        "gender": alternating("Female", "Male"), "SeniorCitizen": alternating(0, 1),
        "Partner": alternating("Yes", "No"), "Dependents": alternating("No", "Yes"),
        "tenure": list(range(rows)), "PhoneService": ["Yes"] * rows, "MultipleLines": ["No"] * rows,
        "InternetService": alternating("DSL", "Fiber optic"), "OnlineSecurity": alternating("No", "Yes"),
        "OnlineBackup": ["No"] * rows, "DeviceProtection": ["No"] * rows, "TechSupport": ["No"] * rows,
        "StreamingTV": ["No"] * rows, "StreamingMovies": ["No"] * rows,
        "Contract": alternating("Month-to-month", "Two year"), "PaperlessBilling": ["Yes"] * rows,
        "PaymentMethod": alternating("Electronic check", "Mailed check"),
        "MonthlyCharges": [70.0 + i for i in range(rows)], "TotalCharges": [str(70.0 + i * 2) for i in range(rows)],
        "Churn": ["Yes" if i % 3 == 0 else "No" for i in range(rows)],
    }
    return pd.DataFrame(base)


def test_clean_data_converts_target_and_total_charges():
    result = clean_data(sample_data())
    assert result["Churn"].dtype.name == "Int64"
    assert result["TotalCharges"].dtype.kind in "fi"


def test_train_and_score_customer():
    model, metrics, holdout = train_best_model(sample_data())
    assert set(["precision", "recall", "f1", "roc_auc"]).issubset(metrics.columns)
    assert 0 <= holdout["roc_auc"] <= 1
    customer = sample_data(1).drop(columns=["Churn", "customerID"]).iloc[0].to_dict()
    result = predict_customer(model, customer)
    assert 0 <= result["probability"] <= 1
    assert result["prediction"] in (0, 1)
