"""Train the best churn model and persist dashboard artifacts."""
from pathlib import Path
import pandas as pd
from src.churn_pipeline import train_best_model, save_artifacts

DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
MODEL_DIR = ROOT / "models"


def main() -> None:
    DATA_PATH.parent.mkdir(exist_ok=True)
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH)
    else:
        df = pd.read_csv(DATA_URL)
        df.to_csv(DATA_PATH, index=False)
    model, metrics, holdout = train_best_model(df)
    save_artifacts(model, metrics, holdout, MODEL_DIR)
    print("Saved model artifacts to", MODEL_DIR)
    print(metrics.to_string(index=False))
    print("Holdout:", holdout)


if __name__ == "__main__":
    main()
