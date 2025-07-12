import pandas as pd
import numpy as np
import os
import pickle
import json

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def load_test_data(data_path: str) -> pd.DataFrame:
    """Load test data from a CSV file."""
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Test data file {data_path} not found.")
    except Exception as e:
        print(f"An error occurred while loading test data {data_path}: {e}")
# test_data = pd.read_csv("data/processed/test_processed.csv")

# Prepare the data for evaluation
def prepare_data(data: pd.DataFrame) -> tuple['pd.DataFrame', 'pd.Series']:
    """Prepare the data for model evaluation."""
    try:
        X_test = data.drop(columns=['Potability'], axis=1)  # Fetch all independent variables
        y_test = data['Potability']  # Fetch the dependent variable
        return X_test, y_test
    except FileNotFoundError:
        print("Data file not found.")
    except KeyError as e:
        print(f"Missing key in data preparation: {e}")
    except Exception as e:
        print(f"An error occurred while preparing data: {e}")
# X_test = test_data.iloc[:, 0:-1].values  # Fetch all independent variables
# y_test = test_data.iloc[:, -1].values  # Fetch the dependent variable

# Load the trained model
def load_model(model_path:str):
    """Load the trained model from a file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        print(f"Model file {model_path} not found.")
    except Exception as e:
        print(f"An error occurred while loading the model: {e}")
# model = pickle.load(open("model.pkl", "rb"))

# Evaluate the model
def evaluate_model(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """Evaluate the model and return performance metrics."""
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        return {
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        }
    except Exception as e:
        raise Exception(f"An error occurred while evaluating the model: {e}")   
# y_pred = model.predict(X_test)

# accuracy_score = accuracy_score(y_test, y_pred)
# precision_score = precision_score(y_test, y_pred) 
# recall_score = recall_score(y_test, y_pred)
# f1_score = f1_score(y_test, y_pred)

# metrics = {
#     "Accuracy": accuracy_score,
#     "Precision": precision_score,
#     "Recall": recall_score,
#     "F1 Score": f1_score}

# Save the evaluation metrics
def save_metrics(metrics:dict, metrics_path: str) -> None:
    """Save the evaluation metrics to a JSON file."""
    try:
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {metrics_path}")
    except Exception as e:
        print(f"An error occurred while saving metrics: {e}")
# with open("metrics.json", "w") as f:
#     json.dump(metrics, f, indent=4)


def main():
    """Main function to execute the model evaluation process."""
    try:
        data_path = "data/processed/"
        model_path = "model.pkl"
        metrics_path = "metrics.json"

        test_data = load_test_data(os.path.join(data_path, "test_processed.csv"))
        X_test, y_test = prepare_data(test_data)
        model = load_model(model_path=model_path)
        metrics = evaluate_model(model=model, X_test=X_test, y_test=y_test)
        save_metrics(metrics=metrics, metrics_path=metrics_path)
    except Exception as e:
        print(f"An error occurred in the main function: {e}")

if __name__ == "__main__":
    main()  
