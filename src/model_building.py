import pandas as pd
import numpy as np
import os 
import yaml
import pickle

from sklearn.ensemble import RandomForestClassifier

# Load parameters from params.yaml
def load_params(param_path: str = "params.yaml") -> float:
    """Load parameters from a YAML file."""
    try:
        with open(param_path, 'r') as file:
            params = yaml.safe_load(file)
        return params
    except FileNotFoundError:
        print(f"Parameter file {param_path} not found.")
    except KeyError as e:
        print(f"Missing key in parameters: {e}")
    except Exception as e:
        print(f"An error occurred while loading parameters: {e}")   
# with open("params.yaml", 'r') as file:
#     params = yaml.safe_load(file)
# n_estimators = params['model_building']['n_estimators']


# Load training data
def get_training_data(data_path: str) -> pd.DataFrame:
    """Load training data from a CSV file."""
    try:
        return pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Training data file {data_path} not found.")
    except Exception as e:
        print(f"An error occurred while loading training data{data_path}: {e}")
# train_data = pd.read_csv("data/processed/test_processed.csv")


# x_train = train_data.iloc[:, 0:-1].values # Fetch all independent variables
# y_train = train_data.iloc[:, -1].values # Fetch the dependent variable

def prepare_data(data: pd.DataFrame) -> tuple['pd.DataFrame', 'pd.Series']:
    """Prepare the data for model training."""
    try:
        X_train = data.drop(columns=['Potability'], axis=1)  # Fetch all independent variables
        y_train = data['Potability']  # Fetch the dependent variable
        return X_train, y_train
    except FileNotFoundError:
        print("Data file not found.")
    except KeyError as e:
        print(f"Missing key in data preparation: {e}")
    except Exception as e:
        print(f"An error occurred while preparing data: {e}")
# X_train = train_data.drop(columns=['Potability'], axis=1)  # Fetch all independent variables
# y_train = train_data['Potability'] # Fetch the dependent variable

# Train the model
def train_model(X_train: pd.DataFrame, y_train: pd.Series, n_estimators: int) -> RandomForestClassifier:
    """Train a Random Forest model."""
    try:
        clf = RandomForestClassifier(n_estimators=n_estimators)
        clf.fit(X_train, y_train)
        return clf
    except Exception as e:
        print(f"An error occurred while training the model: {e}")
# clf = RandomForestClassifier(n_estimators=n_estimators)
# clf.fit(X_train, y_train)

# Save the model
def save_model(model: RandomForestClassifier, model_path: str) -> None:
    """Save the trained model to a file."""
    try:
        with open(model_path, 'wb') as file:
            pickle.dump(model, file)
        print(f"Model saved to {model_path}")
    except Exception as e:
        print(f"An error occurred while saving the model{model_path}: {e}")
# pickle.dump(clf, open("model.pkl", "wb"))


def main():
    """Main function to execute the model building process."""
    params = load_params("params.yaml")
    n_estimators = params["model_building"]["n_estimators"]
    train_data_path = "data/processed/train_processed.csv"
    model_path = "model.pkl"

    train_data = get_training_data(data_path=train_data_path)
    X_train, y_train = prepare_data(train_data)
    model = train_model(X_train, y_train, n_estimators=n_estimators)
    save_model(model=model, model_path = model_path)

if __name__ == "__main__":
    main()  