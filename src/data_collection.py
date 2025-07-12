import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
with open("params.yaml", 'r') as file:
    params = yaml.safe_load(file)

test_size = params['data_collection']['test_size']
random_state = params['data_collection']['random_state']

# Load the dataset
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from the specified file path.
    
    :param file_path: Path to the dataset file.
    :return: DataFrame containing the dataset.
    """
    try:
        return pd.read_csv(file_path)
    except Exception as e:
        raise Exception(f"Error loading data from {file_path}: {e}")
# data = pd.read_csv(r"/Users/dhavalantala/Downloads/water_potability.csv")



def split_data(data: pd.DataFrame, test_size: float, random_state: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into training and testing sets.
    
    :param data: DataFrame containing the dataset.
    :param test_size: Proportion of the dataset to include in the test split.
    :param random_state: Controls the shuffling applied to the data before applying the split.
    :return: Tuple containing training and testing DataFrames.
    """
    try:
        return train_test_split(data, test_size=test_size, random_state=random_state)
    except ValueError as e:
        raise ValueError(f"Error splitting data: {e}")
# train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save the DataFrame to a CSV file.
    
    :param data: DataFrame to be saved.
    :param file_path: Path where the DataFrame will be saved.
    """
    try:
        data.to_csv(file_path, index=False)
    except Exception as e:
        raise Exception(f"Error saving data to {file_path}: {e}")

def main():
    try:
        data_file_path = "/Users/dhavalantala/Downloads/water_potability.csv"
        raw_data_path = os.path.join("data", "raw")

        data = load_data(data_file_path)
        train_data, test_data = split_data(data, test_size=test_size, random_state=random_state)
        os.makedirs(raw_data_path, exist_ok=True)  # Create directory if it doesn't exist
        save_data(train_data, os.path.join(raw_data_path, "train.csv"))
        save_data(test_data, os.path.join(raw_data_path, "test.csv"))
    except Exception as e:
        raise Exception(f"An error occurred in data collection: {e}")

if __name__ == "__main__":
    main()
# This code is responsible for collecting data, splitting it into training and testing sets, and saving
# the processed data into the specified directories. It uses parameters defined in params.yaml for configuration.