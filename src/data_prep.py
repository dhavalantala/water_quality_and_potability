import pandas as pd
import numpy as np
import os 

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
# train_data = pd.read_csv("data/raw/train.csv")
# test_data = pd.read_csv("data/raw/test.csv")

def fill_missing_values(df):
    try:
        # Fill missing values with the mean of each column
        for column in df.columns:
            if df[column].isnull().any():
                median_value = df[column].median()
                df[column].fillna(median_value, inplace=True)
        return df
    except Exception as e:
        raise Exception(f"Error filling missing values: {e}")

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
# train_processed_data = fill_missing_values(train_data)
# test_processed_data = fill_missing_values(test_data)

def main():
    try:
        train_data = load_data("data/raw/train.csv")
        test_data = load_data("data/raw/test.csv")
        train_processed_data = fill_missing_values(train_data)
        test_processed_data = fill_missing_values(test_data)
        data_path = os.path.join("data", "processed")
        os.makedirs(data_path, exist_ok=True)  # Create directory if it doesn't exist
        save_data(train_processed_data, os.path.join(data_path, "train_processed.csv"))
        save_data(test_processed_data, os.path.join(data_path, "test_processed.csv"))
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
# This code is responsible for loading the raw data, filling missing values, and saving the processed
# data into the specified directories. It uses exception handling to manage errors during data processing.  