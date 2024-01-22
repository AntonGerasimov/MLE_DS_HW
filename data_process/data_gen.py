import numpy as np
import pandas as pd
import os
import sys
import json
import logging
from sklearn import datasets
from sklearn.datasets import load_iris

# Define directories
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
from utils import singleton, get_project_dir, configure_logging

# Create logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

CONF_FILE = "settings.json"

logger.info("Loading configuration settings from JSON...")
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Define paths
logger.info("Defining paths...")
DATA_DIR = get_project_dir(conf['general']['data_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
TEST_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])

@singleton
class IrisSetLoader:
    def __init__(self):
        iris_data = load_iris()
        self.df = pd.DataFrame(iris_data.data, columns=load_iris().feature_names)
        self.df['target'] = iris_data.target
        self.conf_file = "settings.json"
        self.load_config()

    def load_config(self):
        logger.info(f"Loading configuration from {self.conf_file}...")
        with open(self.conf_file, 'r') as f:
            config = json.load(f)
            self.save_path_train = config.get("save_path_train", "train_data.csv")
            self.save_path_test = config.get("save_path_test", "test_data.csv")
            self.test_data_split = config.get("test_data_split", 0.2)
        
    def create(self, test_data_split: float, save_path_train: str, save_path_test: str):
        train_data_split = 1 - test_data_split
        train_len = int(len(self.df) * train_data_split)
        train_df, test_df = self.split_data(train_len)
        self.save(train_df, save_path_train)
        self.save(test_df, save_path_test)
    
    # Method to split the data into train and test
    def split_data(self, train_len: int):
        logger.info("Splitting data into train and test datasets...")
        train_df = self.df.iloc[:train_len, :]
        test_df = self.df.iloc[train_len:, :]
        return train_df, test_df
    
    # Method to save data
    def save(self, df: pd.DataFrame, out_path: str):
        logger.info(f"Saving data to {out_path}...")
        df.to_csv(out_path, index=False)
        
# Main execution
if __name__ == "__main__":
    configure_logging()
    logger.info("Starting script...")
    iris = IrisSetLoader()
    iris.create(0.2, save_path_train=TRAIN_PATH, save_path_test=TEST_PATH)
    logger.info("Script completed successfully.")