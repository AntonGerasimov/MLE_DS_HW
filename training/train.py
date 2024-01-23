import numpy as np
import pandas as pd
import os
import sys

import json
import logging

import time
from datetime import datetime


import torch
from torch import optim
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Comment this lines if you have problems with MLFlow installation
#import mlflow
#mlflow.autolog()

# Adds the root directory to system path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))

# Change to CONF_FILE = "settings.json" if you have problems with env variables
CONF_FILE = "settings.json"

from utils import get_project_dir, configure_logging, IrisPredictionModel

# Loads configuration settings from JSON
with open(CONF_FILE, "r") as file:
    conf = json.load(file)

# Defines paths
DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])


def get_accuracy_multiclass(pred_arr, original_arr):
    if len(pred_arr)!=len(original_arr):
        return False
    pred_arr = pred_arr.numpy()
    original_arr = original_arr.numpy()
    final_pred= []
 
    for i in range(len(pred_arr)):
        final_pred.append(np.argmax(pred_arr[i]))
    final_pred = np.array(final_pred)
    count = 0
    #here we are doing a simple comparison between the predicted_arr and the original_arr to get the final accuracy
    for i in range(len(original_arr)):
        if final_pred[i] == original_arr[i]:
            count+=1
    return count/len(final_pred)

class DataProcessor():
    def __init__(self) -> None:
        pass

    def prepare_data(self, max_rows: int = None) -> pd.DataFrame:
        logging.info("Preparing data for training...")
        df = self.data_extraction(TRAIN_PATH)
        df = self.data_rand_sampling(df, max_rows)
        return df

    def data_extraction(self, path: str) -> pd.DataFrame:
        logging.info(f"Loading data from {path}...")
        return pd.read_csv(path)
    
    def data_rand_sampling(self, df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
        if not max_rows or max_rows < 0:
            logging.info('Max_rows not defined. Skipping sampling.')
        elif len(df) < max_rows:
            logging.info('Size of dataframe is less than max_rows. Skipping sampling.')
        else:
            df = df.sample(n=max_rows, replace=False, random_state=conf['general']['random_state'])
            logging.info(f'Random sampling performed. Sample size: {max_rows}')
        return df


class Training():
    def __init__(self) -> None:
        self.model = IrisPredictionModel()
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)

    def run_training(self, df: pd.DataFrame, out_path: str = None, test_size: float = 0.33) -> None:
        logging.info("Running training...")
        
        X_train, X_test, y_train, y_test = self.data_split(df, test_size=test_size)
        X_train, X_test = self.data_scaling(X_train, X_test)
        X_train = torch.FloatTensor(np.array(X_train))
        X_test = torch.FloatTensor(np.array(X_test))
        y_train = torch.LongTensor(np.array(y_train))
        y_test = torch.LongTensor(np.array(y_test))
        
        num_epochs = conf['train']['num_epochs']
        train_losses = np.zeros(num_epochs)
        test_losses  = np.zeros(num_epochs)
        
        start_time = time.time()
        self.train(X_train, y_train, X_test, y_test, num_epochs, train_losses, test_losses)
        end_time = time.time()
        
        logging.info(f"Training completed in {end_time - start_time} seconds.")
        self.test(X_test, y_test)
        self.save(out_path)


    def data_split(self, df: pd.DataFrame, test_size: float = 0.3) -> tuple:
        logging.info("Splitting data into training and test sets...")
        features = df.columns.drop('target')
        outcome = 'target'
        return train_test_split(df[features], df[outcome], test_size=test_size, random_state=conf['general']['random_state'])
    
    def data_scaling(self, X_train, X_test):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        return X_train, X_test
    
    def train(self, X_train: pd.DataFrame,y_train: pd.DataFrame,X_test: pd.DataFrame,y_test: pd.DataFrame,num_epochs,train_losses,test_losses) -> None:
        
        torch.manual_seed(conf['general']['random_state'])
        
        model = self.model
        optimizer = self.optimizer
        criterion = self.criterion
        
        logging.info("Training the model...")
        for epoch in range(num_epochs):
            #clear out the gradients from the last step loss.backward()
            self.optimizer.zero_grad()
            
            #forward feed
            output_train = model(X_train)

            #calculate the loss
            loss_train = criterion(output_train, y_train)
            
            #backward propagation: calculate gradients
            loss_train.backward()

            #update the weights
            optimizer.step()

            output_test = model(X_test)
            loss_test = criterion(output_test,y_test)

            train_losses[epoch] = loss_train.item()
            test_losses[epoch] = loss_test.item()

            if (epoch + 1) % 20 == 0:
                logging.info(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {loss_train.item():.4f}, Test Loss: {loss_test.item():.4f}")

        logging.info(f"Final Training Loss: {train_losses[-1]:.4f}, Final Test Loss: {test_losses[-1]:.4f}")
        
        predictions_train = []
        with torch.no_grad():
            predictions_train = model(X_train)
        logging.info(f"Training accuracy = {get_accuracy_multiclass(predictions_train,y_train)}")


    def test(self, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
        logging.info("Testing the model...")
        predictions_test =  []
        with torch.no_grad():
            predictions_test = self.model(X_test)
            
        res = get_accuracy_multiclass(predictions_test,y_test)
        logging.info(f"Test accuracy = {res}")

        return res

    def save(self, path: str) -> None:
        logging.info("Saving the model...")
        if not os.path.exists(MODEL_DIR):
            os.makedirs(MODEL_DIR)

        if not path:
            path = os.path.join(MODEL_DIR, datetime.now().strftime(conf['general']['datetime_format']) + '.pth')
        else:
            path = os.path.join(MODEL_DIR, path)

        torch.save(self.model.state_dict(), path)
        
def main():
    configure_logging()

    data_proc = DataProcessor()
    tr = Training()

    df = data_proc.prepare_data(max_rows=conf['train']['data_sample'])
    tr.run_training(df, test_size=conf['train']['test_size'])


if __name__ == "__main__":
    main()