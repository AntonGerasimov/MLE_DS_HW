import unittest
import pandas as pd
import os
import sys
import json
import torch
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(ROOT_DIR))
CONF_FILE = "settings.json"

from training.train import DataProcessor

from utils import get_project_dir, IrisPredictionModel

with open(CONF_FILE, "r") as file:
    conf = json.load(file)

DATA_DIR = get_project_dir(conf['general']['data_dir'])
MODEL_DIR = get_project_dir(conf['general']['models_dir'])
TRAIN_PATH = os.path.join(DATA_DIR, conf['train']['table_name'])
TEST_PATH = os.path.join(DATA_DIR, conf['inference']['inp_table_name'])


class TestDataProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(CONF_FILE, "r") as file:
            conf = json.load(file)
        cls.data_dir = conf['general']['data_dir']
        cls.train_path = os.path.join(cls.data_dir, conf['train']['table_name'])
        
    def test_train_data_presence(self):
        df = pd.read_csv(TRAIN_PATH)
        self.assertEqual(df.shape[0], 120)
        
    def test_test_data_presence(self):
        df = pd.read_csv(TEST_PATH)
        self.assertEqual(df.shape[0], 30)    
        

    def test_data_extraction(self):
        dp = DataProcessor()
        df = dp.data_extraction(self.train_path)
        self.assertIsInstance(df, pd.DataFrame)

    def test_prepare_data(self):
        dp = DataProcessor()
        df = dp.prepare_data(100)
        self.assertEqual(df.shape[0], 100)


class TestTraining(unittest.TestCase):
    
    def test_model_presence(self):
        model = IrisPredictionModel()
        self.assertIsInstance(model, IrisPredictionModel, "model is not an instance of IrisPredictionModel")
        
        
    def test_model_forward(self):
        # Test the forward pass of the model
        model = IrisPredictionModel()
        input_data = torch.randn(32, 4)  # Assuming input size is 4, and a batch size of 32

        # Ensure the forward pass doesn't raise any errors
        try:
            output = model(input_data)
        except Exception as e:
            self.fail(f"Forward pass raised an exception: {e}")

        # Assert the shape of the output tensor
        self.assertEqual(output.shape, (32, 3), msg="Output shape is not as expected")
        
    def test_model_training(self):
        # Test the training loop of the model
        model = IrisPredictionModel()
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Generate some dummy data
        input_data = torch.randn(64, 4)  
        target = torch.randint(0, 3, (64,))  # Multi-class classification task with 3 classes

        # Ensure the training step doesn't raise any errors
        try:
            optimizer.zero_grad()
            output = model(input_data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        except Exception as e:
            self.fail(f"Training step raised an exception: {e}")

if __name__ == '__main__':
    unittest.main()