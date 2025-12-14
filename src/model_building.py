import os
import yaml
import pickle
import logging
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Logging File

log_dir = 'Logs'
os.makedirs(log_dir, exist_ok=True)

# Log Configuration
logger = logging.getLogger('model_training')
logger.setLevel('DEBUG')

# StreamHandler & FileHandler

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'model.building.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

#Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Logging Completed


# Yaml Parameters Setup

def load_params(params_path: str) -> dict:
    """This function loads parameters from a params.yaml file"""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retieved from: %s', file)
        return params
    except FileNotFoundError:
        logger.error('File not Found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML Error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error while retrieving parameters: %s', e)
        raise


# Model Training

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load Data from a CSV file
    
    :param file_path: path to the CSV file
    :return: Loaded DataFrame
    """
    try:
        df = pd.read_csv(file_path)
        logger.debug('Data Loaded from %s of the shape %s', file_path, df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to Parse the CSV file %s', e)
        raise
    except FileNotFoundError as e:
        logger.error('File not Found', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading the data', e)
        raise

def train_model(X_train: np.ndarray, y_train: np.ndarray, params: dict) -> RandomForestClassifier:
    """
    Train the RandomForest Model
    
    :param X_train: Training Features
    :type X_train: np.ndarray
    :param y_train: Training Labels
    :type y_train: np.ndarray
    :param params: Dictionary of Hyperparameters
    :type params: dict
    :return: Trained RandomForestClassifer
    :rtype: RandomForestClassifier
    """
    try:
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError('The Number of Sample in X_train and y_train must be same')
        
        logger.debug('Initializing RandomForest Model with Parameters %s', params)
        clf = RandomForestClassifier(n_estimators=params['n_estimators'], random_state = params['random_state'])

        logger.debug('Model Training Started with %d samples', X_train.shape[0])
        clf.fit(X_train, y_train)
        logger.debug('Model Training Completed')

        return clf
    except ValueError as e:
        logger.error('ValueError during Model Training: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected Error while Model Trainig: %s', e)
        raise

def save_model(model, file_path: str) -> None:
    """
    Save the trained Model
    
    :param model: Trained Model Object
    :param file_path: Path to save the Model File
    """
    try:
        #Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, 'wb') as file:
            pickle.dump(model, file)
        logger.debug('Model is saved to %s', file_path)
    except FileNotFoundError as e:
        logger.error('FilePath Not Found: %s', e)
        raise
    except Exception as e:
        logger.error('Error Occured while saving the model %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')['model_building']
        train_data= load_data('./data/processed/train_tfidf.csv')
        X_train= train_data.iloc[:, :-1].values
        y_train= train_data.iloc[:, -1].values

        clf = train_model(X_train, y_train, params)

        model_save_path = 'models/model.pkl'
        save_model(clf, model_save_path)
    except Exception as e:
        logger.error('Failed to complete the model building process: %s',e)
        print(f'Error: {e}')

if __name__ == '__main__':
    main()
    
