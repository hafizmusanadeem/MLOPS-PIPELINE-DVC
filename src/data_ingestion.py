import os
import yaml
import logging
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure the "logs" directory exists
log_dir = 'Logs'
os.makedirs(log_dir, exist_ok=True)

# Logging Configuration

logger = logging.getLogger('data_ingestion')
logger.setLevel('DEBUG')

# Handler // This handler will print the logs in our terminal

# Stream Handler // CommandLine
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG') # Level set to debug, to give logging info.

# File Handler // File Logging
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter
# setting the format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# Giving the made format to both console handler and the file handler
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)


# Now, returning both, the StreamHandler and the FileHandler to the logger after formatting.
logger.addHandler(console_handler)
logger.addHandler(file_handler)

'''Now, the logger object will tell me the logs in the command line and in a file as well to keep long term record. '''

# Logging Finished.


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



# DataIngestion Pipeline Start

#Load the Data
def load_data(data_url:str)->pd.DataFrame:
    """Load data from a CSV File"""
    # Try/except is useful for error handling.
    try:
        # ReadData
        df = pd.read_csv(data_url)
        # MakeLog
        logger.debug('Data loaded from %s', data_url)
        # Return the Data
        return df
    # Write the exception.
    except pd.errors.ParserError as e:
        logger.error('Failed to Parse the CSV File: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error occured while loading hte data: %s',e)
        raise
    # Load Data Complete

# Preprocessing the Data.
def preprocess_data(df: pd.DataFrame)->pd.DataFrame:
    """Preprocess the Data"""
    #Adding error handling, try/except
    try:
        df.drop(columns = ['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], inplace = True)
        df.rename(columns = {'v1':'target', 'v2':'text'}, inplace = True)
        # MakeLog
        logger.debug('Data Preprocessing Complete')
        return df
    # Write the excpetion
    except KeyError as e:
        logger.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected Error while preprocessing: %s',e)
        raise
    #Preprocess Data Complete

# Save Data Function
def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    # Error Handling, try/except
    try:
        #Make a folder and save the data to that folder.
        #DataPath is given, to that location , a new folder will be created where datafile will be stored.
        raw_data_path = os.path.join(data_path, 'raw')
        #Make Folder
        os.makedirs(raw_data_path, exist_ok=True)
        
        # Save the train/test
        train_data.to_csv(os.path.join(raw_data_path, 'train.csv'), index=False)
        test_data.to_csv(os.path.join(raw_data_path, 'test.csv'), index=False)

        #Add Logger
        logger.debug('Train & Test data saved to %s', raw_data_path)
        #Write the excpetion
    except Exception as e:
        logger.error('Unexpected Error Occured while saving the data %s',e)
        raise
    #Save Data Complete

#Main Function
def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        data_path = 'https://raw.githubusercontent.com/vikashishere/Datasets/refs/heads/main/spam.csv'
        df = load_data(data_path)
        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df,  test_size=test_size, random_state=2)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logger.error('Failed to complete the data ingestion process %s', e)
        raise

if __name__ == '__main__':
    main()
