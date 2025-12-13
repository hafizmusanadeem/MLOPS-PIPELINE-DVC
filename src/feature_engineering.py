import os
import logging
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Logging

log_dir = 'Logs'
os.makedirs(log_dir, exist_ok=True)

# logging config
logger = logging.getLogger('feature_engineering')
logger.setLevel('DEBUG')

# StreamHandler and FileHandler

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir, 'feature_engineering.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Logging Completed

# Feature Extraction

def load_data(file_path: str)->pd.DataFrame:
    """
    Load Data from a Csv File
    """
    try:

        df = pd.read_csv(file_path)
        df.fillna('', inplace = True)
        logger.debug('Data Loaded and NaNs filled from: %s', file_path)
        return df
    # Write the Excpetion
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV file: %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected error while loading the data file: %s', e)
        raise

# Apply Bagofwords to the text (features)
def apply_tfidf(train_data: pd.DataFrame, test_data: pd.DataFrame, max_features: int) -> tuple:
    """
    Apply TfIdf technique to the Data
    """
    try:
        vectorizer = TfidfVectorizer(max_features=max_features)

        X_train = train_data['text'].values
        y_train = train_data['target'].values
        X_test = test_data['text'].values
        y_test = test_data['target'].values

        X_train_bow = vectorizer.fit_transform(X_train)
        X_test_bow = vectorizer.transform(X_test)

        # Arranging the Numpy arrayed Dataset back to the train_df, test_df
        train_df = pd.DataFrame(X_train_bow.toarray())
        train_df['label'] = y_train

        test_df = pd.DataFrame(X_test_bow.toarray())
        test_df['label'] = y_test

        #MakeLog
        logger.debug('TfIdf applied and Data Transformed')

        return train_df, test_df
    # Write the Exception
    except Exception as e:
        logger.error('Error During TfIdf Transformation: %s', e)
        raise

# Save this data

def save_data(df: pd.DataFrame, file_path: str) ->None:
    """This saves the bow applied dataset to a CSV file"""

    try:
        os.makedirs(os.path.dirname(file_path), exist_ok = True)
        df.to_csv(file_path, index = False)
        logger.debug('Data Saved to %s', file_path)
    #Write the Exception
    except Exception as e:
        logger.error('Unexpected Error Occured while Saving the Data', e)
        raise

# Main function

def main():
    try: 
        max_features = 50
        # Loading the Data
        train_data = load_data('./data/interim/train_processed.csv')
        test_data = load_data('./data/interim/test_processed.csv')
        
        train_data, test_data = apply_tfidf(train_data = train_data, test_data = test_data, max_features=max_features)

        save_data(train_data, os.path.join("./data", "processed", "train_tfidf.csv"))
        save_data(test_data, os.path.join("./data", "processed", "test_tfidf.csv"))
    #Write the Exception
    except Exception as e:
        logger.error('Failed to Complete the Feature Engineering Process: %s', e)
        print(f"Error: {e}")

# Run the Code
if __name__ == '__main__':
    main()
