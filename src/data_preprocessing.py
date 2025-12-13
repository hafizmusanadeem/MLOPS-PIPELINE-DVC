import os
import nltk
import string
import logging
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import LabelEncoder

nltk.download('stopwords')
nltk.download('punkt_tab')

# Logging
#Making Directory
log_dir = 'Logs' 
os.makedirs(log_dir, exist_ok=True)

# Setting up Logger
logger = logging.getLogger('data_preprocessing')
logger.setLevel('DEBUG')

# StreamHandler
console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

# FileHandler
# Make file path first
log_file_path = os.path.join(log_dir, 'data_preprocessing.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')


# Formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

#Setting formatting for console and file handler.
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Logging Completed.

# Data Preprocessing

def transform_text(text):
    """
    Transform the input text by converting it to lowercase, tokenizing, removing stopwords and punctutation, and stemming
    """
    ps = PorterStemmer()
    #Convert to LowerCase
    text = text.lower()
    #Tokenize the text
    text = nltk.word_tokenize(text)
    #Remove Non AlphaNumeric Tokens
    text = [word for word in text if word.isalnum()] 
    #Remove StopWords and Punctuation
    text = [word for word in text if word not in stopwords.words('english') and word not in string.punctuation]
    #Stem the Words
    text = [ps.stem(word) for word in text]
    #Join the Tokens back into the string
    return " ".join(text)

# PreProcessing
def preprocess_df(df, text_column = 'text', target_column = 'target'):
    """
    Preprocesses the DataFrame by encoding the target column, removing duplicates, and Transfroming the text column.
    """
    #Error Handling, try/except
    try:
        #MakeLog
        logger.debug('Start Preprocessing for DataFrame')

        #Encode the target column
        encoder = LabelEncoder()
        df[target_column] = encoder.fit_transform(df[target_column])
        logger.debug('Target Column Encoded')

        #Remove Duplicate Rows
        df = df.drop_duplicates(keep = 'first')
        logger.debug('Duplicates Removed')

        #Apply text Transformation to the specified text column
        df.loc[:, text_column] = df[text_column].apply(transform_text)
        logger.debug('Text Transform Applied')
        return df
    
    except KeyError as e:
        logger.error('Column not found: %s', e)
        raise
    except Exception as e:
        logger.error('Error during text normalization: %s', e)
        raise
    # text preprocessing completed.

# main function
def main(text_column = 'text', target_column = 'target'):
    try:
        """
        Main function to load raw data, preprocess it and save the preprocessed data.
        """
        # Load the Data
        train_data = pd.read_csv('./data/raw/train.csv')
        test_data = pd.read_csv('./data/raw/test.csv')
        #MakeLog
        logger.debug('Data Loaded Properly')

        # Transform the data
        train_processed_data = preprocess_df(train_data, text_column=text_column, target_column=target_column)
        test_processed_data = preprocess_df(test_data, text_column=text_column, target_column=target_column)
    
        #Store the Processed Data into ./data/raw
        data_path = os.path.join('./data', 'interim')
        os.makedirs(data_path, exist_ok=True)

        #Directory is made and confirmed so now saving the processed data file
        train_processed_data.to_csv(os.path.join(data_path, 'train_processed.csv'), index = False)
        test_processed_data.to_csv(os.path.join(data_path, 'test_processed.csv'), index = False)
    
        #MakeLog
        logger.debug('Processed data saved to %s', data_path)

        #Write the exception
    except FileNotFoundError as e:
        logger.error('File not Found %s', e)
        raise
    except pd.errors.EmptyDataError as e:
        logger.error('No data: %s', e)
        raise
    except Exception as e:
        logger.error('Failed to complete the Data Preprocessing Step: %s', e)
        print(f'Error: {e}')
        raise
    
# To Run the code
if __name__ == '__main__':
    main()