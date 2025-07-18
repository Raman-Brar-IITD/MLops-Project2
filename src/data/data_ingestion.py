import numpy as np
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)

import os
from sklearn.model_selection import train_test_split
import yaml
import logging
from src.logger import logging
from src.connections import s3_connection 

def load_params(params_path:str)->dict:
    """Loads parameters from YaML file"""
    try:
        with open(params_path,"rb") as file:
            params=yaml.safe_load(file)
        logging.debug("Parameters loaded from %s",params_path)
        return params
    except FileNotFoundError :
        logging.error("F ile not found:%s",params_path)
        raise
    except yaml.YAMLError as e:
        logging.error("YAML error : %s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error occured:%s",e)

def load_data(data_url:str)->pd.DataFrame:
    """Loads data from a csv file"""
    try:
        df=pd.read_csv(data_url)
        logging.info("Data lodded from:%s",data_url)
        return df
    except pd.errors.ParserError as e:
        logging.error("Parsing Error ocuured :%s",e)
        raise
    except Exception as e:
        logging.error("Unexpected error ocuured:%s",e)

def preprocess_data(df:pd.DataFrame)->pd.DataFrame:
    """Preprocess the data"""
    try:

        logging.info("pre-processing...")
        final_df = df[df['sentiment'].isin(['positive', 'negative'])]
        final_df['sentiment'] = final_df['sentiment'].replace({'positive': 1, 'negative': 0})
        logging.info('Data preprocessing completed')
        return final_df
    except KeyError as e:
        logging.error('Missing column in the dataframe: %s', e)
        raise
    except Exception as e:
        logging.error('Unexpected error during preprocessing: %s', e)
        raise

def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)
        logging.debug('Train and test data saved to %s', raw_data_path)
    except Exception as e:
        logging.error('Unexpected error occurred while saving the data: %s', e)
        raise

def main():
    try:
        params = load_params(params_path='params.yaml')
        test_size = params['data_ingestion']['test_size']
        # test_size = 0.2
        

        
        acesskey=os.getenv("ACKEY")
        secretkey=os.getenv("SCKEY")
        s3 = s3_connection.s3_operations("mlopsproject2", acesskey, secretkey)
        df = s3.fetch_file_from_s3("data.csv")
        # df=pd.read_csv("./notebooks/data.csv")


        final_df = preprocess_data(df)
        train_data, test_data = train_test_split(final_df, test_size=test_size, random_state=42)
        save_data(train_data, test_data, data_path='./data')
    except Exception as e:
        logging.error('Failed to complete the data ingestion process: %s', e)
        print(f"Error: {e}")

if __name__ == '__main__':
    main()






