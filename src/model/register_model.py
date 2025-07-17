import json
import mlflow
import logging
from src.logger import logging
import os
import dagshub
import time

import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

dagshub_token = os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/Raman-Brar-IITD/MLops-Project2.mlflow")

def load_model_info(file_path: str) -> dict:
    """Load the model info from a JSON file."""
    try:
        with open(file_path, 'r') as file:
            model_info = json.load(file)
        logging.debug('Model info loaded from %s', file_path)
        return model_info
    except FileNotFoundError:
        logging.error('File not found: %s', file_path)
        raise
    except Exception as e:
        logging.error('Unexpected error occurred while loading the model info: %s', e)
        raise

def register_model(model_name:str,model_info:dict):
    """Register the model to the MLflow Model Registry"""
    try:
        model_uri=f"runs:/{model_info['run_id']}/{model_info['model_path']}"
        
        model_version=mlflow.register_model(model_uri,model_name)
        client=mlflow.tracking.MlflowClient()
        for _ in range(10):
            status = client.get_model_version(model_name, model_version.version).status
            if status == "READY":
                break
            time.sleep(2)
        else:
            raise Exception("Model registration timed out.")
        
        client.transition_model_version_stage(name=model_name,version=model_version.version,stage="Staging")
        logging.debug(f'Model {model_name} version {model_version.version} registered and transitioned to Staging.')
    except Exception as e:
        logging.error("Error during model registration:%s",e)
        raise

def main():
    try:
        model_info_path='reports/experiment_info.json'
        model_info=load_model_info(model_info_path)

        model_name="my_model"
        register_model(model_name,model_info)
    except Exception as e:
        logging.error("Failed to complete the model registration process:%s",e)
        raise

if __name__=="__main__":
    main()
