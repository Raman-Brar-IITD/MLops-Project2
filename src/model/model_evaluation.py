import numpy as np
import pandas as pd
import pickle
import json
from sklearn.metrics import accuracy_score,precision_score,f1_score,roc_auc_score,recall_score
from src.logger import logging
import mlflow
import mlflow.sklearn
import dagshub
import os
from src.features.feature_engineering  import load_data

dagshub_token=os.getenv("CAPSTONE_TEST")
if not dagshub_token:
    raise EnvironmentError("CAPSTONE_TEST environment variable is not set"
                           )

os.environ["MLFLOW_TRACKING_USERNAME"]=dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"]=dagshub_token

mlflow.set_tracking_uri("https://dagshub.com/Raman-Brar-IITD/MLops-Project2.mlflow")

def load_model(file_path:str):
    """loads the trained model from a file"""
    try:
        with open(file_path,"rb") as file:
            model=pickle.load(file)
        logging.info("Model loaded succesfully")
        return model
    except FileNotFoundError:
        logging.error("Model file not found at:%s",file_path)
        raise
    except Exception  as e:
        logging.error("Unexpected error ocured while loading Model: %s",e)
        raise

def evaluate_model(clf,X_ts:np.ndarray,y_ts:np.ndarray)-> dict:
    """Evalute the model and return the evaluation metrics"""
    try:
        y_pred=clf.predict(X_ts)
        y_pred_prob=clf.predict_proba(X_ts)[:,1]

        accuracy = accuracy_score(y_ts, y_pred)
        precision = precision_score(y_ts, y_pred)
        recall = recall_score(y_ts, y_pred)
        auc = roc_auc_score(y_ts, y_pred_prob)

        metrics_dict = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'auc': auc
        }
        logging.info('Model evaluation metrics calculated')
        return metrics_dict
    except Exception as e:
        logging.error('Error during model evaluation: %s', e)
        raise

def save_metrics(mtx:dict,file_path:str)-> None:
    '''Saves the evaluation  metrics in a file JSON'''
    try:
        with open(file_path,"w") as file:
            json.dump(mtx,file,indent=4)
        logging.info('Metrics saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the metrics: %s', e)
        raise


def save_model_info(run_id: str, model_path: str, file_path: str) -> None:
    """Save the model run ID and path to a JSON file."""
    try:
        model_info = {'run_id': run_id, 'model_path': model_path}
        with open(file_path, 'w') as file:
            json.dump(model_info, file, indent=4)
        logging.debug('Model info saved to %s', file_path)
    except Exception as e:
        logging.error('Error occurred while saving the model info: %s', e)
        raise

def main():
    mlflow.set_experiment("my-dvc-pipeline")
    with mlflow.start_run() as run:  # Start an MLflow run
        try:
            clf = load_model('./models/model.pkl')
            test_data = load_data('./data/processed/test_bow.csv')
            
            X_test = test_data.iloc[:, :-1].values
            y_test = test_data.iloc[:, -1].values

            metrics = evaluate_model(clf, X_test, y_test)
            
            save_metrics(metrics, 'reports/metrics.json')
            
            # Log metrics to MLflow
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            
            # Log model parameters to MLflow
            if hasattr(clf, 'get_params'):
                params = clf.get_params()
                for param_name, param_value in params.items():
                    mlflow.log_param(param_name, param_value)
            
            # Log model to MLflow
            mlflow.sklearn.log_model(clf, "model")
            
            # Save model info
            save_model_info(run.info.run_id, "model", 'reports/experiment_info.json')
            
            # Log the metrics file to MLflow
            mlflow.log_artifact('reports/metrics.json')

        except Exception as e:
            logging.error('Failed to complete the model evaluation process: %s', e)
            print(f"Error: {e}")

if __name__ == '__main__':
    main()