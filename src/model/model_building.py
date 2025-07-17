import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle 
import yaml
from src.logger import logging
from src.features.feature_engineering import load_data

def train_model(X_tr:np.ndarray,y_tr:np.ndarray)->LogisticRegression:
    """
    trains the logistic regression model given x-train and y-train ndarray"""
    try:
        clf=LogisticRegression(C=1,penalty="l2",solver="liblinear")
        clf.fit(X_tr,y_tr)
        logging.info("model training done")
        return clf
    except Exception as e:
        logging.error("Unexcepted error ocuured while model training :%s",e)
        raise

def save_model(model,path_url:str)->None:
    """Ssaves the model in pickle file takes model and path AS INPUT"""
    try:
        with open(path_url,"wb") as file:
            pickle.dump(model,file)
        logging.info("model saved to :%s",path_url)
    except Exception as e:
        logging.error("Unexpected error occured during saving the model: %s",e)
        raise

def main():
    try:
        train_data=load_data("./data/processed/train_bow.csv")
        X_tr=train_data.iloc[:,:-1].values
        y_tr=train_data.iloc[:,-1].values

        clf=train_model(X_tr,y_tr)

        save_model(clf,"models/model.pkl")
    except Exception as e:
        logging.error('Failed to complete the model building process: %s', e)
        print(f"Error: {e}")
        raise
if __name__=="__main__":
    main()


