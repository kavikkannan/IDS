import os
import sys

import numpy as np 
import pandas as pd
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_models(Train,Test,models,P,D,Q):
    try:
        report = {}

        print(Train["Close"])
        model = list(models.values())[0](endog=Train["Close"],order=(5,1,1))
        
        """ model.endog=Train
        model.order(1,1,1) """
        model_fit=model.fit()
        import matplotlib.pyplot as plt

        fitted_values = model_fit.fittedvalues

        plt.figure(figsize=(12, 6))
        plt.plot(Train["Close"].dropna(), label='Original Data')
        plt.plot(fitted_values, color='red', label='Fitted Values')
        plt.title('ARIMA Model Fit')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
        pred = model_fit.predict(start=Test.index[0], end=Test.index[-1], typ='levels')

       
        print("train",Train["Close"])
        print("test",pred)
        model_score = r2_score(Train, pred)


        report[list(models.keys())[0]] = model_score

        return report 

    except Exception as e:
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)