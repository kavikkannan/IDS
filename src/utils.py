import os
import sys

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
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
    
def evaluate_models(Data1,Train, Test, models, P, D, Q):
    try:
        report = {}

        Train['Date'] = pd.to_datetime(Train['Date'])
        Train.set_index('Date', inplace=True)

        Test['Date'] = pd.to_datetime(Test['Date'])
        Test.set_index('Date', inplace=True)
        import statsmodels.api as sm

        model = list(models.values())[0](endog=Train["Close"], order=(5,2,1))
        #model=sm.tsa.statespace.SARIMAX(Train['Close'],order=(5, 2, 1),seasonal_order=(5,2,1,12))

        model_fit = model.fit()

        
        
        Train['forecast'] = model_fit.predict(start=0 ,end=len(Train), typ='levels', dynamic=False)
        Train[['Close', 'forecast']].plot(figsize=(12, 8)) 
        plt.savefig('Train')
        print(Train)

        future_data=Test    

        future_data['forecast']= model_fit.predict(start=0 ,end=len(Test), typ='levels', dynamic=True)
        Test[['Close', 'forecast']].plot(figsize=(12, 8)) 
        plt.savefig('Test')
        #model_score = r2_score(Train["Close"].dropna(), future_df['forecast'][:len(Train)].dropna())
        
        Data1.plot(figsize=(12, 8))
        plt.savefig('MainData')
        
        plt.figure(figsize=(12, 8))

        plt.plot(Train['Close'], color='red', label='Train Close')
        plt.plot(Train['forecast'], color='orange', linestyle='--', label='Train Forecast')

        plt.plot(Test['Close'], color='green', label='Test Close')
        plt.plot(Test['forecast'], color='lightgreen', linestyle='--', label='Test Forecast')

        plt.plot(Data1['Close'], color='blue', label='MainData Close')

        plt.title("Train, Test, and Main Data Comparison")
        plt.xlabel("Time")
        plt.ylabel("Close Price")
        plt.legend()

        plt.savefig('Combined_Plot.png')

        plt.show()
        report[list(models.keys())[0]] = 1 

        return report

    except Exception as e:
        raise CustomException(e, sys)


    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)