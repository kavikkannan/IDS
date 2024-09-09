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
    
def evaluate_models(Train, Test, models, P, D, Q):
    try:
        report = {}

        # Ensure Train and Test are properly indexed by datetime
        Train['Date'] = pd.to_datetime(Train['Date'])
        Train.set_index('Date', inplace=True)

        Test['Date'] = pd.to_datetime(Test['Date'])
        Test.set_index('Date', inplace=True)
        import statsmodels.api as sm

        # Create and fit the ARIMA model using the training data
        model = list(models.values())[0](endog=Train["Close"], order=(1,1,1))
        model=sm.tsa.statespace.SARIMAX(Train['Close'],order=(5, 2, 1),seasonal_order=(5,2,1,12))

        model_fit = model.fit()

        # Concatenate Train and Test into a single DataFrame for future prediction
        future_df = pd.concat([Train, Test])
        # Perform prediction using ARIMA without dynamic=True
        
        future_df['forecast'] = model_fit.predict(start=Train.index[-1], end=275, typ='levels', dynamic=True)
        future_df[['Close', 'forecast']].plot(figsize=(12, 8)) 
        plt.savefig('kk')
        print(future_df)
        # Calculate R2 score on the training data (optional, since this is not future data)
        """ model_score = r2_score(Train["Close"].dropna(), future_df['forecast'][:len(Train)].dropna())
        report[list(models.keys())[0]] = model_score """

        return report

    except Exception as e:
        raise CustomException(e, sys)


    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)