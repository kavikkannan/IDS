import os
import sys
from dataclasses import dataclass
import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from sklearn.metrics import r2_score
from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self):
        try:
            trained_model_file_path=os.path.join("artifacts","model.pkl")
            data1=pd.read_csv(os.path.join("artifacts","data.csv"))
            train_data=pd.read_csv(os.path.join("artifacts","train.csv"))

            test_data=pd.read_csv(os.path.join("artifacts","test.csv"))

            logging.info("Split training and test input data")
            
            models = {
                "Arima":ARIMA,
                
            }
            
           
            
            model_report:dict=evaluate_models(Data1=data1,Train=train_data,Test=test_data,models=models,P=5,D=1,Q=1)
            
            best_model_score = max(sorted(model_report.values()))


            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            #predicted=best_model.predict(train_data)

            #r2_square = r2_score(test_data, predicted)
            return 0
            



            
        except Exception as e:
            raise CustomException(e,sys)