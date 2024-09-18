import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import yfinance as yf

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer
@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts',"train.csv")
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            ticker = 'AAPL'
            stock_data = yf.download(ticker, start='2010-01-01', end='2023-01-01')
            stock_data.to_csv('apple_stock_data.csv')
            df = pd.read_csv('apple_stock_data.csv', index_col='Date', parse_dates=True, infer_datetime_format=True)
            
            full_range = pd.date_range(start=df.index.min(), end=df.index.max())
            df_full = df.reindex(full_range)

            df_full['Close'] = df_full['Close'].interpolate(method='linear')

            df_full.index.name = 'Date'

            df_full.to_csv('apple_stock_data_filled.csv')
            df=pd.read_csv('apple_stock_data_filled.csv')
            df['Date']=pd.to_datetime(df['Date'])
            df.set_index('Date',inplace=True)
            df = df[['Close']].copy()
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=True,header=True)

            logging.info("Train test split initiated")
          
            train_set,test_set=train_test_split(df,test_size=0.2,shuffle=False)

            train_set.to_csv(self.ingestion_config.train_data_path,index=True,header=True)

            test_set.to_csv(self.ingestion_config.test_data_path,index=True,header=True)

            logging.info("Inmgestion of the data iss completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=DataIngestion()
    train_data,test_data=obj.initiate_data_ingestion()

    """ data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data) """

    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer())



