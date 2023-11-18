import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass 
class DataPaths:
    raw_path: str=os.path.join('artifacts', 'data.csv')
    train_path: str=os.path.join('artifacts', 'train.csv')
    test_path: str=os.path.join('artifacts', 'test.csv')
    

class DataImport:
    def __init__(self):
        self.paths = DataPaths

    def import_data(self,data_path):
        try:
            logging.info('Data import start')

            df = pd.read_csv(data_path)
            df['species']=df['species'].map({'setosa': 0, 'versicolor':1, 'virginica':2}) ###to pipeline

            logging.info('Dataset imported as DF')

            train_data, test_data = train_test_split(df, test_size=0.33, stratify=df['species'], random_state=42)
            
            logging.info('Train Test split')

            os.makedirs(os.path.dirname(self.paths.train_path),exist_ok=True)

            df.to_csv(self.paths.raw_path, index=False, header=True)
            train_data.to_csv(self.paths.train_path, index=False, header=True)
            test_data.to_csv(self.paths.test_path, index=False, header=True)

            logging.info('Data import done, train and test data saved separately as .csv')

            return (self.paths.train_path, self.paths.test_path)
        
        except Exception as e:
            raise CustomException(e,sys)


