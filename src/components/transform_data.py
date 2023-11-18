import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler, OrdinalEncoder

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

@dataclass
class PreprocessorPath:
    preprocessor_path=os.path.join('artifacts',"preprocessor.pkl")

class DataTransformationPreprocessor:
    def __init__(self):
        self.data_transformation_path=PreprocessorPath()
    
    def transformer_object(self):

        try:
            numerical_columns = ['elevation', 'sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'sepal_area', 'petal_area', 'sepal_aspect_ratio', 'petal_aspect_ratio', 'sepal_to_petal_length_ratio', 'sepal_to_petal_width_ratio', 'sepal_petal_length_diff', 'sepal_petal_width_diff', 'petal_curvature_mm', 'petal_texture_trichomes_per_mm2', 'leaf_area_cm2', 'sepal_area_sqrt', 'petal_area_sqrt', 'area_ratios']
            categorical_columns = ['soil_type']
            target=['species']

            logging.info(f"Cat. cols: {categorical_columns}")
            logging.info(f"Num. cols: {numerical_columns}")

            cat_data_transform = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('encoder', OneHotEncoder(sparse_output=False, drop="if_binary", handle_unknown="ignore"))
                ]
            )

            num_data_transform = Pipeline(
                steps = [
                    ('imputer', SimpleImputer(strategy='median'))
                ]
            )

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_data_transform,numerical_columns),
                ("cat_pipeline",cat_data_transform,categorical_columns)

                ],
                remainder='passthrough',
                verbose_feature_names_out=False


            )

            preprocessor.set_output(transform='pandas')

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self, train_path,test_path):
        try:
            logging.info('Data transformation started')
            logging.info('Train and test data loading for transformation')

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Obtaining preprocessor obj.")

            transformer_obj = self.transformer_object()

            logging.info("Applying preprocessor obj. on train and test DFs")

            train_transformed_df=transformer_obj.fit_transform(train_df)
            test_transformed_df=transformer_obj.transform(test_df)

            logging.info('Train and test data transformed')

            save_object(

                file_path=self.data_transformation_path.preprocessor_path,
                obj=transformer_obj

            )

            logging.info(f"Preprocessor obj. saved")

            train_transformed_df.to_csv(os.path.join('artifacts', 'train_preprocessed.csv'), index=False, header=True)
            test_transformed_df.to_csv(os.path.join('artifacts', 'test_preprocessed.csv'), index=False, header=True)

            logging.info("Preprocessed train and test DFs saved as cvs")

            return train_transformed_df, test_transformed_df, self.data_transformation_path.preprocessor_path,

        except Exception as e:
            raise CustomException(e,sys)

