import os
import sys
from src.exception import CustomException
from src.logger import logging

from src.components.transform_data import DataTransformationPreprocessor
from src.components.transform_data import PreprocessorPath

from src.components.model_trainer1 import ModelPath
from src.components.model_trainer1 import TrainModel
from src.components.data_import import DataImport
import optuna

if __name__=="__main__":
    obj=DataImport()
    train_data,test_data=obj.import_data('data\data.csv')

    data_transformation=DataTransformationPreprocessor()
    train_df,test_df,_=data_transformation.initiate_data_transformation(train_data,test_data)


    modeltrainer=TrainModel()

    modeltrainer.initiate_model_trainer(train_df,test_df)
    
    study = optuna.create_study(
    direction="maximize",
    study_name='XGB_classifier_model',
    sampler=optuna.samplers.TPESampler(),
    )

    study.optimize(modeltrainer.objective, n_trials=5)