import os
import sys
from dataclasses import dataclass

from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import optuna
import xgboost as xgb

import pickle

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            self.X_train,self.y_train,self.X_test,self.y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            #return X_train,y_train,X_test,y_test
        
        except Exception as e:
            raise CustomException(e,sys)

    def objective(self, trial):
    
        model_name = trial.suggest_categorical("classifier", ['xgb'])#"lgbm",
    
        X_train=self.X_train
        y_train=self.y_train


        if model_name =="xgb":
            xgb_n_estimators = trial.suggest_int("xgb_n_estimators", 200, 2500)
            xgb_max_depth = trial.suggest_int("xgb_max_depth", 1, 10)
            xgb_learning_rate = trial.suggest_float("xgb_learning_rate", 0.01, 0.99)
            xgb_gamma = trial.suggest_float("xgb_gamma", 0.01, 10)
            xgb_subsample = trial.suggest_float("xgb_subsample", 0.50, 0.90)
            xgb_colsample_bytree = trial.suggest_float("xgb_colsample_bytree", 0.50, 0.90)
            xgb_colsample_bynode = trial.suggest_float("xgb_colsample_bynode", 0.50, 0.90)
        
            model = xgb.XGBClassifier( random_state=42,use_label_encoder=False,
                n_estimators=xgb_n_estimators,
                max_depth=xgb_max_depth,
                learning_rate=xgb_learning_rate,
                gamma=xgb_gamma,
                subsample=xgb_subsample,
                colsample_bytree=xgb_colsample_bytree,
                colsample_bynode=xgb_colsample_bynode
        )

    # starting point for the optimization
        cv_metric = "f1_weighted"

        score = cross_val_score(model, X_train, y_train, cv=5,scoring=cv_metric)
        cv_metric_mean = score.mean()
        cv_metric_std = score.std()
    
        global best_score
        global best_margin_error

    # If the classification metric of the saved model is improved ...
        path_best_model = f'm_model.pickle'

        best_score = 0
        best_margin_error = 5

        if cv_metric_mean < best_score:
        #model.save(path_best_model)
            with open(path_best_model, 'wb') as study_file:
                pickle.dump(model, file=study_file)

        # Update the classification metric.
            best_score = cv_metric_mean
            best_margin_error = cv_metric_std

    # Delete the previous model with these hyper-parameters from memory.
        del model
    
        return cv_metric_mean    


    
