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

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")

            self.X_train=train_array.drop('species', axis=1),
            self.y_train=train_array.loc[:,'species'],
            self.X_test=test_array.drop('species', axis=1),
            self.y_test=test_array.loc[:,'species']
            
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

        score = cross_val_score(model, self.X_train, self.y_train, cv=2,scoring=cv_metric)
        cv_metric_mean = score.mean()
        cv_metric_std = score.std()
    
        global best_score
        global best_margin_error

    # If the classification metric of the saved model is improved ...
        path_best_model = f'm_model.pickle'

        best_score = 0
        best_margin_error = 5

        if cv_metric_mean > best_score:
                save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model)
        #model.save(path_best_model)
            #with open(self.trained_model_file_path, 'wb') as study_file:
                #pickle.dump(model, file=study_file)

        # Update the classification metric.
                best_score = cv_metric_mean
                best_margin_error = cv_metric_std

    # Delete the previous model with these hyper-parameters from memory.
        del model
    
        return cv_metric_mean   

