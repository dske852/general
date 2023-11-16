import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

import optuna

X, y = load_iris(return_X_y=True)
X_train, X_valid, y_train, y_valid = train_test_split(X, y)

class sk:
    def __init__(self):
        pass


    def objective(self, trial):
        clf = SGDClassifier(random_state=0)
        for step in range(100):
            clf.partial_fit(X_train, y_train, np.unique(y))
            intermediate_value = clf.score(X_valid, y_valid)
            trial.report(intermediate_value, step=step)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return clf.score(X_valid, y_valid)

obj=sk()

study = optuna.create_study(direction="maximize")
study.optimize(obj.objective, n_trials=2)

#if __name__=="__main__":
    #obj=skuska()
    #obj.sk1()

class trainer:
    def __init__(self):
        pass
    def data(self,c,d):
        X_train=c
        y=d
        return X_train, y

    def model(self):
        a,b=self.data()
        return print(a,b)

    
    #def model(self):
        #X_train,y_train=self.initiate_model_trainer()
    

obj=trainer()
obj.data('a','b')

obj.model()


class Test():
    def abc(self,a,b,c):
        a = a
        b = b
        c = c
        return a,b,c
    def defg(self):
        print('inside defg second function')
        a,b,c = self.abc()
        return a,b,c

c = Test()
c.defg()



    