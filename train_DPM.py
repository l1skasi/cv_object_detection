import numpy as np
import json
from sklearn.metrics import classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from skimage.feature import hog
from sklearn.model_selection import GridSearchCV
import pandas as pd
import cv2
import random

def hyperparams_tuning(model,x_train,y_train):
    params  =  {'max_depth': [3, 6, 9, 12],
                'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
                'n_estimators': [25, 50, 75, 100],
                'colsample_bytree': [0.3, 0.4, 0.5, 0.7],
                'reg_lambda':[0,1],
                'reg_alpha':[30,40,50,60,70],
                'gamma':[1,2,5,8],}

    clf = GridSearchCV(estimator=model, 
                   param_grid=params, 
                   cv = 3,
                   n_jobs = -1,
                   verbose=1)
    
    clf.fit(x_train, y_train)
    print(clf.best_params_)

def check_accuracy(model,x_test,y_test):
    y_pred = model.predict(x_test)
    print("Accuracy: "+str(accuracy_score(y_test, y_pred))+'\n')
    print(classification_report(y_test.ravel(), y_pred))

