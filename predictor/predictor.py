from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import time 
import pandas as pd
import numpy as np

from helpers import Normalize

class Predictor(object):

    def __init__(self, model):
        self.model = model

    def execute(self, dataset, datasetcomplete):
        X = dataset
        X, X = Normalize(X, X)
        pred_result = self.model.predict(X)
        self.prediction = dataset.iloc[np.nonzero(pred_result)]

    def getPrediction(self):
        return self.prediction
            