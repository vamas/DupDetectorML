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

from helpers import SaveModel, RestoreModel, Normalize, RemoveOutliers

class Trainer(object):
    
    def __init__(self, dataset, test_size):
        self.dataset = dataset
        self.test_size = test_size

    def execute(self):
        self.testTrainSplit()
        self.train()
        return self.bestModel()
    
    def testTrainSplit(self):
        X = self.dataset.copy()
        del X['Label']
        y = self.dataset['Label']

        X, X = Normalize(X, X)
        X, y = RemoveOutliers(X, y, residual_threshold = 1.5) 

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.test_size, random_state=42)

    def train(self):
        classifiers = {
              'clf1':LogisticRegression(), 
              'clf2':RidgeClassifier(),
              # 'clf3':AdaBoostClassifier(), 
              'clf4':SVC()
             }

        PARAM_RANGE_GAMMA = [0.001, 0.01, 0.1]
        PARAM_RANGE_C = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        N_FEATURES_OPTIONS = [5, 10]

        param_grids = {
                    'clf1':[{'pca__n_components': N_FEATURES_OPTIONS, 'clf__C': PARAM_RANGE_C}], #LinearRegression
                    'clf2':[{'pca__n_components': N_FEATURES_OPTIONS, 'clf__alpha': [0.0001, 0.001, 0.01, 0.1]}], #Ridge
                    'clf3':[{'pca__n_components': N_FEATURES_OPTIONS, 'clf__base_estimator':[LogisticRegression()],  #AdaBoost
                                'clf__n_estimators': [256, 512],
                                'clf__learning_rate': [0.3, 0.4]}],
                    'clf4':[{'pca__n_components': N_FEATURES_OPTIONS, 'clf__C': [10, 25, 50], 'clf__gamma': PARAM_RANGE_GAMMA, 
                                'clf__kernel': ['linear','rbf']
                            }]
                    }

        param_grids = {
                    'clf1':[{'clf__fit_intercept': [True, False], 'clf__C': PARAM_RANGE_C}], #LinearRegression
                    'clf2':[{'clf__alpha': [0.0001, 0.001, 0.01, 0.1]}], #Ridge
                    'clf3':[{'clf__base_estimator':[LogisticRegression()],  #AdaBoost
                                'clf__n_estimators': [256, 512],
                                'clf__learning_rate': [0.3, 0.4]}], #Ridge
                    'clf4':[{'clf__C': [10, 25, 50], 'clf__gamma': PARAM_RANGE_GAMMA, 'clf__kernel': ['linear','rbf']}] #SVC
                    }
        scores = []

        for clf in classifiers:
            classifier = classifiers[clf]
            param_grid = param_grids[clf]
            pipe_lr = Pipeline([
                            #('pca', PCA),                    
                            ('clf', classifier)
                        ])
            
            print(classifier)    
            
            start = time.time() 
            
            gs = GridSearchCV(estimator=pipe_lr, 
                            param_grid=param_grid, 
                            scoring='f1_weighted',
                            n_jobs=-1)   
            gs = gs.fit(self.X_train, self.y_train) 

            end = time.time()

            scores.append([
                        classifier.__class__.__name__,  end - start, 
                        gs.best_params_, 
                        gs.best_estimator_,
                        gs.best_score_
                        ])
                
                
        self.scores_tuned = pd.DataFrame(scores, columns=['Algorithm','Time',
                                                    #'Acc train','F1 train','Acc test','F1 test',
                                                    'Parameters','Estimator','Score'])

    def bestModel(self):
        best_model = self.scores_tuned.iloc[self.scores_tuned['Score'].argmax()]
        print(best_model)
        model = best_model['Estimator']

        model.fit(self.X_train, self.y_train)
        result = model.predict(self.X_test)

        print("Accuracy: {0}".format(accuracy_score(self.y_test, result)))
        print("F1: {0}".format(f1_score(self.y_test, result)))

        return model
