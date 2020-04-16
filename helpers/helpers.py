import pickle
from sklearn.linear_model import RANSACRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def SaveModel(model, filename):
    pickle.dump(model, open(filename, 'wb'))
    
def RestoreModel(filename):
    return pickle.load(open(filename, 'rb'))

def Normalize(X_train, X_test):
    imp = MinMaxScaler()
    X_train_norm = imp.fit_transform(X_train)
    X_test_norm = imp.transform(X_test)
    return X_train_norm, X_test_norm

def RemoveOutliers(X_train, y_train, residual_threshold = 5):
    X = X_train.copy()
    y = y_train.copy()
    ransac = RANSACRegressor(RandomForestRegressor(), 
                             max_trials=1000, 
                             min_samples=50, 
                             loss='squared_loss', 
                             residual_threshold=residual_threshold, 
                             random_state=0)

    

    ransac.fit(X, y)
    return X[ransac.inlier_mask_], y[ransac.inlier_mask_]    
