import time
from sklearn import preprocessing
from io_utils import append_write, out_time

def desc_pp(preprocess, X_train, X_test):
    if preprocess == "none":
        X_train_pp = X_train.toarray()
        X_test_pp = X_test.toarray()

    elif preprocess == "norm":
        X_train_pp = preprocessing.normalize(X_train.toarray())
        X_test_pp = preprocessing.normalize(X_test.toarray())
        
    elif preprocess == "standard":
        X_train_pp = preprocessing.scale(X_train.toarray())
        X_test_pp = preprocessing.scale(X_test.toarray())

#    elif preprocess == "norm2":
#        x_mean, x_std = X_train.toarray().mean(axis=0), X_train.toarray().std(axis=0)
#        x_std = x_std + 1 # to ensure we don't divide by zero. if std is 0    
#        X_train = (X_train-x_mean)/x_std
#        X_test = (X_test-x_mean)/x_std
#        X_train_pp = X_train
#        X_test_pp = X_test


    return X_train_pp, X_test_pp
