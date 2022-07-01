import time
from sklearn import preprocessing
from io_utils import append_write, out_time

def desc_pp_notest(preprocess, X_train):
    #-- normalization
    #start = time.time()
    #f.write("starting preprocess " + "\n"),f.flush()
    if preprocess == "none":
        X_train_pp = X_train.toarray()

    elif preprocess == "norm":
        X_train_pp = preprocessing.normalize(X_train.toarray())

    elif preprocess == "standard":
        X_train_pp = preprocessing.scale(X_train.toarray())

    return X_train_pp
