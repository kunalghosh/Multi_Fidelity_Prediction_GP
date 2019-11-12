import time
from sklearn import preprocessing
from io_utils import append_write, out_time

def desc_pp(preprocess, X_train, X_test):
    #-- normalization
#    start = time.time()
#    append_write(out_name, "starting preprocess \n")    

    if preprocess == "none":
        X_train_pp = X_train.toarray()
        X_test_pp = X_test.toarray()

    elif preprocess == "norm":
        X_train_pp = preprocessing.normalize(X_train.toarray())
        X_test_pp = preprocessing.normalize(X_test.toarray())
        
    elif preprocess == "standard":
        X_train_pp = preprocessing.scale(X_train.toarray())
        X_test_pp = preprocessing.scale(X_test.toarray())

#    process_time = time.time() - start
#    out_time(out_name, process_time)        

    return X_train_pp, X_test_pp
