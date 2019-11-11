from sklearn import preprocessing

def desc_pp(preprocess, X_train, X_test):
    #-- normalization
    #start = time.time()
    #f.write("starting preprocess " + "\n"),f.flush()
    if preprocess == "none":
        X_train_pp = X_train.toarray()
        X_test_pp = X_test.toarray()

    elif preprocess == "norm":
        X_train_pp = preprocessing.normalize(X_train.toarray())
        X_test_pp = preprocessing.normalize(X_test.toarray())
        
    elif preprocess == "standard":
        X_train_pp = preprocessing.scale(X_train.toarray())
        X_test_pp = preprocessing.scale(X_test.toarray())
        
    return X_train_pp, X_test_pp
