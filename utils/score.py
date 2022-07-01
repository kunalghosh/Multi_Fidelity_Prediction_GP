def r2_byhand(y_test, mu_s):
    y_true = y_test
    y_pred = mu_s
    u = ((y_true - y_pred)**2).sum()
    v = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1-u/v
    return r2
