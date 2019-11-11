from sklearn.model_selection import train_test_split
import numpy as np

def pre_rem_split(prediction_set_size, remaining_idxs):
    if len(remaining_idxs) == prediction_set_size :
        prediction_idxs = remaining_idxs
        remaining_idxs = np.array([])
#        remaining_idxs = {}
    elif len(remaining_idxs) != prediction_set_size:
        prediction_idxs, remaining_idxs = train_test_split(remaining_idxs, train_size = prediction_set_size)#, random_state=0)
    return prediction_idxs, remaining_idxs
