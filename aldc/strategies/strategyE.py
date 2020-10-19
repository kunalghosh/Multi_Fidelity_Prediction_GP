from aldc.models import SKLearnGPModel
from aldc.utils import sort_and_get_last_x, get_last_x
from aldc.utils import cluster
import numpy as np

def strategy_E(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, mbtr_data, debug=False, logger=None) -> list:
    """
    Combination of B. and C.
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the _top half_ of molecules with the highest uncertainty
    4. Cluster the set into as many clusters as the next batch_size.
    5. Sort each cluster based on uncertainty and pick molecule with highest uncertainty.

    debug : boolean
    Used to enable debug logs and plots.
    """

    if len(heldout_set) == batch_size:
        prediction_set = heldout_set
        heldout_set = np.array([])
    else:
        prediction_set = heldout_set

        X_train = mbtr_data[prediction_set, :]

        #-- Preprocessing
        #X_train_pp = desc_pp_notest(preprocess, X_train)
        X_train = X_train.toarray()

        #making predictions on th entire dataset
        mu_s, std_s = gp.predict(X_train)

        n_high = int(len(heldout_set)/2.0)
        if batch_size > n_high:
            n_high = batch_size

        prediction_set = sort_and_get_last_x(prediction_set ,std_s, n_high)

        std_s_sorted = list(sorted(std_s))

        prediction_set_std_s = get_last_x(std_s_sorted, n_high)


        labels, _ = cluster(prediction_set, batch_size, random_seed, mbtr_data)

        pick_idxs = np.empty(batch_size, dtype = int)

        for j in range(batch_size):
            cluster_idxs = np.array(np.where(labels == j)).flatten()
            prediction_clu_idxs = np.array(prediction_set)[cluster_idxs]
            std_s_clu = np.array(prediction_set_std_s)[cluster_idxs]
            if len(std_s_clu) == 1:
                pick_idxs[j] = prediction_clu_idxs
            else:
                pick_idxs[j] = sort_and_get_last_x(prediction_clu_idxs,std_s_clu, 1)[0]
                #pick_idxs_temp = np.argpartition(-std_s_clu, 1)[:1]
                #pick_idxs[j] = prediction_clu_idxs[pick_idxs_temp]

        prediction_set =  pick_idxs

        #heldout_set = np.setdiff1d(heldout_set, prediction_set)


    return list(prediction_set)
