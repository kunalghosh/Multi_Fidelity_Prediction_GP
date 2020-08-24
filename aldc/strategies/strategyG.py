from aldc.models import SKLearnGPModel
from aldc.utils import sort_and_get_last_x
from aldc.utils import cluster
import numpy as np

def strategy_G(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, mbtr_data, debug=False) -> list:
    """
    Cluster and Uncertainty:
      1. Cluster the entire held out set, into as many clusters as the next batch_size.
      2. Make predictions for the entire held out set. (1 and 2 can happen independently.)
      3. Pick the molecule with the highest uncertainty in each cluster (if cluster has one element pick that).

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
        
        
        labels, _ = cluster(prediction_set, batch_size, random_seed, mbtr_data)
        
        pick_idxs = np.empty(batch_size, dtype = int)
        
        for j in range(batch_size):
            cluster_idxs = np.array(np.where(labels == j)).flatten() 
            prediction_clu_idxs = np.array(prediction_set)[cluster_idxs]
            std_s_clu = np.array(std_s)[cluster_idxs] 
            if len(std_s_clu) == 1:
                pick_idxs[j] = prediction_clu_idxs
            else:
                pick_idxs[j] = sort_and_get_last_x(prediction_clu_idxs,std_s_clu, 1)[0]
                #pick_idxs_temp = np.argpartition(-std_s_clu, 1)[:1]
                #pick_idxs[j] = prediction_clu_idxs[pick_idxs_temp]
        
        prediction_set =  pick_idxs
    return prediction_set
