from aldc.models import SKLearnGPModel
from aldc.utils import cluster, get_closest_to_center
import numpy as np


def strategy_C(heldout_set: list, batch_size: int, random_seed: int, mbtr_data, debug=False) -> list:
    """
    Clustering:
    1. Cluster the held out set, into as many clusters as the next batch_size
    2. Pick molecule closest to the cluster centers
    
    debug : boolean
    Used to enable debug logs and plots.
    """
    labels, centers = cluster(heldout_set, batch_size, random_seed, mbtr_data)

    closest = get_closest_to_center(heldout_set, labels, centers,mbtr_data)

    #-- Calculate centers
    prediction_set = np.array(heldout_set)[closest]

    #heldout_set = np.setdiff1d(heldout_set, prediction_set )
    
    return list(prediction_set)
