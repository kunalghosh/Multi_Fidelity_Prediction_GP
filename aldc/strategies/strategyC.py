from aldc.models import SKLearnGPModel

def strategy_C(helout_set: list, batch_size: int, random_seed: int, debug=False) -> list:
    """
    Clustering:
    1. Cluster the held out set, into as many clusters as the next batch_size
    2. Pick molecule closest to the cluster centers
    
    debug : boolean
    Used to enable debug logs and plots.
    """
    pass
