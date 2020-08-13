from aldc.models import SKLearnGPModel

def strategy_G(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
  """
  Cluster and Uncertainty:
    1. Cluster the entire held out set, into as many clusters as the next batch_size.
    2. Make predictions for the entire held out set. (1 and 2 can happen independently.)
    3. Pick the molecule with the highest uncertainty in each cluster (if cluster has one element pick that).

    debug : boolean
      Used to enable debug logs and plots.
  """
  pass
