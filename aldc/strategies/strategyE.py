from aldc.models import SKLearnGPModel

def strategy_E(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
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
  pass
