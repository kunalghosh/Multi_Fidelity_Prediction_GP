from aldc.models import SKLearnGPModel


def strategy_F(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
  """
  Combination of A. and B.
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the _top half_ of molecules with the highest uncertainty
    4. Randomly pick as many molecules as the next batch_size (NOTE !! The figure needs to be updated.)

  debug : boolean
    Used to enable debug logs and plots.
  """
  pass
