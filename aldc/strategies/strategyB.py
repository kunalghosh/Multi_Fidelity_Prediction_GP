from aldc.models import SKLearnGPModel

def strategy_B(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, debug=False) -> list:
  """
  Uncertainty:
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the molecules with the highest uncertainty

    debug : boolean
      Used to enable debug logs and plots.
  """
  pass

