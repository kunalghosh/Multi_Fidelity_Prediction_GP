def strategy_A(heldout_set: list, batch_size: int, random_seed: int, debug=False) -> list:
  """
  Random strategy :
    1. Pick molecules randomly from the held out set.
  """
  prediction_set, heldout_set = pre_rem_split(batch_size, heldout_set, random_seed) 
  return prediction_set
