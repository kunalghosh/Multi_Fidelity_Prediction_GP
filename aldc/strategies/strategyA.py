from sklearn.model_selection import train_test_split
import numpy as np

def strategy_A(heldout_set: list, batch_size: int, random_seed: int, debug=False) -> list:
  """
  Random strategy :
    1. Pick molecules randomly from the held out set.
  """
  if batch_size == len(heldout_set):
      prediction_set = heldout_set
      heldout_set = np.array([])
  else:
      prediction_set, heldout_set = train_test_split(heldout_set, train_size = batch_size, random_state=random_seed)
  #prediction_set, heldout_set = pre_rem_split(batch_size, heldout_set, random_seed) 
  return prediction_set
