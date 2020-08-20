from aldc.models import SKLearnGPModel
from aldc.utils import sort
import numpy as np

def strategy_B(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int,mbtr_data ,debug=False) -> list:
  """
  Uncertainty:
  1. Use the GP trained on the previous batch to make predictions on the held out set.
  2. Sort molecules based on prediction uncertainty.
  3. Pick the molecules with the highest uncertainty
  
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
      X_train_pp = X_train.toarray()
    

      #making predictions on th entire dataset
      mu_s, std_s = gp.predict(X_train_pp) #mu->mean? yes
  
      sorted_data = sort(prediction_set ,std_s)

      K = len(sorted_data) - batch_size
      prediction_set = sorted_data[K:]
  
      #heldout_set = np.setdiff1d(heldout_set, prediction_set)
  
  return prediction_set
