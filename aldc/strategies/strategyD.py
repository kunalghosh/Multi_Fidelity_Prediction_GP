from aldc.models import SKLearnGPModel
#from aldc.strategies import stra


def strategy_D(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed: int, mbtr_data, debug=False) -> list:
    
    from . import StrategyGetter
    strategy_getter = StrategyGetter()

    strategy_B = strategy_getter.get_strategy("strategy_B")

    strategy_C = strategy_getter.get_strategy("strategy_C")

    n_high = int(len(heldout_set)/2.0)
    if batch_size > n_high:
      n_high = batch_size 
    
    prediction_set = strategy_B(gp,heldout_set, n_high, random_seed, mbtr_data, debug)
        
    prediction_set = strategy_C(prediction_set, batch_size, random_seed, mbtr_data, debug)
              
  
    return list(prediction_set)

