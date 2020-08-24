from aldc.models import SKLearnGPModel


def strategy_F(gp: SKLearnGPModel, heldout_set: list, batch_size: int, random_seed:int, mbtr_data, debug=False) -> list:
    """
    Combination of A. and B.
    1. Use the GP trained on the previous batch to make predictions on the held out set.
    2. Sort molecules based on prediction uncertainty.
    3. Pick the _top half_ of molecules with the highest uncertainty
    4. Randomly pick as many molecules as the next batch_size (NOTE !! The figure needs to be updated.)

    debug : boolean
    Used to enable debug logs and plots.
    """
    from . import StrategyGetter
    strategy_getter = StrategyGetter()

    strategy_B = strategy_getter.get_strategy("strategy_B")

    strategy_A = strategy_getter.get_strategy("strategy_A")

    n_high = int(len(heldout_set)/2.0)
    if batch_size > n_high:
      n_high = batch_size 

    prediction_set = strategy_B(gp, heldout_set, n_high, random_seed, mbtr_data, debug)
    
    prediction_set = strategy_A(prediction_set, batch_size, random_seed, debug) 
    
    return prediction_set
