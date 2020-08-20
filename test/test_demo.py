from aldc.strategies import StrategyGetter
import numpy as np

strategy_getter = StrategyGetter()

def test_dummy_strategy_A():
    strategyA = strategy_getter.get_strategy("strategy_A")

    #edge case: batch_size == size of out out
    output = strategyA(heldout_set = [1,2,3,4], batch_size=4, random_seed=1)
    assert output == [1,2,3,4]


    #edge case: only one elment
    heldout_set = [1,2,3,4]
    prediction_set = strategyA(heldout_set=heldout_set, batch_size=1, random_seed=12)
    assert len(prediction_set)==1
    assert prediction_set[0] in heldout_set


    #base case: should produce right output
    heldout_set = list(np.linspace(0,99,100, dtype=int))
    batch_size = 88
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=88)
    #prediction_set is a right size
    assert len(prediction_set) == batch_size
    #predicition set is a subset of heldout
    assert all(pred_id in heldout_set for pred_id in prediction_set)
    #test that there are no duplicates
    assert len(prediction_set) == len(set(prediction_set))

    
