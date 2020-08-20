from aldc.strategies import StrategyGetter
from aldc.models import SKLearnGPModel 
import numpy as np
from scipy.sparse import load_npz

strategy_getter = StrategyGetter()

def test_dummy_strategy_A():
    strategyA = strategy_getter.get_strategy("strategy_A")

    #edge case: batch_size == size of out out
    output = strategyA(heldout_set = [1,2,3,4], batch_size=4, random_seed=1)
    assert sorted(output) == [1,2,3,4]


    #edge case: only one elment
    heldout_set = [1,2,3,4]
    prediction_set = strategyA(heldout_set=heldout_set, batch_size=1, random_seed=12)
    aqcuisition_basic_tests(1, prediction_set, heldout_set)  


    #base case: should produce right output
    heldout_set = list(np.linspace(0,99,100, dtype=int))
    batch_size = 88
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=88)
    aqcuisition_basic_tests(batch_size, prediction_set, heldout_set)

def aqcuisition_basic_tests(batch_size, prediction_set, heldout_set):
    #prediction_set is a right size 
    assert len(prediction_set) == batch_size 
    #predicition set is a subset of heldout 
    assert all(pred_id in heldout_set for pred_id in prediction_set) 
    #test that there are no duplicates   
    assert len(prediction_set) == len(set(prediction_set))

def test_strategy_B():
    strategyB = strategy_getter.get_strategy("strategy_B")
    gp = SKLearnGPModel("constRBF")
    train_x = load_npz("mbtr_train_set.npz")
    train_y = np.loadtxt("homo_lowfid_train_set.txt")
    gp.fit(train_x.toarray(), train_y)
    heldout_set = list(np.linspace(0,99,100, dtype=int))  
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    homo_lowfid = np.loadtxt("homo_lowfid_toy_set.txt")

    #edge case: all included
    batch_size = 100
    prediction_set = strategyB(gp, heldout_set, batch_size, 88, mbtr_data)
    assert sorted(prediction_set) == heldout_set
    
    #edge case: only one in batch
    batch_size = 1
    prediction_set = strategyB(gp, heldout_set, batch_size, 8, mbtr_data)
    aqcuisition_basic_tests(batch_size, prediction_set, heldout_set)

    #base case
    batch_size = 57
    prediction_set = strategyB(gp, heldout_set, batch_size, 8, mbtr_data) 
    aqcuisition_basic_tests(batch_size, prediction_set, heldout_set)
    


    
    
    


    
