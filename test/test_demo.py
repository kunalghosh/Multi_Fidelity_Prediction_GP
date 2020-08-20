from aldc.strategies import StrategyGetter
from aldc.models import SKLearnGPModel 
import numpy as np
from scipy.sparse import load_npz
import os

strategy_getter = StrategyGetter()

def test_strategy_A_edge_case_1():
    #take all the items to a batch
    strategyA = strategy_getter.get_strategy("strategy_A")
    output = strategyA(heldout_set = [1,2,3,4], batch_size=4, random_seed=1)
    assert sorted(output) == [1,2,3,4]

def test_strategy_A_right_size(): 
    #base case: should produce right output
    strategyA = strategy_getter.get_strategy("strategy_A")
    heldout_set = list(np.linspace(0,99,100, dtype=int))
    batch_size = 88
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=88)
    assert len(prediction_set) == batch_size 

def test_strategy_A_prediction_set_sensible():
    strategyA = strategy_getter.get_strategy("strategy_A")
    heldout_set = list(np.linspace(0,99,100, dtype=int))
    batch_size = 57
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=88)
    assert all(pred_id in heldout_set for pred_id in prediction_set)  and len(prediction_set) == len(set(prediction_set))


#helper function
def fit_gp():
    gp = SKLearnGPModel("constRBF")
    train_x = load_npz("mbtr_train_set.npz")
    train_x = train_x.toarray()
    train_y = np.loadtxt("homo_lowfid_train_set.txt")
    if os.path.isfile("gp_params_test.txt"):
        file = open("gp_params_text.txt",'r')
        const = file.readline()
        lenght = file.readline()
        gp.kernel.k1.constant_value = const
        gp.kernel.k2.length_scale = lenght
        gp.fit(train_x, train_y)
    else:
        gp.fit(train_x, train_y)
        params = gp.get_params()
        file = open("gp_params_text.txt",'w+')
        file.write(str(params["constant_value"]) + "\n")
        file.write(str(params["length_scale"]) + "\n")
    return gp

#check what happens in edge case
def test_strategy_B_edge_case_1():
    strategyB = strategy_getter.get_strategy("strategy_B")
    gp = fit_gp()
    heldout_set = list(np.linspace(0,99,100, dtype=int))  
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    #homo_lowfid = np.loadtxt("homo_lowfid_toy_set.txt")
    batch_size = 100
    prediction_set = strategyB(gp, heldout_set, batch_size, 88, mbtr_data)
    assert sorted(prediction_set) == heldout_set

#check that the output is the right size
def test_strategy_B_right_size():
    strategyB = strategy_getter.get_strategy("strategy_B")
    gp = fit_gp()
    batch_size = 57
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,99,100, dtype=int)) 
    prediction_set = strategyB(gp, heldout_set, batch_size, 8, mbtr_data) 
    assert len(prediction_set) == batch_size 


#check that the output set is sensible, subset of heldout and no duplicates
def test_strategy_B_sensible_output():
    strategyB = strategy_getter.get_strategy("strategy_B")
    gp = fit_gp()
    batch_size = 64
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,99,100, dtype=int)) 
    prediction_set = strategyB(gp, heldout_set, batch_size, 8, mbtr_data) 
    assert all(pred_id in heldout_set for pred_id in prediction_set)  and len(prediction_set) == len(set(prediction_set))


#check that the mean uncertainty of selected set is higher than heldout
def test_strategy_B_higher_uncertainty():
    strategyB = strategy_getter.get_strategy("strategy_B")
    gp = fit_gp()
    batch_size = 73
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,99,100, dtype=int)) 
    X_train_heldout = (mbtr_data[heldout_set, :]).toarray()
    mu_heldout, std_heldout = gp.predict(X_train_heldout)
    seeds = [22,54,13423]
    std_predictions = []
    for seed in seeds:
        prediction_set = strategyB(gp, heldout_set, batch_size, seed, mbtr_data) 
        X_train_prediction = (mbtr_data[prediction_set, :]).toarray()
        mu_prediction, std_prediction = gp.predict(X_train_prediction)
        std_predictions.append(std_prediction)
    #check that the mean uncertainty is higher
    assert [np.mean(std_pred) >= np.mean(std_heldout) for std_pred in std_predictions]

#check what happens in edge case
def test_strategy_C_edge_case():
    strategyC = strategy_getter.get_strategy("strategy_C")
    heldout_set = list(np.linspace(0,99,100, dtype=int))  
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    #homo_lowfid = np.loadtxt("homo_lowfid_toy_set.txt")
    batch_size = 100
    prediction_set = strategyC(heldout_set, batch_size, 14, mbtr_data)
    assert sorted(prediction_set) == heldout_set

#check that the output is the right size
def test_strategy_C_right_size():
    strategyC = strategy_getter.get_strategy("strategy_C")
    batch_size = 36
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,99,100, dtype=int)) 
    prediction_set = strategyC(heldout_set, batch_size, 16, mbtr_data) 
    assert len(prediction_set) == batch_size 


#check that the output set is sensible, subset of heldout and no duplicates
def test_strategy_C_sensible_output():
    strategyC = strategy_getter.get_strategy("strategy_C")
    batch_size = 29
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,99,100, dtype=int)) 
    prediction_set = strategyC(heldout_set, batch_size, 27, mbtr_data) 
    assert all(pred_id in heldout_set for pred_id in prediction_set)  and len(prediction_set) == len(set(prediction_set))














    
    
    


    
