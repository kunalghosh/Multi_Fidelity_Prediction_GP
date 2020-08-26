from aldc.strategies import StrategyGetter
from aldc.models import SKLearnGPModel 
import numpy as np
from scipy.sparse import load_npz, save_npz
import os
from aldc.strategies import acq_fn
import json

strategy_getter = StrategyGetter()

#helper functions
def fit_gp():
    gp = SKLearnGPModel("constRBF")
    train_x = load_npz("mbtr_train_set.npz")
    train_x = train_x.toarray()
    train_y = np.loadtxt("homo_lowfid_train_set.txt")
    if os.path.isfile("gp_params.txt"):
    #if False:
        #print("Here")
        file = open("gp_params.txt",'r')
        const = float(file.readline())
        lenght = float(file.readline())
        gp.kernel.k1.constant_value = const
        gp.kernel.k2.length_scale = lenght
        gp.fit(train_x, train_y)
    else:
        gp.fit(train_x, train_y)
        params = gp.get_params()
        file = open("gp_params.txt",'w+')
        file.write(str(params["constant_value"]) + "\n")
        file.write(str(params["length_scale"]) + "\n")
    return gp

def strategy_X_edge_case_test(name: str, with_gp, random_seed: int):
    strategy = strategy_getter.get_strategy(name)  
    heldout_set = list(np.linspace(0,49,50, dtype=int))  
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    batch_size = 50
    if with_gp:
        gp = fit_gp()
        prediction_set = strategy(gp, heldout_set, batch_size, random_seed, mbtr_data)
    else:
        prediction_set = strategy(heldout_set, batch_size, random_seed, mbtr_data)
    assert sorted(prediction_set) == heldout_set

def strategy_X_sensible_output(name: str, with_gp, random_seed: int, batch_size: int):
    strategy = strategy_getter.get_strategy(name)
    gp = fit_gp()
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    if with_gp:
        gp = fit_gp()
        prediction_set = strategy(gp, heldout_set, batch_size, random_seed, mbtr_data) 
    else:
        prediction_set = strategy(heldout_set, batch_size, random_seed, mbtr_data) 
        
    assert all(pred_id in heldout_set for pred_id in prediction_set)  and len(prediction_set) == len(set(prediction_set))

def strategy_X_right_size(name: str, with_gp, random_seed: int, batch_size: int):
    strategy = strategy_getter.get_strategy(name)
    gp = fit_gp()
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    if with_gp:
        gp = fit_gp()
        prediction_set = strategy(gp, heldout_set, batch_size, random_seed, mbtr_data) 
    else:
        prediction_set = strategy(heldout_set, batch_size, random_seed, mbtr_data)    
    assert len(prediction_set) == batch_size 

def strategy_X_higher_uncertainty(name: str, random_seeds: list, batch_size: int):
    strategy = strategy_getter.get_strategy(name)
    gp = fit_gp()
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    X_train_heldout = (mbtr_data[heldout_set, :]).toarray()
    mu_heldout, std_heldout = gp.predict(X_train_heldout)
    std_predictions = []
    for seed in random_seeds:
        prediction_set = strategy(gp, heldout_set, batch_size, seed, mbtr_data) 
        X_train_prediction = (mbtr_data[prediction_set, :]).toarray()
        mu_prediction, std_prediction = gp.predict(X_train_prediction)
        std_predictions.append(std_prediction)
    #check that the mean uncertainty is higher
    assert [np.mean(std_pred) >= np.mean(std_heldout) for std_pred in std_predictions]



#tests
def test_strategy_A_edge_case():
    #take all the items to a batch
    strategyA = strategy_getter.get_strategy("strategy_A")
    output = strategyA(heldout_set = [1,2,3,4], batch_size=4, random_seed=1)
    assert sorted(output) == [1,2,3,4]

def test_strategy_A_right_size(): 
    #base case: should produce right output
    strategyA = strategy_getter.get_strategy("strategy_A")
    heldout_set = list(np.linspace(0,49,50, dtype=int))
    batch_size = 37
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=88)
    assert len(prediction_set) == batch_size 

def test_strategy_A_prediction_set_sensible():
    strategyA = strategy_getter.get_strategy("strategy_A")
    heldout_set = list(np.linspace(0,49,50, dtype=int))
    batch_size = 33
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=88)
    assert all(pred_id in heldout_set for pred_id in prediction_set)  and len(prediction_set) == len(set(prediction_set))


#check what happens in edge case
def test_strategy_B_edge_case():
    strategy_X_edge_case_test("strategy_B", True, 7676)


#check that the output is the right size
def test_strategy_B_right_size():
    strategy_X_right_size("strategy_B", True, 15, 26)


#check that the output set is sensible, subset of heldout and no duplicates
def test_strategy_B_sensible_output():
    strategy_X_sensible_output("strategy_B", True, 3589, 30)


#check that the mean uncertainty of selected set is higher than heldout
def test_strategy_B_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_B", [22,54,13423], 33)

#check what happens in edge case
def test_strategy_C_edge_case():
    strategy_X_edge_case_test("strategy_C", False, 76)


#check that the output is the right size
def test_strategy_C_right_size():
    strategy_X_right_size("strategy_C", False, 14, 36)
 
#check that the output set is sensible, subset of heldout and no duplicates
def test_strategy_C_sensible_output():
    strategy_X_sensible_output("strategy_C", False, 85, 29)

def test_strategy_C_clustering_edge():
    strategyC = strategy_getter.get_strategy("strategy_C")
    batch_size = 1
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    prediction_set = strategyC(heldout_set, batch_size, 33, mbtr_data) 
    assert len(prediction_set) == 1 and prediction_set[0] in heldout_set

def test_strategy_D_edge_case():
    strategy_X_edge_case_test("strategy_D", True, 7676)


def test_strategy_D_right_size():
    strategy_X_right_size("strategy_D", True, 15 , 2)


#check that the output set is sensible, subset of heldout and no duplicates
def test_strategy_D_sensible_output():
    strategy_X_sensible_output("strategy_D", True, 32, 11)


#check that we are picking moelcules with high uncertainty
def test_strategy_D_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_D", [21,51,32], 17)


def test_strategy_E_edge_case():
    strategy_X_edge_case_test("strategy_E", True, 754)

def test_strategy_E_right_size():
    strategy_X_right_size("strategy_E", True, 1114, 14)

def test_strategy_E_sensible_output():
    strategy_X_sensible_output("strategy_E", True, 27, 18)

def test_strategy_E_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_E", [21,56,74], 21)

def test_strategy_F_edge_case():
    strategy_X_edge_case_test("strategy_F", True, 754)

def test_strategy_F_right_size():
    strategy_X_right_size("strategy_F", True, 1114, 14)

def test_strategy_F_sensible_output():
    strategy_X_sensible_output("strategy_F", True, 27, 18)

def test_strategy_F_high_uncertainty():
    strategy_X_higher_uncertainty("strategy_F", [121,11,12], 27)

def test_strategy_G_edge_case():
    strategy_X_edge_case_test("strategy_G", True, 754)

def test_strategy_G_right_size():
    strategy_X_right_size("strategy_G", True, 1114, 14)

def test_strategy_G_sensible_output():
    strategy_X_sensible_output("strategy_G", True, 27, 18)

def test_strategy_G_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_G", [21,56,74], 21)

def test_save_old_A():
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,48,49, dtype=int)) 
    random_seed = 15
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["A"]
    strategyA = strategy_getter.get_strategy("strategy_A")
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=random_seed)
    assert sorted(pred_idxs) == sorted(prediction_set)

def test_save_old_B():
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,48,49, dtype=int))
    random_seed = 15
    gp = fit_gp()
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["B"]
    strategy = strategy_getter.get_strategy("strategy_B")
    prediction_set = strategy(gp,heldout_set, batch_size, random_seed, mbtr_data) 
    assert sorted(pred_idxs) == sorted(prediction_set)

def test_save_old_C():
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,48,49, dtype=int)) 
    random_seed = 15
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["C"]
    strategy = strategy_getter.get_strategy("strategy_C")
    prediction_set = strategy(heldout_set, batch_size, random_seed, mbtr_data) 
    assert sorted(pred_idxs) == sorted(prediction_set)

def test_save_old_D():
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,48,49, dtype=int)) 
    random_seed = 15
    gp = fit_gp()
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["D"]
    strategy = strategy_getter.get_strategy("strategy_D")
    prediction_set = strategy(gp,heldout_set, batch_size, random_seed, mbtr_data) 
    assert sorted(pred_idxs) == sorted(prediction_set)

def test_save_old_E():
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,48,49, dtype=int)) 
    random_seed = 15
    gp = fit_gp()
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["E"]
    strategy = strategy_getter.get_strategy("strategy_E")
    prediction_set = strategy(gp,heldout_set, batch_size, random_seed, mbtr_data) 
    assert sorted(pred_idxs) == sorted(prediction_set)

def test_save_old_F():
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,48,49, dtype=int)) 
    random_seed = 15
    gp = fit_gp()
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["F"]
    strategy = strategy_getter.get_strategy("strategy_F")
    prediction_set = strategy(gp,heldout_set, batch_size, random_seed, mbtr_data) 
    assert all(pred_id in pred_idxs for pred_id in prediction_set)

def test_save_old_G():
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,48,49, dtype=int)) 
    random_seed = 15
    gp = fit_gp()
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["G"]
    strategy = strategy_getter.get_strategy("strategy_G")
    prediction_set = strategy(gp,heldout_set, batch_size, random_seed, mbtr_data) 
    assert sorted(pred_idxs) == sorted(prediction_set)


















    
    
    


    
