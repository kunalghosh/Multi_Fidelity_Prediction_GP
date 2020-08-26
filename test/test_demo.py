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

#tests
def test_strategy_A_edge_case():
    #take all the items to a batch
    strategyA = strategy_getter.get_strategy("strategy_A")
    output = strategyA(heldout_set = [1,2,3,4], batch_size=4, random_seed=1)
    assert sorted(output) == [1,2,3,4]


#check what happens in edge case
def test_strategy_B_edge_case():
    strategy_X_edge_case_test("strategy_B", True, 7676)

#check what happens in edge case
def test_strategy_C_edge_case():
    strategy_X_edge_case_test("strategy_C", False, 76)


def test_strategy_C_clustering_edge():
    strategyC = strategy_getter.get_strategy("strategy_C")
    batch_size = 1
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    prediction_set = strategyC(heldout_set, batch_size, 33, mbtr_data) 
    assert len(prediction_set) == 1 and prediction_set[0] in heldout_set

def test_strategy_D_edge_case():
    strategy_X_edge_case_test("strategy_D", True, 7676)


def test_strategy_E_edge_case():
    strategy_X_edge_case_test("strategy_E", True, 754)


def test_strategy_F_edge_case():
    strategy_X_edge_case_test("strategy_F", True, 754)


def test_strategy_G_edge_case():
    strategy_X_edge_case_test("strategy_G", True, 754)

def test_save_old_A():
    batch_size = 11
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


















    
    
    


    
