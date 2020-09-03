from aldc.strategies import StrategyGetter
from aldc.models import SKLearnGPModel 
import numpy as np
from scipy.sparse import load_npz, save_npz
import os
from aldc.strategies import acq_fn
import json

strategy_getter = StrategyGetter()

########## helper functions #############

## fitting gp model with small training set

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

## Testing for edge case where batch_size == len(heldout_set)
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

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
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
    assert all(pred_id in heldout_set for pred_id in prediction_set)  and len(prediction_set) == len(set(prediction_set)) and len(prediction_set) == batch_size

## test that the uncertainty is higher
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
    assert [np.mean(std_pred) >= np.mean(std_heldout) for std_pred in std_predictions]


########## tests for acquisition strategies #############

#### strategy A ####

## Testing for edge case where batch_size == len(heldout_set) for strategy A
def test_strategy_A_edge_case_maximum_batch_size():
    strategyA = strategy_getter.get_strategy("strategy_A")
    output = strategyA(heldout_set = list(np.linspace(0,49,50, dtype=int)), batch_size=50, random_seed=1)
    assert sorted(output) == list(np.linspace(0,49,50, dtype=int))

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
def test_strategy_A_prediction_set_sensible():
    strategyA = strategy_getter.get_strategy("strategy_A")
    heldout_set = list(np.linspace(0,49,50, dtype=int))
    batch_size = 33
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=88)
    assert all(pred_id in heldout_set for pred_id in prediction_set)  and len(prediction_set) == len(set(prediction_set)) and len(prediction_set) == batch_size

## compare output of A to a reference   
def test_A_with_reference():
    batch_size = 11
    heldout_set = list(np.linspace(0,48,49, dtype=int)) 
    random_seed = 15
    with open('correct_test_strategies.json') as json_file:
        data = json.load(json_file)
    pred_idxs = data["A"]
    strategyA = strategy_getter.get_strategy("strategy_A")
    prediction_set = strategyA(heldout_set = heldout_set, batch_size = batch_size, random_seed=random_seed)
    assert sorted(pred_idxs) == sorted(prediction_set)


#### strategy B ####

## Testing for edge case where batch_size == len(heldout_set) for strategy B
def test_strategy_B_edge_case_maximum_batch_size():
    strategy_X_edge_case_test("strategy_B", True, 7676)

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
def test_strategy_B_sensible_output():
    strategy_X_sensible_output("strategy_B", True, 3589, 30)

## compare output of B to a reference       
def test_B_with_reference():
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

## check that the mean uncertainty of selected set is higher than heldout
def test_strategy_B_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_B", [22,54], 33)


## test that strategy B differs from straetegy F, uncertainty is higher than just random
def test_strategy_B_differs_from_F():
    strategyF = strategy_getter.get_strategy("strategy_F")
    strategyB = strategy_getter.get_strategy("strategy_B")
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    gp = fit_gp()
    prediction_set_F = strategyF(gp,heldout_set, batch_size, 12, mbtr_data) 
    prediction_set_B = strategyB(gp,heldout_set, batch_size, 12, mbtr_data) 
    assert not sorted(prediction_set_F) == sorted(prediction_set_B)


#### strategy C ####

## Testing for edge case where batch_size == len(heldout_set) for strategy C
def test_strategy_C_edge_case_maximum_batch_size():
    strategy_X_edge_case_test("strategy_C", False, 76)

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
def test_strategy_C_sensible_output():
    strategy_X_sensible_output("strategy_C", False, 85, 29)

## compare output of C to a reference 
def test_C_with_reference():
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

## Testing for edge case where we only take batch size of 1 for clustering
def test_strategy_C_clustering_edge():
    strategyC = strategy_getter.get_strategy("strategy_C")
    batch_size = 1
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    prediction_set = strategyC(heldout_set, batch_size, 33, mbtr_data) 
    assert len(prediction_set) == 1 and prediction_set[0] in heldout_set


#### strategy D ####    

## Testing for edge case where batch_size == len(heldout_set) for strategy D
def test_strategy_D_edge_case_maximum_batch_size():
    strategy_X_edge_case_test("strategy_D", True, 7676)

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
def test_strategy_D_sensible_output():
    strategy_X_sensible_output("strategy_D", True, 32, 11)

## compare output of D to a reference 
def test_D_with_reference():
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

## test that the output has higher mean uncertainty than heldout
def test_strategy_D_high_uncertainty():
    strategyC = strategy_getter.get_strategy("strategy_C")
    strategyD = strategy_getter.get_strategy("strategy_D")
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    gp = fit_gp()
    prediction_set_C = strategyC(heldout_set, batch_size, 33, mbtr_data) 
    prediction_set_D = strategyD(gp,heldout_set, batch_size, 33, mbtr_data) 
    assert not sorted(prediction_set_D) == sorted(prediction_set_C)

## check that the mean uncertainty of selected set is higher than heldout
def test_strategy_D_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_D", [21,51], 17)

## does strategy D differ from strategy B, the clustering effects the output
def test_strategy_D_not_strategy_B():
    strategyD = strategy_getter.get_strategy("strategy_D")
    strategyB = strategy_getter.get_strategy("strategy_B")
    batch_size = 12
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    gp = fit_gp()
    prediction_set_D = strategyD(gp,heldout_set, batch_size, 33, mbtr_data) 
    prediction_set_B = strategyB(gp,heldout_set, batch_size, 33, mbtr_data) 
    assert not sorted(prediction_set_D) == sorted(prediction_set_B)


#### strategy E ####

## Testing for edge case where batch_size == len(heldout_set) for strategy E
def test_strategy_E_edge_case_maximum_batch_size():
    strategy_X_edge_case_test("strategy_E", True, 754)

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
def test_strategy_E_sensible_output():
    strategy_X_sensible_output("strategy_E", True, 27, 18)

## compare output of E to a reference 
def test_E_with_reference():
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

## check that the mean uncertainty of selected set is higher than heldout
def test_strategy_E_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_E", [21,56], 21)

## test that strategy E differs from D, double clustering has an effect
def test_strategy_E_not_D():
    strategyE = strategy_getter.get_strategy("strategy_E")
    strategyD = strategy_getter.get_strategy("strategy_D")
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    gp = fit_gp()
    prediction_set_E = strategyE(gp,heldout_set, batch_size, 11, mbtr_data) 
    prediction_set_D = strategyD(gp,heldout_set, batch_size, 11, mbtr_data) 
    assert not sorted(prediction_set_D) == sorted(prediction_set_E)


#### strategy F ####

## Testing for edge case where batch_size == len(heldout_set) for strategy F
def test_strategy_F_edge_case_maximum_batch_size():
    strategy_X_edge_case_test("strategy_F", True, 754)

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
def test_strategy_F_sensible_output():
    strategy_X_sensible_output("strategy_F", True, 27, 18)

## compare output of F to a reference 
def test_F_with_reference():
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

## check that the mean uncertainty of selected set is higher than heldout
def test_strategy_F_high_uncertainty():
    strategy_X_higher_uncertainty("strategy_F", [121,12], 27)

## Test that F and G differ, the order shoudl matter
def test_strategy_F_and_G():
    strategyF = strategy_getter.get_strategy("strategy_F")
    strategyG = strategy_getter.get_strategy("strategy_G")
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    gp = fit_gp()
    prediction_set_G = strategyG(gp,heldout_set, batch_size, 12, mbtr_data) 
    prediction_set_F = strategyF(gp,heldout_set, batch_size, 12, mbtr_data) 
    assert  not sorted(prediction_set_F) == sorted(prediction_set_G)


#### strategy G ####

## Testing for edge case where batch_size == len(heldout_set) for strategy G
def test_strategy_G_edge_case_maximum_batch_size():
    strategy_X_edge_case_test("strategy_G", True, 754)

## test that the output is not overlapping with heldoutset, and there are no duplicates, and the size is right
def test_strategy_G_sensible_output():
    strategy_X_sensible_output("strategy_G", True, 27, 18)

## check that the mean uncertainty of selected set is higher than heldout
def test_strategy_G_higher_uncertainty():
    strategy_X_higher_uncertainty("strategy_G", [21,56], 21)

## compare output of G to a reference 
def test_G_with_reference():
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

## test that strategy G is not the same as E, first uncertainty should have an effect
def test_strategy_G_and_E():
    strategyE = strategy_getter.get_strategy("strategy_E")
    strategyG = strategy_getter.get_strategy("strategy_G")
    batch_size = 11
    mbtr_data = (load_npz("mbtr_toy_set.npz"))
    heldout_set = list(np.linspace(0,49,50, dtype=int)) 
    gp = fit_gp()
    prediction_set_G = strategyG(gp,heldout_set, batch_size, 12, mbtr_data) 
    prediction_set_E = strategyE(gp,heldout_set, batch_size, 12, mbtr_data) 
    assert  not sorted(prediction_set_E) == sorted(prediction_set_G)






































    
    
    


    
