from aldc.strategies import StrategyGetter

strategy_getter = StrategyGetter()

def test_dummy_strategy_A():
    strategyA = strategy_getter.get_strategy("strategy_A")
    output = strategyA(heldout_set = [1,2,3,4], batch_size=1, random_seed=1)
    assert output == [1,2,3,4]
