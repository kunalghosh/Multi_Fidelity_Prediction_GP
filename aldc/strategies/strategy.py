from aldc.strategies.strategyA import strategy_A
from aldc.strategies.strategyB import strategy_B
from aldc.strategies.strategyC import strategy_C
from aldc.strategies.strategyD import strategy_D
from aldc.strategies.strategyE import strategy_E
from aldc.strategies.strategyF import strategy_F
from aldc.strategies.strategyG import strategy_G

class StrategyGetter:
    def __init__(self, logger=None):
        super(StrategyGetter, self).__init__()
        self.strategyA = strategy_A
        self.strategyB = strategy_B
        self.strategyC = strategy_C
        self.strategyD = strategy_D
        self.strategyE = strategy_E
        self.strategyF = strategy_F
        self.strategyG = strategy_G

    def get_strategy(self, name:str):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise RuntimeError(f"Strategy with name {name} doesn't exist")
