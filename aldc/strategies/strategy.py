from aldc.strategies.strategyA import strategy_A
from aldc.strategies.strategyB import strategy_B
from aldc.strategies.strategyC import strategy_C
from aldc.strategies.strategyD import strategy_D
from aldc.strategies.strategyE import strategy_E
from aldc.strategies.strategyF import strategy_F
from aldc.strategies.strategyG import strategy_G

class StrategyGetter:
    def __init__(self):
        super(StrategyGetter, self).__init__()
        self.strategy_A = strategy_A
        self.strategy_B = strategy_B
        self.strategy_C = strategy_C
        self.strategy_D = strategy_D
        self.strategy_E = strategy_E
        self.strategy_F = strategy_F
        self.strategy_G = strategy_G
    
    def get_strategy(self, name:str):
        if hasattr(self, name):
            return getattr(self, name)
        else:
            raise RuntimeError(f"Strategy with name {name} doesn't exist")


