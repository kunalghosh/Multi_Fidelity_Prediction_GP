class Model(object):
  """docstring for Model."""

  def __init__(self):
    super(Model, self).__init__()
    self.params = dict() # dictionary of model parameters

  def train(self):
    pass

  def predict(self, input_data):
    pass

  def load_params(self, path: str):
    pass

  def save_params(self, path: str):
    pass

  def get_params(self):
    return self.params
