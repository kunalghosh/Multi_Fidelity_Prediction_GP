class Model(object):
  """docstring for Model."""

  def __init__(self, arg):
    super(Model, self).__init__()
    self.arg = arg
    self.params = dict() # dictionary of model parameters

  def fit(self, arg):
    pass

  def predict(self, arg):
    pass

  def load_params(self, path: str):
    pass

  def save_params(self, path: str):
    pass

  def get_params(self):
    return self.params
