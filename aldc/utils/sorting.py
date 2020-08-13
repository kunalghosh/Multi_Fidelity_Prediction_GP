def sort(data: list, based_on: list) -> list:
  pass

def get_last_x(data: list, top_k: float) -> list:
  """
  if top_k = 0.5, then return second half, if top_k = 1/3 then return last 1/3 of data.
  """
  pass

def sort_and_get_last_x(data: list, based_on: list, top_k: float) -> list:
  sorted_data = sort(data, based_on)
  last_x = get_last_x(sorted_data, top_k)
  return last_x
