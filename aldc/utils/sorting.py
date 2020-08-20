def sort(data: list, based_on: list) -> list:
  return [data_point for _, data_point in sorted(zip(based_on, data))]

def get_last_x(data: list, top_k: float) -> list:
  """
  if top_k = 0.5, then return second half, if top_k = 1/3 then return last 1/3 of data.
  """
  K = len(data) - int(len(data)*top_k)
  return data[K:]


def sort_and_get_last_x(data: list, based_on: list, top_k: float) -> list:
  sorted_data = sort(data, based_on)
  last_x = get_last_x(sorted_data, top_k)
  return last_x
