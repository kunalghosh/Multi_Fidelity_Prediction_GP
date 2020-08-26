import numpy as np

def sort(data: list, based_on: list) -> list:
  return [data_point for _, data_point in sorted(zip(based_on, data))]

def get_last_x(data: list, top_k: int) -> list:
  """
  We take top_k molecules from sorted data:
  """
  K = len(data) - top_k
  return data[K:]


def sort_and_get_last_x(data: list, based_on: list, top_k: int) -> list:
  sorted_data = sort(data, based_on)
  last_x = get_last_x(sorted_data, top_k)
  return last_x
