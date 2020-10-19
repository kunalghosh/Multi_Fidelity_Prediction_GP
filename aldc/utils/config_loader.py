import json
from aldc.utils import Config

def get_config_from_json(json_file):
    with open(json_file, "r") as json_file:
      json_data = json_file.read()
    json_config = json.loads(json_data)
    config = Config(**json_config)
    return config
