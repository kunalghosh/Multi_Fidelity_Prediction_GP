import json

def loadjson(path):
    with open(path, "r") as json_file:
        json_data = json_file.read()
    return json.loads(json_data)
