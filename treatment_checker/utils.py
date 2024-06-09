import json


def read_json(filename):
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def read_text(filename):
    with open(filename, "r") as f:
        data = f.read()
    return data


def split_on_nl(x):
    return x.split("\n")
