import gzip
import pickle
import os
import glob

# util
import yaml


def slurpjson(fn):
    import json
    with open(fn, 'r') as f:
        return json.load(f)


def mkdir(path):
    if not os.path.isdir(path):
        os.mkdir(path)
    return path


def mdir(path):
    try:
        os.makedirs(path)
        # print("Directory ", path, " Created ")
    except FileExistsError:
        print("Directory ", path, " already exists")


def save1(fn, a):
    with gzip.open(fn, 'wb', compresslevel=2) as f:
        pickle.dump(a, f, 2)


def load1(fn):
    with gzip.open(fn, 'rb') as f:
        return pickle.load(f)


def spit_json(fn, data):
    import json
    with open('{}.json'.format(fn), 'w') as outfile:
        json.dump(data, outfile)


def read_file(file_path):
    """
    Loads file content into Python object representation.
    Args:
        file_path: path to YAML file.

    Returns: dictionary representing YAML content
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        loader = yaml.Loader(f)
        return loader.get_single_data()


def get_all_folder(path):
    return glob.glob('{}/*'.format(path))


def is_file_in_filtered_criteria(fn):
    sufx = ['Karen von Schmieden.png', 'all.p']
    identif = ['User information', 'watch']
    retr = ['retrievement', 'retrieving']
    #
    status = False
    if not str(os.path.basename(fn)).endswith('.json'):
        status = True
    elif any(sf in str(fn).split('_') for sf in sufx):
        status = True
    elif any(idf in str(fn).split('_') for idf in identif):
        status = True
    elif any(r in os.path.basename(fn) for r in retr):
        status = True

    return status


def spit_json():
    return None
