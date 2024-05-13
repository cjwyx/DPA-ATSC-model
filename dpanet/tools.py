import numpy as np
import os
from datetime import datetime
import torch

def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
def speeddata_transform(data:np.ndarray):
    '''
    data: 必须是raw的数据，必须是大于0
    '''
    return 1 / (1 + data)


def merge_dicts(*dict_args):
    result = {}
    for dictionary in dict_args:
        if not isinstance(dictionary, dict):
            raise ValueError("All arguments must be dictionaries.")
        for key in dictionary:
            if key in result:
                raise ValueError(f"Duplicate key found: {key}")
            else:
                result[key] = dictionary[key]
    return result

def generate_id():
    now = datetime.now()
    id = now.strftime('%Y_%m_%d_%H_%M_%S_%f')
    return id

def load_modelcache(cache_name):
    checkpoint = torch.load(cache_name)
    return checkpoint['model_state_dict']
if __name__ == "__main__":
    print(generate_id())