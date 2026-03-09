import random
import json
import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
import torch
import collections
import os


def get_base_dir():
    root_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(root_dir)
    return root_dir


def get_description(data_name, desctiption_types=['llama'], description_length=-1):
    valid_types = get_valid_types(data_name)
    valid_descriptions = {key:[] for key in valid_types}
    data_dir = os.path.join(get_base_dir(), "data")

    for description_type in desctiption_types:
        print(description_type)
        if description_type == 'llama':
            with open(os.path.join(data_dir, 'description_llama.json'), 'r') as f:
                description = json.load(f)
        elif description_type == 'gpt':
            with open(os.path.join(data_dir, 'description_gpt.json'), 'r') as f:
                description = json.load(f)
        if description_length != -1: # get the most frequent ones
            for key, value in description.items():
                new_length = int(description_length * len(value))
                counter = collections.Counter(value)
                frequents = counter.most_common(new_length)
                frequent_values = [item for item, count in frequents]
                description[key] = frequent_values


        for key in description:
            if key in valid_descriptions:
                valid_descriptions[key].extend(list(map(str, description[key])))

    return valid_descriptions

def get_valid_types(data_name):
    with open(os.path.join(get_base_dir(), 'data', 'types.json'), 'r') as f:
        valid_types = json.load(f)

    if data_name in ['msato0', 'msato1', 'msato2', 'msato3', 'msato4', 'sato0', 'sato1', 'sato2', 'sato3', 'sato4']:
        return valid_types['type78']
    elif data_name in['sota-schema', 'sota-schema-small', 'dataset-sota', 'dataset-sota-small', 'dataset-sota-corner']:
        return valid_types['sota-schema']
    elif data_name =='sota-dbpedia' or data_name =='dataset-sota-dbpedia':
        return valid_types['sota-dbpedia']
    elif data_name == 'dataset-t2d':
        return valid_types['t2d']
    elif data_name == 'dataset-limaye':
        return valid_types['limaye']
    elif data_name == 'dataset-wikipedia':
        return valid_types['wikipedia']
    elif data_name in ['dataset-turl', 'turl']:
        return valid_types['type255']
    return valid_types[data_name]


def set_seed(seed: int):
    """https://github.com/huggingface/transformers/blob/master/src/transformers/trainer.py#L58-L63"""    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    """Add the following 2 lines
    https://discuss.pytorch.org/t/how-could-i-fix-the-random-seed-absolutely/45515
    """
    torch.backends.cudnn.enabled=False
    torch.backends.cudnn.deterministic=True

    """
    For detailed discussion on the reproduciability on multiple GPU
    https://discuss.pytorch.org/t/reproducibility-over-multigpus-is-impossible-until-randomness-of-threads-is-controled-and-yet/47079
    """
