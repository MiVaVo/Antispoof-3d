import os
import pickle
import re
from datetime import datetime as dt

import cv2
import numpy as np

from functools import wraps
from time import time

def timing(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        function_name=f.__name__
        start=dt.now()
        result = f(*args, **kwargs)
        end=dt.now()
        print(f'Elapsed time of {function_name:>30}  {(end-start).total_seconds():>20.1}')
        return result
    return wrapper



@timing
def classify(classifier, distances):
    if isinstance(classifier, list):
        # prob_of_fake={clf.__class__.__name__:clf.predict_proba(get_statistics(distances[np.newaxis,:]))[0,0] for clf in classifier}
        prob_of_fake = {clf.__class__.__name__: clf.predict_proba(distances[np.newaxis, :])[0, 0] for clf in
                        classifier}

    else:
        prob_of_fake = classifier.predict_proba(distances[np.newaxis, :])[0, 0]
    return prob_of_fake


def get_image_depth(path):
    with open(path, 'rb') as f:
        data_new = pickle.load(f)
    if 'color' in data_new.keys():
        return cv2.cvtColor(data_new['image'], cv2.COLOR_RGB2BGR), data_new['depth']
    else:
        return data_new['image'], data_new['depth']


def absoluteFilePaths(directory, ext=".pk"):
    for dirpath, _, filenames in os.walk(directory):
        for f in filenames:
            if ext in f:
                yield os.path.abspath(os.path.join(dirpath, f))


def save_ds(path_to_folder, img, depth):
    json_save = {"image": img, "depth": depth}
    save_file = f'{os.path.join(path_to_folder, re.sub("[^0-9]", "", str(dt.now())))}'
    with open(f"{save_file}.pk", "wb") as file:
        pickle.dump(json_save, file, protocol=pickle.HIGHEST_PROTOCOL)
    cv2.imwrite(f'{save_file}.jpg', img)
    return 0
