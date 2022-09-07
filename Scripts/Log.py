"""
Whenever a model is trained it should have the weights saved and some descriptive information readily available
in a document, so that it is possible to know the settings used for the model.

Data used for training should be designated, and feature augmented data should be stored in /Augmented_Data

"""
from pathlib import Path
import sys
from datetime import datetime
import json


def log_it(k_obj):

    # Creates parent directory
    m = Path('./Models')
    n = datetime.now()
    new = m.joinpath(str(n.strftime('%d-%m-%Y_%H.%M.%S\\')))
    Path.mkdir(new, parents=True, exist_ok=True)

    # Logging the JSON control file
    c_fname = new.joinpath('control.json').resolve()
    c_file = open(c_fname, 'w')
    ctrl = json.dumps(k_obj._control)
    c_file.write(ctrl)

    # Logging the ML weights
    wdir = new / 'ML weights'
    wdir.mkdir()

# Only used for testing the code
class object:
    def __init__(self):
        pass

if __name__ == '__main__':
    k_obj = object()
    k_obj._control = {'test' : [1, 2, 3]}
    log_it(k_obj)
