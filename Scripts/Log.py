"""
Whenever a model is trained it should have the weights saved and some descriptive information readily available
in a document, so that it is possible to know the settings used for the model.

Data used for training should be designated, and feature augmented data should be stored in /Augmented_Data

"""
from pathlib import Path
import sys
from datetime import datetime

def log_it():
    p = Path('./Models')
    return p


if __name__ == '__main__':
    m = log_it()
    n = datetime.now()
    new = m.joinpath(str(n.strftime('%d-%m-%Y_%H:%M:%S')))
    new.mkdir()
