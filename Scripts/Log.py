"""
Whenever a model is trained it should have the weights saved and some descriptive information readily available
in a document, so that it is possible to know the settings used for the model.
"""
from pathlib import Path
import sys
from datetime import datetime

def log_it():
    p = Path('./Models')
    print(p.is_dir())
    return p



if __name__ == '__main__':
    m = log_it()
    n = datetime.now()
    new = m.joinpath(str(n.strftime(r'%d-%m-%Y %H:%M:%S')))
    print(new)
    new.mkdir()
