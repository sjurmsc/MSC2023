"""
Whenever a model is trained it should have the weights saved and some descriptive information readily available
in a document, so that it is possible to know the settings used for the model.

Data used for training should be designated, and feature augmented data should be stored in /Augmented_Data

"""
from pathlib import Path
import sys
from datetime import datetime
import json
from PIL import Image

def log_it(k_obj):
    """
    Creates a log of the model being run and puts it in the \Models 
    directory. Uses the RunModel object to extract the needed values
    """
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


def replace_md_image(filepath):
    """
    Replaces the image in the github markdown document with the image at
    the given filepath
    """
    with open('README.md', 'r') as readfile:
        lines = readfile.readlines()

        # Gets first instance of markdown image
        j = [i for i, str in enumerate(lines) if str.startswith('!')][0]
        lines[j] = f'![]({filepath})\n'

        # Adds descriptive text underneath the image
        if 'Models\\' in lines[j+1]: lines[j+1] = f'{filepath}\n'
        else: lines[j] += f'{filepath}\n'

        with open('README.md', 'w') as writefile:
            writefile.writelines(lines)

from PIL import PSDraw

def compare_pred_to_gt_image(fp, im_pred, im_true, imagesize=(3508, 2480), font = 'carlito', fontsize=20):
    """
    Function creates a side by side image of the prediction versus the
    ground truth image
    """
    d = PSDraw(fp) # fp?
    d.begin_document()
    d.setfont(font, fontsize)

    # Predicted image
    d.image()
    d.text()

    # True image
    d.image()
    d.text
    
    d.end_document()


#%% Only used for testing the code

if __name__ == '__main__':
    # k_obj = object()
    # k_obj._control = {'test' : [1, 2, 3]}
    # log_it(k_obj)
    replace_md_image(r'Models\07-09-2022_14.12.12\coming_soon.jpg')
