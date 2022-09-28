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

def gname(old_name):
    """ Takes in the state of group names and outputs the next one
    """
    if not len(old_name): raise TypeError('Max group name encountered')
    new_name = old_name
    if old_name[-1] != 'Z':
        new_name = old_name[:-1] + chr(ord(old_name[-1])+1)
    else:
        new_name = gname(old_name[:-1]) + 'A'
    return new_name

def new_group():
    """
    Creates a log of the model being run and puts it in the \Models 
    directory. Uses the RunModel object to extract the needed values
    """
    from string import ascii_uppercase

    
    with open('./Models/_groupstate.json', 'r+') as state:
        group = json.loads(state.read())
        group['Group'] = gname(group['Group'])
        group_name = group['Group']

        state.write(json.dumps(group))

    print(group_name)
    # Creates parent directory
    m = Path('./Models')
    n = datetime.now()
    new = m.joinpath(group_name)
    Path.mkdir(new, parents=True, exist_ok=True)
    return group_name


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
        if 'Models/' in lines[j+1]: lines[j+1] = f'{filepath}\n'
        else: lines[j] += f'{filepath}\n'

        with open('README.md', 'w') as writefile:
            writefile.writelines(lines)



def compare_pred_to_gt_image(fp, im_pred, im_true, imagesize=(3508, 2480), font = 'carlito', fontsize=20, dpi=300):
    """
    Function creates a side by side image of the prediction versus the
    ground truth image
    """
    from PIL import PSDraw
    
    d = PSDraw.PSDraw('TEMP/test') # fp?
    d.begin_document()
    d.setfont(font, fontsize)
    raise_text = 20

    # Predicted image
    l_box = _ #pass
    d.image(l_box, im_pred, dpi=dpi)
    tl_loc = (l_box[0], l_box[1]-raise_text) # Text raised by 20 px
    d.text(tl_loc, 'Predicted image')

    # True image
    r_box = _ #pass
    d.image(r_box, im_true, dpi=dpi)
    tr_loc = (r_box[0], r_box[1]-raise_text)
    d.text(tr_loc, 'Ground Truth')
    
    d.end_document()

import numpy as np
from numpy.linalg import norm

def create_pred_image_from_1d(model, X, gt_data, ratio=1.33333):
    # Decide based on stats which section is the best predicting
    # Moving window statistics
    samples = _ #pass # Amount of rows
    traces = int(ratio*samples)  #the breadth of the image is the aspect_ratio*height

    pred = model.predict(X=X) # 

    scr = []

    # Decide what slice is best, by loss (l2 error norm)
    for s_idx in range(len(gt_data)-traces):
        s = slice(s_idx, s_idx+traces)
        score = norm(pred[s]-gt_data[s], 2)
        scr.append(score)

    slce = np.index(np.min(scr))
    s = slice(slce, slce+traces)

    pred_matrix = pred[s] ; gt_matrix = gt_data[s]

    return pred_matrix, gt_matrix




#%% Only used for testing the code

if __name__ == '__main__':
    # k_obj = object()
    # k_obj._control = {'test' : [1, 2, 3]}
    # log_it(k_obj)
    replace_md_image(r'Models/07-09-2022_14.12.12/coming_soon.jpg')
