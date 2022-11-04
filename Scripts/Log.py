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
import numpy as np


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

def update_groupstate(gname):
    """Keeps groupstate updated after any model run
    """
    from git import Repo
    try:
        repo = Repo('.')
        repo.git.add('Models/_groupstate.json')
        repo.index.commit('Modelrun {} automatic push'.format(gname))
        origin = repo.remote(name = 'origin')
        origin.push()
    except:
        print('Unable to push to remote repo')

def new_group(push=True):
    """
    Creates a log of the model being run and puts it in the \Models 
    directory. Uses the RunModel object to extract the needed values
    """    
    with open('./Models/_groupstate.json', 'r+') as state:
        group = json.loads(state.read())
        state.seek(0)
        group['Group'] = gname(group['Group'])
        group_name = group['Group']

        state.write(json.dumps(group))
    

    # Creates parent directory
    m = Path('./Models')
    new = m.joinpath(group_name)
    Path.mkdir(new, parents=True, exist_ok=True)
    
    if push: update_groupstate(group['Group'])
    return group_name

def nats(k):
    yield k
    yield from nats(k+1)

def give_modelname():
    n = nats(0)
    groupname = new_group()
    while True: # Always
        yield groupname, next(n)


def update_scores(modelname, score):
    score_file = 'Models/_scores.json'
    
    with open(score_file, 'r') as readfile:
        scores = json.loads(readfile.read())

    if len(score)>1:
        regression_score, reconstruction_score = score
        if np.any([regression_score < x for x in scores['regression_scores'].values()]):
            scores['regression_scores'][modelname] = regression_score

    else: reconstruciton_score = score

    if np.any([reconstruction_score < x for x in scores['recon_scores'].values()]):
        scores['recon_scores'][modelname] = reconstruction_score
    else:
        return False

    # Pruning condition (Amount of etries must not exceed 10)

    with open(score_file, 'w') as writefile:
        writefile.write(json.dumps(scores, inline=2))
    return True


def replace_md_image(filepath, score):
    """
    Replaces the image in the github markdown document with the image at
    the given filepath
    """
    if len(list(score))>1:
        score = score[1]

    with open('README.md', 'r') as readfile:
        lines = readfile.readlines()

        # Truncates filepath
        abs_fp_idx = filepath.find('MSC2023') + len('MSC2023')
        trunc_filepath = '.' + filepath[abs_fp_idx:]

        

        # Gets first instance of markdown image
        j = [i for i, str in enumerate(lines) if str.startswith('!')][0]

        # Gets old score
        score_loc = j-1
        score_line = lines[score_loc]
        old_score_idx = score_line.find('score ') + len('score ')
        old_score = float(score_line[old_score_idx:score_line.find(':')])

        if score > old_score: return # The image only gets replaced if the score is better
        
        new_score_line = score_line[:old_score_idx] + str(score) + ':\n'
        lines[score_loc] = new_score_line

        # Adds the new image
        lines[j] = f'![]({trunc_filepath})\n'

        # Adds descriptive text underneath the image
        if '.jpg' in lines[j+1]: lines[j+1] = f'{trunc_filepath}\n'
        else: lines[j] += f'*{trunc_filepath}*\n'

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
    l_box = None #pass
    d.image(l_box, im_pred, dpi=dpi)
    tl_loc = (l_box[0], l_box[1]-raise_text) # Text raised by 20 px
    d.text(tl_loc, 'Predicted image')

    # True image
    r_box = None #pass
    d.image(r_box, im_true, dpi=dpi)
    tr_loc = (r_box[0], r_box[1]-raise_text)
    d.text(tr_loc, 'Ground Truth')
    
    d.end_document()

import numpy as np
from numpy.linalg import norm

def create_pred_image_from_1d(model, gt_data, aspect_r=1.33333, mode='sbs'):
    # Decide based on stats which section is the best predicting
    # Moving window statistics

    X = gt_data
    if len(X) == 2:
        truth, X = X
    elif len(X) == 1:
        truth = X
    truth = np.array(truth)
    samples = truth.shape[1] #pass # Amount of columns (to be rows)
    pred = model.predict(X)
    if len(pred) == 2: pred, pred_recon = pred
    
    traces = int(aspect_r*samples)  #the breadth of the image is the aspect_ratio*height
    
    if mode == 'sbs':
        traces //= 2

    #truth = truth.reshape(truth.shape[:-1])
    
    # pred = np.array([])
    # for i in range(X.shape[0]):
    #     pred = np.row_stack(pred, model.predict(gt_data[i, :]))
     # 
    
    scr = []

    # Decide what slice is best, by loss (l2 error norm)
    for s_idx in range(len(truth)-traces):
        s = slice(s_idx, s_idx+traces)
        score = norm(pred[s]-truth[s], 2)
        scr.append(score)

    slce = scr.index(np.min(scr))
    s = slice(slce, slce+traces)

    pred_matrix = pred[s] ; gt_matrix = truth[s]

    p = np.row_stack((pred_matrix, gt_matrix))
    return p.T, (pred_matrix.T, gt_matrix.T) # quickfix

def save_training_progression(data, model_fp):
    """Dumps the progression into npz file, so that it may be plotted with
    different rcParams in the future
    """
    filename = 'train_progress'
    data = np.array(data)
    data.savez(model_fp + '/' + filename)



#%% Only used for testing the code

if __name__ == '__main__':
    # k_obj = object()
    # k_obj._control = {'test' : [1, 2, 3]}
    # log_it(k_obj)
    # replace_md_image('Models/07-09-2022_14.12.12/coming_soon.jpg')
    print('Hello World!')
