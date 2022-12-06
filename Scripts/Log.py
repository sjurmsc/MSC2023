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
from git import Repo
from matplotlib.pyplot import hist
import numpy as np
from numpy.linalg import norm
from matplotlib.colors import Normalize


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
    fp = 'Models/_groupstate.json'
    message = 'Modelrun {} automatic push'.format(gname)
    repo_push(fp, message)


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
        scores = json.loads(readfile.read()) # Make sure score file is not empty

    if len([score])>1:
        regression_score, reconstruction_score = score
        if np.any([regression_score < x for x in scores['regression_scores'].values()]):
            scores['regression_scores'][modelname] = regression_score

    else: reconstruction_score = score

    if np.any([reconstruction_score < x for x in scores['recon_scores'].values()]):
        scores['recon_scores'][modelname] = reconstruction_score
    else:
        return False

    # Pruning condition (Amount of etries must not exceed 10)

    with open(score_file, 'w') as writefile:
        writefile.write(json.dumps(scores, indent=2))


    return True


def replace_md_image(filepath, score):
    """
    Replaces the image in the github markdown document with the image at
    the given filepath
    """


    if len([score])>1:
        score = score[1]

    with open('README.md', 'r') as readfile:
        lines = readfile.readlines()

    # Truncates filepath
    abs_fp_idx = 0
    if 'MSC2023' in filepath:
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

    with open('.gitignore', 'a') as file:
        file.write('\n!' + trunc_filepath[3:])  # 3 at the start to skip ./
        
    fps = ['.gitignore', 'README.md', 'Models/_scores.json']
    message = 'Replacing Markdown Image'
    repo_push(fps, message)


def create_ai_error_image(e, seismic_image, image_normalize=True):
    """
    e: prediction error
    This function presumes that the depth of e and the seismic image is the same
    seismic image is presumed to be raw data
    """
    seismic_image = np.array(seismic_image)
    e = np.array(e)
    
    scaled_e = Image.fromarray(e, mode='RGBA').resize(seismic_image.shape)

    if image_normalize:
        norm = Normalize(np.min(seismic_image, axis=None), np.max(seismic_image, axis=None))
        seismic_image = Image.fromarray(norm(seismic_image), mode='RGBA')

    error_image = seismic_image.alpha_composite(scaled_e)

    return error_image


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


def create_pred_image(model, gt_data, aspect_r=1.33333, mode='sbs'):
    # Decide based on stats which section is the best predicting
    # Moving window statistics
    print('starting to create image')
    X, truth = gt_data
    truth = np.array(truth)
    pred = model.predict(X)
    if len(pred) == 2: 
        pred, pred_recon = pred
        

    if len(truth.shape) == 3:
        print('shape is 3')
        num_images, num_traces, samples = truth.shape
        width_of_image = num_images*num_traces
        truth = np.reshape(truth, (width_of_image, samples))
        X = np.reshape(X, (width_of_image, samples))
        pred = np.reshape(pred, (width_of_image, samples))
        pred_recon = np.reshape(pred_recon, (width_of_image, samples))

    elif len(truth.shape) == 2:
        num_traces, samples = truth.shape #pass # Amount of columns (to be rows)
        print('reshaping predictions')
        pred = np.reshape(pred, truth.shape)
        pred_recon = np.reshape(pred_recon, X.shape)
        print('end')
    
    traces = int(aspect_r*samples)  #the breadth of the image is the aspect_ratio*height
    
    if mode == 'sbs':
        traces //= 2

    # Decide what slice is best, by loss (l2 error norm)
    print('creating norm list')
    norm_list = norm(pred-truth, 2, axis=0)
    print(type(norm_list))
    norm_arr = np.array(norm_list)
    moving_window_mean = list(np.convolve(norm_arr, np.ones(traces), mode='valid'))

    slce = moving_window_mean.index(np.min(moving_window_mean))
    s = slice(slce, slce+traces)

    pred_matrix = pred[s] ; gt_matrix = truth[s]
    pred_recon_matrix = pred_recon[s] ; gt_recon_matrix = X[s]

    target_pred_compare = np.row_stack((pred_matrix, gt_matrix))
    recon_pred_compare = np.row_stack((pred_recon_matrix, gt_recon_matrix))

    print('X_shape: {}, y_shape: {}'.format(recon_pred_compare.shape, target_pred_compare.shape))

    return target_pred_compare.T, recon_pred_compare.T


def save_training_progression(data, model_fp):
    """Dumps the progression into npz file, so that it may be plotted with
    different rcParams in the future
    """
    filename = 'train_progress'
    data = np.array(data)
    np.savez(model_fp + '/' + filename, data)


def prediction_histogram(pred, true, **kwargs):
    pred = np.array(pred) ; true = np.array(true)
    pred = pred.flatten() ; true = true.flatten()
    return hist((pred, true), **kwargs)


def repo_push(fps, message):
    """
    Automatically pushes selected files to repo
    with an automatic commit message.
    """
    try:
        repo = Repo('.')
        for fp in [fps]:
            repo.git.add(fp)
        repo.index.commit(message)
        origin = repo.remote(name = 'origin')
        origin.push()
    except:
        print('Unable to push to remote repo')


# Only used for testing the code
if __name__ == '__main__':
    # k_obj = object()
    # k_obj._control = {'test' : [1, 2, 3]}
    # log_it(k_obj)
    # replace_md_image('Models/07-09-2022_14.12.12/coming_soon.jpg')
    print('Hello World!')
