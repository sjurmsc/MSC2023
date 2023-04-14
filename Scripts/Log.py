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
from matplotlib.colors import Normalize, Colormap, ListedColormap
import matplotlib.pyplot as plt
from openpyxl import load_workbook
from pandas import read_csv, DataFrame


msc_color = '#6bb4edff'

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


def create_ai_error_image(e, seismic_array, image_normalize=True, filename = False):
    """
    e: prediction error
    This function presumes that the depth of e and the seismic image is the same
    seismic image is presumed to be raw data
    """
    seismic_array = np.array(seismic_array).T
    e = np.array(e, dtype=float).T

    alpha_norm = Normalize(np.min(e, axis=None), np.max(e, axis=None))
    
    norm = Normalize(np.min(seismic_array, axis=None), np.max(seismic_array, axis=None))
    seismic_alpha = np.ones_like(seismic_array)-norm(seismic_array)

    if image_normalize:
        seismic_image = Image.fromarray((plt.cm.gray(norm(seismic_array), alpha=seismic_alpha)*255).astype(np.uint8), mode='RGBA')
    else:
        seismic_image = Image.fromarray((plt.cm.gray(np.array(seismic_array), alpha=seismic_alpha)*255).astype(np.uint8), mode='RGBA')

    cmap = lambda x: plt.cm.Reds(alpha_norm(x), alpha=(np.ones_like(x)-alpha_norm(x)))*255
    scaled_e = Image.fromarray(cmap(e).astype(np.uint8), mode='RGBA').resize(seismic_image.size)

    error_image = Image.new('RGBA', scaled_e.size)
    error_image = Image.alpha_composite(error_image, seismic_image)
    error_image = Image.alpha_composite(error_image, scaled_e)

    if filename:
        error_image.save(filename)
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
    X, truth = gt_data
    truth = np.array(truth)
    pred = model.predict(X)
    if len(pred) == 2: 
        pred, pred_recon = pred
        

    if len(truth.shape) == 3:
        num_images, num_traces, samples = truth.shape
        width_of_image = num_images*num_traces
        truth = np.reshape(truth, (width_of_image, samples))
        X = np.reshape(X, (width_of_image, samples))
        pred = np.reshape(pred, (width_of_image, samples))
        pred_recon = np.reshape(pred_recon, (width_of_image, samples))

    elif len(truth.shape) == 2:
        num_traces, samples = truth.shape #pass # Amount of columns (to be rows)
        pred = np.reshape(pred, truth.shape)
        pred_recon = np.reshape(pred_recon, X.shape)
    
    traces = int(aspect_r*samples)  #the breadth of the image is the aspect_ratio*height
    
    if mode == 'sbs':
        traces //= 2

    # Decide what slice is best, by loss (l2 error norm)
    target_pred_diff = pred-truth
    norm_list = norm(pred-truth, 2, axis=0)
    norm_arr = np.array(norm_list)
    moving_window_mean = list(np.convolve(norm_arr, np.ones(traces), mode='valid'))

    slce = moving_window_mean.index(np.min(moving_window_mean))
    s = slice(slce, slce+traces)

    pred_matrix = pred[s] ; gt_matrix = truth[s]
    pred_recon_matrix = pred_recon[s] ; gt_recon_matrix = X[s]

    target_pred_compare = np.row_stack((pred_matrix, gt_matrix))
    recon_pred_compare = np.row_stack((pred_recon_matrix, gt_recon_matrix))

    return target_pred_compare.T, recon_pred_compare.T, target_pred_diff


def save_training_progression(data, model_fp):
    """Dumps the progression into npz file, so that it may be plotted with
    different rcParams in the future
    """
    filename = 'train_progress'
    data = np.array(data)
    np.savez(model_fp + '/' + filename, data)


def save_config(model_loc, config):
    with open(model_loc + '/' + 'config.json', 'w') as w_file:
        dummy_config = config.copy()
        if not isinstance(config['activation'], str):
            dummy_config['activation'] = str(config['activation'].name)
        dummy_config['convolusion_func'] = str(config['convolution_func'].__name__)
        w_file.write(json.dumps(dummy_config, indent=2))

def prediction_histogram(pred, true, **kwargs):
    pred = np.array(pred) ; true = np.array(true)
    pred = pred.flatten() ; true = true.flatten()
    return hist((pred, true), **kwargs)


def prediction_crossplot(pred, 
                         true, 
                         title='',
                         xlabel='Estimated property',
                         ylabel='Ground Truth',
                         save=False):
    plt.scatter(pred, true)
    plt.grid()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if save:
        plt.savefig('prediction_crossplot')


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

def plot_history(History, filename=None):
    """Plots the history of the model training"""
    for key in History.history.keys():
        if 'loss' in key:
            if key == 'loss':
                plt.plot(History.history[key], label=key, color='k', linewidth=2, zorder=2)
            else:
                plt.plot(History.history[key], label=key, linewidth=2, zorder=1)
    plt.yscale('log')
    plt.title('Model loss', fontsize=20)
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()

def plot_histories(Histories, val=False, filename=None):
    """Plots the history of the model training"""
    for i, History in enumerate(Histories):
        plt.plot(History.history['loss'], label='Train', color='red', linewidth=2)
        # if val:
        #     plt.plot(History.history['val_loss'], label=f'Validation {i}', color='k', linewidth=2, linestyle='--')
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.savefig(filename)
    plt.close()


def create_results_xl():
    """Creates a new excel file with the results of the model"""
    xl = "C:/Users/SjB/MSC2023/Results/NGI_stdd_{}.xlsx"
    wb = load_workbook(xl)
    # Open the worksheet q_c
    ws = wb['q_c']

    wb.save('results.xlsx')

# import boundary norm
from matplotlib.colors import BoundaryNorm

def get_GGM_cmap(GGM):
    """Creates a segmented colorbar with units colored by their uid"""

    umap = read_csv('../OneDrive - NGI/Documents/NTNU/MSC_DATA/StructuralModel_unit_mapping.csv')
    GGM_names = []

    unique_uid = np.sort(np.unique(GGM.flatten())).astype(int)

    for u in unique_uid:
        GGM_names.append(umap.loc[umap['uid'] == u]['unit'].values[0])

    n_colors = len(np.unique(umap['uid']))
    # Add a segmented colorbar with unique colors for the different units
    cmap = plt.cm.get_cmap('gnuplot', n_colors)

    # Define the bins and normalize
    bounds = np.linspace(0, n_colors, n_colors+1)
    norm = BoundaryNorm(bounds, cmap.N)

    # Create a colorbar with only the unique GGM provided to the function given the right colors
    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ticks=np.arange(len(unique_uid))+0.5)
    # cbar.ax.set_yticklabels(GGM_names)
    cbar = None
    return cmap, norm, cbar


def describe_data(X, y, groups, GGM, mdir=''):
    """Creates a text file with the description of the data"""

    unit_mapping = read_csv('../OneDrive - NGI/Documents/NTNU/MSC_DATA/StructuralModel_unit_mapping.csv', index_col='uid')

    filename = mdir + 'dataset_description.csv'
    latex_filename = mdir + 'dataset_description.tex'

    tex_string = ''
    data_df = DataFrame(columns=['qc', 'fs', 'u2', 'GGM'])

    # Make histogram of the GGM counts with unit mapping on the x axis centered at the bars
    fig, ax = plt.subplots()

    umap = lambda x: unit_mapping['unit'][int(x)]
    
    # Make a flattened sorted array of the GGM values
    ggm_flatsort = np.sort(GGM.flatten())
    string_GGM = np.vectorize(umap)(ggm_flatsort)
    ax.hist(string_GGM, bins=np.arange(len(np.unique(GGM))+1), color=msc_color, edgecolor='k')
    ax.set_xticks(np.arange(len(np.unique(GGM))) + 0.5)
    ax.set_xticklabels(np.unique(string_GGM, return_counts=True)[0])
    # Rotate the xtick labels
    plt.setp(ax.get_xticklabels(), rotation=90, rotation_mode="anchor", ha="right", va="center")
    ax.set_xlabel('Unit')
    ax.set_ylabel('Count')
    fig.subplots_adjust(bottom=0.2)
    fig.suptitle('CPT values per GGM unit')
    fig.savefig(mdir + 'GGM_histogram.png', dpi=500)
    plt.close(fig)

    
    ggm = GGM.reshape(*y.shape[:-1], 1)
    flat_data = np.concatenate((y, ggm), axis=-1).reshape(-1, 4)

    df = DataFrame(flat_data, columns=['$q_c$', '$f_s$', '$u_2$', 'GGM'])
    df.to_csv(mdir+'data.csv')

    for g in df.groupby('GGM'):
        unit = umap(g[0])

        g_data = g[1].iloc[:, :-1]
        
        desc = g_data.describe().T
        desc.columns = ['n', '$\mu$', '$\sigma$', 'min', 'Q1', 'Q2', 'Q3', 'max']

        beg = '\\begin{table}[h]\n'
        string = beg + f'\\caption{{Summary statistics for {unit}}}\n'
        string += desc.to_latex(escape=False)
        string += '\\end{table}\n\n'
        tex_string += string

        data_df = data_df.append(g[1].describe())

    data_df.to_csv(filename)
    with open(latex_filename, 'w') as f:
        f.write(tex_string)

    
def get_umap_func():

    unit_mapping = read_csv('../OneDrive - NGI/Documents/NTNU/MSC_DATA/StructuralModel_unit_mapping.csv', index_col='uid')

    def umap(uid):
        return unit_mapping['unit'][int(uid)]

    return umap
    
from NGI.GM_Toolbox import evaluate_modeldist

def make_cv_excel(filename, COMP_DF):
    """Creates a new excel file with the results of the model"""
    xl = "Results/NGI_stdd_{}.xlsx"
    wb = load_workbook(xl)
    # Open the worksheet q_c
    
    params = ['q_c', 'f_s', 'u_2']
    
    unit_mapping = read_csv('../OneDrive - NGI/Documents/NTNU/MSC_DATA/StructuralModel_unit_mapping.csv', index_col='uid')
    
    for g in COMP_DF.groupby('GGM'):
        unit = unit_mapping['unit'][int(g[0])]
        g_data = g[1]

        for p in params:
            ws = wb[p]
            param = p.replace('_', '')
            for method in ['CNN', 'RF', 'LGBM']:
                true = g_data['True_{}'.format(param)]
                pred = g_data['{}_{}'.format(method, param)]
                std = evaluate_modeldist(true, pred)[4]
                
                # Find the cell row by searching for the GGM in the first column
                for row in range(2, ws.max_row + 1):
                    if ws.cell(row=row, column=1).value == unit:
                        cellrow = row
                        break
                # Find the cell column by searching for the method in the first row
                for col in range(2, ws.max_column + 1):
                    if '_'+method in ws.cell(row=1, column=col).value:
                        cellcol = col
                        break
                
                # set the cell value to the std
                ws.cell(row=cellrow, column=cellcol).value = std

    if not filename.endswith('.xlsx'):
        filename += '.xlsx'
    wb.save(filename)


def add_identity(axes, *line_args, **line_kwargs):
    """Author: JaminSore (https://stackoverflow.com/users/1940479/jaminsore)"""
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes


if __name__ == '__main__':
    pass
    # plt.imshow(np.array([1]))
    # a = np.random.randint(1, 10, size=(20, 10))
    # b = np.random.random((20, 10))

    # e = create_ai_error_image(b, a, filename = 'test.png')
    # print(e)
