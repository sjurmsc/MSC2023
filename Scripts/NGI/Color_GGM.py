import numpy as np

colmax=np.array([[142,70,149], [0,152,129], [100, 100,100], [240,64,40], 
                 [34,181,233], [103,143,102], [202,0,197], [255,230,25]])/256

colmin=np.array([[229,203,228], [182,225,194], [200,200,200], [250,167,148], 
                 [179,227,238], [191,194,107], [230,185,205], [255,255,174]])/256

def color_range(N, cmax, cmin, unitcount): 
    vals = np.ones((N, 4))
    vals[:, 0] = np.linspace(cmax[0], cmin[0], N)
    vals[:, 1] = np.linspace(cmax[1], cmin[1], N)
    vals[:, 2] = np.linspace(cmax[2], cmin[2], N)
    vals[:, 0:3]=vals[:, 0:3]
    vals=np.round(vals, 3)
    return vals[unitcount, :]