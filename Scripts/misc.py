import matplotlib.pyplot as plt

def find_all_relevant_soil_units():
    import segyio
    import numpy as np

    filename = '../OneDrive - NGI/Documents/NTNU/MSC_DATA/StructuralModel.sgy'
    with segyio.open(filename, ignore_geometry=True) as segyfile:
        z = segyfile.samples
        traces = segyio.collect(segyfile.trace)
        cdps = segyio.collect(segyfile.attributes(segyio.TraceField.CDP))
        lines = segyio.collect(segyfile.attributes(segyio.TraceField.INLINE_3D))
    
    print('Done reading the seismic')

    # Find all unique values in the seismic
    unique_values = np.unique(traces.T[np.where(z<100)])
    print('The unique soil units in the upper 100m are: ', unique_values)
    print(traces.shape)

    print(np.unique(lines))
    plt.plot(cdps)
    plt.show()

if __name__ == '__main__':
    find_all_relevant_soil_units()