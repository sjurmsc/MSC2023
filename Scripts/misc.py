
def find_all_relevant_soil_units():
    import segyio
    import numpy as np

    filename = r'P:\2019\07\20190798\Deliverables\Digital_Deliverables\02_IGM\00-StructuralModel\StructuralModel.sgy'
    with segyio.open(filename, ignore_geometry=True) as segyfile:
        segyfile.mmap()
        z = segyfile.samples
        traces = segyio.collect(segyfile.trace)
    
    # Find all unique values in the seismic
    unique_values = np.unique(traces.T[np.where(z<100)])
    print('The unique soil units in the upper 100m are: ', unique_values)
    print(traces.shape)

if __name__ == '__main__':
    find_all_relevant_soil_units()