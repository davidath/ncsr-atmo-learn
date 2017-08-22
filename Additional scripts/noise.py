import sys
sys.path.append('..')

from operator import attrgetter
from argparse import ArgumentParser
import numpy as np
from netcdf_subset import netCDF_subset
from Dataset_transformations import Dataset_transformations

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-n', '--num',required=True, type=int,
                        help='number of slices')
    parser.add_argument('-o', '--output',required=True, type=str,
                        help='export path/name')
    opts = parser.parse_args()
    getter = attrgetter('input','num','output')
    inp,num,output = getter(opts)
    # data_dict = netCDF_subset(
    #  inp, [500,700,900], ['UU','VV','GHT'], lvlname='num_metgrid_levels', timename='Times')
    # items = [data_dict.extract_piece(range(0,16072),range(0,64),range(0,64))]
    # items = np.array(items)
    # ds = Dataset_transformations(items, 1000, items.shape)
    # print ds._items.shape
    # np.savez_compressed('UVGEO_3lvl',ds._items)
    dataset = np.load(inp)
    data_x = dataset.shape[0]
    data_y = dataset.shape[1]
    noisy = []
    for i in range(0,data_x):
        noisy.append(dataset[i,:])
        for j in range(0,num):
             rng = np.random.RandomState().uniform(-0.1, 0.1, size=data_y)
             noise = np.add(dataset[i,:], np.multiply(dataset[i,:], rng))
             noisy.append(noise)
    np.save(output,noisy)
