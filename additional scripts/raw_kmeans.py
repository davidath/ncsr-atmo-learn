import sys
sys.path.append('..')

import os
os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'

from netcdf_subset import netCDF_subset
from operator import attrgetter
from argparse import ArgumentParser
from Dataset_transformations import Dataset_transformations
from Clustering import Clustering
from theano import tensor as T
import dataset_utils as utils
import numpy as np
import datetime

PREFIX = "GHT700_CONV"
MODEL_PATH = "/mnt/disk1/thanasis/NIPS/models/conv_ght_700/CONV_GHT_700_model_cpu.zip"
NC_PATH = '/mnt/disk1/thanasis/data/11_train.nc'

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='input file')
    parser.add_argument('-o', '--output', type=str,
                        help='output file')
    opts = parser.parse_args()
    getter = attrgetter('input', 'output')
    inp, outp = getter(opts)
    export_template = netCDF_subset(
        NC_PATH, [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
    ds = Dataset_transformations(np.load(inp), 1000, np.load(inp).shape)
    print ds._items.shape
    times = export_template.get_times()
    nvarin = []
    for var in export_template.get_times():
        str = ""
        for v in var:
            str += v
        nvarin.append(str)
    times = []
    for var in nvarin:
        under_split = var.split('_')
        date_split = under_split[0].split('-')
        time_split = under_split[1].split(':')
        date_object = datetime.datetime(int(date_split[0]), int(date_split[1]), int(
            date_split[2]), int(time_split[0]), int(time_split[1]))
        times.append(date_object)
    print times[0:10]
    if MODEL_PATH != "":
        print 'Loading model.....'
        m = utils.load_single(MODEL_PATH)
        ds._items = m.get_hidden(ds._items)
    print ds._items.shape
    clust_obj = Clustering(ds,n_clusters=15,n_init=100,features_first=False)
    clust_obj.kmeans()
    clust_obj.create_density_descriptors(12,times)
    export_template = netCDF_subset(
        NC_PATH, [700], ['GHT'], lvlname='num_metgrid_levels', timename='Times')
    clust_obj.mult_desc_date(export_template)
    utils.export_descriptor_mult_dense(outp,export_template,clust_obj)
    clust_obj.save(PREFIX+'_mult_dense.zip')
