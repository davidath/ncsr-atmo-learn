#!/usr/bin/env python
import numpy as np
import sys
sys.path.append('..')
sys.path.append('/mnt/disk1/thanasis/bde-pilot-2/backend')
from netcdf_subset import netCDF_subset
from netCDF4 import Dataset
import dataset_utils as utils
from disputil import displayz
import lasagne.layers as lasagne
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import scipy.misc
import matplotlib.gridspec as gridspec

# idx: 500

# based on http://ufldl.stanford.edu/wiki/index.php/Visualizing_a_Trained_Autoencoder
def visualize_W(nnmodel,num):
    # print lasagne.get_all_params(nnmodel._network)
    # print len(lasagne.get_all_params(nnmodel._network))
    W = lasagne.get_all_param_values(nnmodel._network)[num]
    sumW = np.sum(W)
    return (W / np.sqrt(sumW ** 2))

def map_contour(data, title=None):
    inp = '/mnt/disk1/thanasis/data/11_train.nc'
    nc_fid = Dataset(inp,'r')
    lats = nc_fid.variables['CLAT'][0,0:64]
    lats = lats[:,0]
    lons = nc_fid.variables['XLONG_M'][0,0:64]
    lons = lons[10,:]

    m = Basemap(width=3900000,height=3500000,
     resolution='l',projection='lcc', lat_0 = 50, lon_0 = 16, lat_ts = 40, k_0=1.035)
    m.drawcoastlines()
    m.drawcountries()
    m.drawmapboundary()
    lons, lats = np.meshgrid(lons, lats)
    x, y = m(lons, lats)
    cs = m.contour(x, y, data, linewidths=1.5, colors = 'k')
    # pcl = m.pcolor(x,y,np.squeeze(cfilter))
    # cbar = m.colorbar(pcl, location='bottom', pad="10%")
    # cbar.set_label("$m^2$/$s^2$")
    if title:
        plt.title(title)
    nc_fid.close()

def visualize_conv(nnmodel):
    train = np.load('/mnt/disk1/thanasis/NIPS/GHT_ALL_vanilla.npy')
    import theano
    nnmodel._input_layer.input_var = theano.shared(name='input_var',
                                  value=np.asarray(train,
                                  dtype=theano.config.floatX),
                                  borrow=True)
    lconv = lasagne.get_all_layers(nnmodel._network)[2]
    filt = lasagne.get_output(lconv).eval()
    filt = filt[500,:]
    fig = plt.figure()
    fig.add_subplot(1,3,1)
    map_contour(train[500,0:4096].reshape(64,64),'500hPa Geopotential Height')
    fig.add_subplot(1,3,2)
    map_contour(train[500,4096:8192].reshape(64,64),'700hPa Geopotential Height')
    fig.add_subplot(1,3,3)
    map_contour(train[500,8192:12288].reshape(64,64),'900hPa Geopotential Height')
    for i in range(30):
        print i
        fig.add_subplot(1,6,i+1)
        cfilter = filt[i,:]
        cfilter = scipy.misc.imresize(cfilter,(64,64))
        map_contour(cfilter, 'Conv Filter No: '+str(i+1))
        if i == 5:
            break
    plt.show()


def main():
    nnmodel = utils.load_single(sys.argv[1])
    conv = visualize_conv(nnmodel)
    # lfeat = visualize_W(nnmodel,0)
    # print lfeat.shape
    # lfeat = lfeat[:,range(0,10)]
    # lfeat = lfeat.T
    # lfeat = lfeat.reshape(10,28,28)
    # print lfeat.shape
    # displayz(lfeat,2,5,0,28,28)


if __name__ == '__main__':
    main()
