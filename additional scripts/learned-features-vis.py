#!/usr/bin/env python
import os
# os.environ['THEANO_FLAGS'] = '
import numpy as np
import sys
sys.path.append('..')
import theano
import theano.tensor as T
import dataset_utils as utils
import matplotlib.pyplot as plt
import lasagne.layers as layers
import lasagne
from datetime import datetime
from disputil import displayz
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import scipy.misc
from sklearn.preprocessing import minmax_scale,scale

def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

PATH=''

def AM_dense(nnmodel):
    X1 = np.load(PATH)
    x = T.matrix('x')
    nnmodel._input_layer.input_var = x
    activation = layers.get_output(nnmodel._encoder_layer).mean()
    # activation = layers.get_output(layers.get_all_layers(nnmodel._network)[1]).mean()
    grad = T.grad(activation, x)
    grad /= T.sqrt(T.mean(T.square(X1)))
    gasc = theano.function(inputs=[x], outputs=[activation, grad])

    # _X = np.random.random(size=(36,784)) * 20 + 128
    _X = X1[0:36, :]

    base_lr = 0.1
    i = 0
    pre_loss = 0
    while True:
        try:
            loss, grad_value = gasc(_X)
            if i % 10 == 0:
                log(str(i) + ' ' + str(loss), label='AM')
            _X += grad_value * base_lr
            i += 1
            pre_loss = loss
        except KeyboardInterrupt:
            break
    displayz(grad_value.reshape(36, 28, 28), 6, 6, 0, 28, 28, binary=True)


def AM_conv(nnmodel, conv, idx):
    X1 = np.load(PATH)
    lconv = layers.get_all_layers(nnmodel._network)[conv]
    x = T.matrix('x')
    nnmodel._input_layer.input_var = x
    activation = layers.get_output(lconv)[:, idx, :, :].mean()
    filter_value = layers.get_output(lconv)[:, idx, :, :]
    grad = T.grad(activation, x)
    grad /= T.sqrt(T.mean(T.square(grad)))
    gasc = theano.function(inputs=[x], outputs=[
                           activation, grad, filter_value])
    # _X = np.random.random((1, 12288)) * 20 + 128
    _X = np.random.uniform(low=-1,high=4,size=(1,12288))
    # _X = X1[500,:].reshape(1,12288)
    base_lr = 0.1
    i = 0
    while True:
        try:
            loss, grad_value,filter_value = gasc(_X)
            if i % 10 == 0:
                log(str(i) + ' ' + str(loss), label='AM')
            if loss > 1000:
                break
            if loss == 0.0:
                _X = np.random.uniform(low=-10,high=10,size=(1,12288))
            else:
                _X += grad_value * base_lr
            i += 1
        except KeyboardInterrupt:
            break
    _X = _X.reshape(3,64,64)
    _X = _X.transpose((1,2,0))
    plt.imshow(_X)
    return _X


def linear_comb(net1, net2, x, idx=None):
    w0 = net1.W
    w1 = net2.W
    diag_part1 = net1.nonlinearity(T.dot(x, w0))
    diag_part2 = 1 - diag_part1
    mult = np.multiply(diag_part1.eval(), diag_part2.eval())
    diag = np.diag(mult)
    if idx:
        v = w1.eval()[:, range(idx)].T
    else:
        v = w1.eval().T
    v = v * diag
    res = np.dot(v, w0.eval().T)
    return res


def unit_vis(nnmodel):
    [X1, labels1] = utils.load_mnist(
        dataset='training', path='../examples/sda')
    x = X1
    net1 = layers.get_all_layers(nnmodel._network)[1]
    net2 = layers.get_all_layers(nnmodel._network)[2]
    res = linear_comb(net1, net2, x, 36).reshape(36, 28, 28)
    print res.shape
    displayz(res, 6, 6, 0, 28, 28, binary=True)

def map_contour(data, title=None):
    inp = '/mnt/disk1/thanasis/data/11_train.nc'
    nc_fid = Dataset(inp,'r')
    lats = nc_fid.variables['CLAT'][0,0:64]
    lats = lats[:,0]
    lons = nc_fid.variables['XLONG_M'][0,0:64]
    lons = lons[10,:]

    m = Basemap(width=3900000,height=3500000,
     resolution='l',projection='lcc', lat_0 = 50, lon_0 = 16, lat_ts = 40, k_0=1.035)
    m.drawcoastlines(linewidth=1.0,color='#aa7755')
    m.drawcountries(linewidth=1.0,color='#aa7755')
    m.drawmapboundary(linewidth=1.0,color='#aa7755')
    lons, lats = np.meshgrid(lons, lats)
    x, y = m(lons, lats)
    try:
        cs = m.contour(x, y, data, linewidths=2.0, colors = 'k')
    except:
        pass
    # pcl = m.pcolor(x,y,np.squeeze(data))
    # cbar = m.colorbar(pcl, location='bottom', pad="10%")
    # cbar.set_label("$m^2$/$s^2$")
    if title:
        plt.title(title)
    nc_fid.close()

def visualize_conv(nnmodel):
    train = np.load(PATH)
    import theano
    nnmodel._input_layer.input_var = theano.shared(name='input_var',
                                  value=np.asarray(train,
                                  dtype=theano.config.floatX),
                                  borrow=True)
    lconv = layers.get_all_layers(nnmodel._network)[2]
    filt = layers.get_output(lconv).eval()
    # # print np.max(filt[:,6,:,:]),np.min(filt[:,6,:,:])
    filt = filt[500,:]
    kfilt = filt
    import matplotlib.gridspec as gridspec
    # kfilt = np.load('filt.npy')
    idx = range(0,30,3)
    for c,j in enumerate(idx):
        filt = kfilt[j:j+3,:]
        fig = plt.figure(figsize=(12,12),frameon=True)
        gs = gridspec.GridSpec(1, 3,
             wspace=0.1, hspace=0.5)
        # fig.add_subplot(1,3,1)
        # map_contour(train[500,0:4096].reshape(64,64),'500hPa Geopotential Height')
        # fig.add_subplot(1,3,2)
        # map_contour(train[500,4096:8192].reshape(64,64),'700hPa Geopotential Height')
        # fig.add_subplot(1,3,3)
        # map_contour(train[500,8192:12288].reshape(64,64),'900hPa Geopotential Height')
        # plt.show()
        # exit(-1)
        for i in range(3):
            print i
            # fig.add_subplot(1,3,i+1)
            sub = plt.subplot(gs[i])
            plt.axis('off')
            sub.set_xticklabels([])
            sub.set_adjustable('box-forced')
            sub.set_yticklabels([])
            cfilter = filt[i,:]
            cfilter = scipy.misc.imresize(cfilter,(64,64))
            map_contour(cfilter)
        # plt.show()
        fig.savefig("out_conv2_"+str(c)+".png", bbox_inches='tight', pad_inches=0)

def main():
    nnmodel = utils.load_single(sys.argv[1])
    newnn = nnmodel
    visualize_conv(newnn)

if __name__ == '__main__':
    main()
