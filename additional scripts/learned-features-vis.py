#!/usr/bin/env python
import os
# os.environ['THEANO_FLAGS'] = '
import numpy as np
import sys
sys.path.append('..')
import theano
import theano.tensor as T
import dataset_utils as utils
import lasagne.layers as layers
import lasagne
from datetime import datetime
from disputil import displayz

def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

def AM(nnmodel):
    # [X1, labels1] = utils.load_mnist(
    #         dataset='training', path='../examples/sda')
    x = T.matrix('x')
    nnmodel._input_layer.input_var = x
    activation = layers.get_output(nnmodel._encoder_layer).mean()
    # activation = layers.get_output(layers.get_all_layers(nnmodel._network)[4]).mean()
    grad = T.grad(activation, x)
    grad /= T.sqrt(T.mean(T.square(grad)))
    gasc = theano.function(inputs=[x],outputs=[activation,grad])

    _X = np.random.random(size=(36,784)) * 20 + 128
    base_lr = 0.000001
    i = 0
    while True:
        try:
            loss, grad_value = gasc(_X)
            if i % 10 == 0:
                log(str(i) + ' ' + str(loss), label='DNT')
            _X += grad_value * base_lr
            i += 1
        except KeyboardInterrupt:
            break
    displayz(grad_value.reshape(36,28,28),6,6,0,28,28,binary=True)

def linear_comb(net1,net2,x,idx=None):
    w0 = net1.W
    w1 = net2.W
    diag_part1 = net1.nonlinearity(T.dot(x,w0))
    diag_part2 = 1 - diag_part1
    mult = np.multiply(diag_part1.eval(),diag_part2.eval())
    diag = np.diag(mult)
    if idx:
        v = w1.eval()[:,range(idx)].T
    else:
        v = w1.eval().T
    v = v*diag
    res = np.dot(v,w0.eval().T)
    return res

def unit_vis(nnmodel):
    [X1, labels1] = utils.load_mnist(
            dataset='training', path='../examples/sda')
    x = X1
    net1 = layers.get_all_layers(nnmodel._network)[2]
    net2 = layers.get_all_layers(nnmodel._network)[3]
    res = linear_comb(net1,net2,x,36)
    print res.shape
    # displayz(res,6,6,0,28,28,binary=True)

def main():
    nnmodel = utils.load_single(sys.argv[1])
    newnn = nnmodel
    # AM(newnn)
    unit_vis(nnmodel)


if __name__ == '__main__':
    main()
