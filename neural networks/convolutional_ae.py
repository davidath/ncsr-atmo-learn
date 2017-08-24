###############################################################################
# This script is used for training/testing and saving convolutional autoencoders,
# Convolutional autoencoders are defined as deep autoencoder that also
# contain conv layers, maxpool layers, deconv layers and unpool layers.
# GPU support is usually enabled by default, if not see the comments below.
# Input files must be numpy array or saved numpy objects (i.e *.npy),
# hyperparameters and models are saved either as numpy objects (*.npy)
# or as compressed pickle objects. This enables compatibility when using models
# on computers with unknown structure (e.g no GPU support).
###############################################################################

import os

# Removing/Adding comment enables/disables theano GPU support
os.environ['THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
# Removing/Adding comment forces/stops theano CPU support, usually used for model saving
# os.environ['THEANO_FLAGS'] = 'device=cpu,force_device=True'

import sys
sys.path.append('..')
import ConfigParser
import numpy as np
import theano
from theano import tensor as T
import datetime
from datetime import datetime
import dataset_utils as utils
import lasagne
from Unpool2DLayer import Unpool2DLayer
import dataset_utils as utils
from sklearn.utils.linear_assignment_ import linear_assignment
from lasagne.regularization import regularize_layer_params, l2
from modeltemplate import Model

MNIST_PATH = '.'

# Load conv autoencoder configuration file

def load_config(input_path):
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    return cp

# Logging messages such as loss,loading,etc.

def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()

# Load training/testing data

def load_data(cp, train):
    log('Loading data........')
    # If 'input file' parameter not defined then assume MNIST dataset
    if cp.get('Experiment', 'inputfile') == '':
        # Get FULL dataset containing both training/testing
        # In this experiment MNIST used as a testing mechanism in order to
        # test the connection between layers, availabilty,etc. and not for
        # hyperparameter tweaking.
        [X1, labels1] = utils.load_mnist(
            dataset='training', path=MNIST_PATH)
        [X2, labels2] = utils.load_mnist(
            dataset='testing', path=MNIST_PATH)
        X = np.concatenate((X1, X2), axis=0)
        labels = np.concatenate((labels1, labels2), axis=0)
        # Will dataset be used for training? then shuffle the dataset then use
        # the same permutation for the labels.
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p].astype(np.float32) * 0.02
            labels = labels[p]
            prefix = cp.get('Experiment', 'prefix')
            np.save(prefix + '_random_perm.npy', p)
        return [X, labels]
    # If 'input file' is specified then load inputfile, our script assumes that
    # the input file will always be a numpy object
    else:
        try:
            X = np.load(cp.get('Experiment', 'inputfile'))
        except:
            log('Input file must be a saved numpy object (*.npy)')
        # Will dataset be used for training? then shuffle the dataset
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p]
            output = cp[length].get('Experiment', 'output')
            prefix = cp[length].get('Experiment', 'prefix')
            np.save(output+prefix + '_random_perm.npy', p)
        return X
    log('DONE........')

# Model initialization and training

def init(cp, dataset):
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    log(dataset.shape)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    # Scalar used for batch training
    index = T.lscalar()
    # Get "special" parameters for the conv/deconv and max/un pool layers
    conv_filters = int(cp.get('NeuralNetwork', 'convfilters'))
    deconv_filters = conv_filters
    filter_sizes = int(cp.get('NeuralNetwork', 'filtersize'))
    featurex = int(cp.get('NeuralNetwork', 'feature_x'))
    featurey = int(cp.get('NeuralNetwork', 'feature_y'))
    channels = int(cp.get('NeuralNetwork', 'channels'))
    pool_size = int(cp.get('NeuralNetwork', 'pool'))
    # Input layer
    input_layer = network = lasagne.layers.InputLayer(shape=(None, channels * featurex * featurey),
                                                      input_var=input_var)
    # Reshape layer is used to transform 2D input shape to
    # a shape that is compatible with conv layer
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], channels, featurex, featurey))
    # Convolutional layer
    # compatible shape -> (sample_size, channels, Xfeaturesize, Yfeaturesize)
    # e.g (N,M) where sqrt(M) is real, then (N, channels, sqrt(M), sqrt(M))
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=conv_filters, filter_size=(
                                             filter_sizes, filter_sizes),
                                         stride=int(cp.get('NeuralNetwork', 'stride')))
    # Allow double stacking of Convolutional layers
    # Check if experiment configuration file specifies stacking
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.Conv2DLayer(incoming=network,
                                             num_filters=conv_filters, filter_size=(
                                                 dual_conv, dual_conv),
                                             stride=int(cp.get('NeuralNetwork', 'dualstride')))
    except:
        pass
    # Maxpool layer
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network, pool_size=(pool_size, pool_size))
    pool_shape = lasagne.layers.get_output_shape(network)
    # Reshape back to 2D, i.e (N,M)
    flatten = network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    # Introduce noise with DropoutLayer
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    # Reconstruct deep autoencoder structure, more info in stackedautoencoders.py
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    encoder_layer = network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec2')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=lasagne.layers.get_output_shape(flatten)[
                                            1],
                                        )
    # Reshape into deconv compatible shape
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], deconv_filters, pool_shape[2], pool_shape[3]))
    # Unpool layer
    network = lasagne.layers.Upscale2DLayer(incoming=network, scale_factor=(pool_size, pool_size))
    # Check for double conv stacking
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                            num_filters=channels, filter_size=(dual_conv, dual_conv), stride=int(cp.get('NeuralNetwork', 'dualstride')), nonlinearity=None)
    except:
        pass
    # Deconvolutional layer
    network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                         num_filters=channels, filter_size=(filter_sizes, filter_sizes), stride=int(cp.get('NeuralNetwork', 'stride')), nonlinearity=None)
    # Reshape into 2D
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network))
    # Init train function
    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.squared_error(
        prediction, input_layer.input_var).mean()
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate, momentum=0.975)
    train = theano.function(
        inputs=[index, learning_rate], outputs=cost,
        updates=updates, givens={input_layer.input_var: input_var[index:index + batch_size, :]})
    prefix = cp.get('Experiment', 'prefix')
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    # Start training
    lr_decay = int(cp.get('NeuralNetwork', 'lrepochdecay'))
    for epoch in xrange(lw_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            loss = train(row, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0] * 1.0 / batch_size)
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss),
                label='CONV')
        # Decrease learning rate, depending on the decay
        if (epoch % lr_decay == 0 and epoch != 0):
            base_lr = base_lr / 10
        # Save model as object, every 100 epochs
        if (epoch % 100 == 0) and (epoch != 0):
            utils.save(prefix + '_conv.zip', network)
    # Saving hyperparameters and model, numpy objects are used for compatibility
    input_layer.input_var = input_var
    np.save(prefix + '_model.npy',
            lasagne.layers.get_all_param_values(network))
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    log(hidden.shape)
    np.save(prefix + '_hidden.npy', hidden)

# Loads hyperparameters from previous pretrained experiment and reconstructs the model.
# Used for model archiving or testing.

def init_pretrained(cp, dataset):
    prefix = cp.get('Experiment', 'prefix')
    print dataset.shape
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    conv_filters = int(cp.get('NeuralNetwork', 'convfilters'))
    deconv_filters = conv_filters
    filter_sizes = int(cp.get('NeuralNetwork', 'filtersize'))
    featurex = int(cp.get('NeuralNetwork', 'feature_x'))
    featurey = int(cp.get('NeuralNetwork', 'feature_y'))
    channels = int(cp.get('NeuralNetwork', 'channels'))
    pool_size = int(cp.get('NeuralNetwork', 'pool'))
    # Reconstructing the convolutional autoencoder structure, more info above
    input_layer = network = lasagne.layers.InputLayer(shape=(None, channels * featurex * featurey),
                                                      input_var=input_var)
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], channels, featurex, featurey))
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=conv_filters, filter_size=(
                                             filter_sizes, filter_sizes),
                                         stride=int(cp.get('NeuralNetwork', 'stride')))
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.Conv2DLayer(incoming=network,
                                             num_filters=conv_filters, filter_size=(
                                                 dual_conv, dual_conv),
                                             stride=int(cp.get('NeuralNetwork', 'dualstride')))
    except:
        pass
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network, pool_size=(pool_size, pool_size))
    pool_shape = lasagne.layers.get_output_shape(network)
    flatten = network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    encoder_layer = network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec3')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec2')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'dec1')),
                                        )
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=lasagne.layers.get_output_shape(flatten)[
                                            1],
                                        )
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], deconv_filters, pool_shape[2], pool_shape[3]))
    network = lasagne.layers.Upscale2DLayer(incoming=network, scale_factor=(pool_size, pool_size))
    try:
        dual_conv = int(cp.get('NeuralNetwork', 'dualconv'))
        network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                            num_filters=channels, filter_size=(dual_conv, dual_conv), stride=int(cp.get('NeuralNetwork', 'dualstride')), nonlinearity=None)
    except:
        pass
    network = lasagne.layers.TransposedConv2DLayer(incoming=network,
                                         num_filters=channels, filter_size=(filter_sizes, filter_sizes), stride=int(cp.get('NeuralNetwork', 'stride')), nonlinearity=None)
    network = lasagne.layers.ReshapeLayer(
        incoming=network, shape=([0], -1))
    # Neural network has been reconstructed, yet hyperparameters are randomly
    # initialized, load pretrained hyperparameters
    lasagne.layers.set_all_param_values(
        network, np.load(prefix + '_model.npy'))
    # Initalize model template object that is used for saving/archiving purposes.
    # model template is a custom class, see modeltemplate.py for more info
    model = Model(input_layer=input_layer, encoder_layer=encoder_layer,
                  decoder_layer=network, network=network)
    model.save(prefix+'_model.zip')
    # If the function was called for testing purposes, save hidden and output layer
    # results.
    input_layer.input_var = input_var
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    np.save(prefix + '_pretrained_hidden.npy', hidden)
    output = lasagne.layers.get_output(network).eval()
    np.save(prefix + '_pretrained_output.npy', output)

# Control flow
def main(path, train):
    cp = load_config(path)
    # Check if MNIST dataset was loaded
    try:
        [X, labels] = load_data(cp, train)
    except:
        X = load_data(cp, train)
    # Check training/testing flag
    if train == 'train':
        init(cp, X)
    else:
        init_pretrained(cp, X)


from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Training/Testing convolutional autoencoders')
    # Configuration file path
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    # Training/testing flag
    parser.add_argument('-t', '--train', required=True, type=str,
                        help='training/testing')
    opts = parser.parse_args()
    getter = attrgetter('input', 'train')
    inp, train = getter(opts)
    main(inp, train)
