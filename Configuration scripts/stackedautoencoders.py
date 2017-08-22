###############################################################################
# This script is used for training/testing and saving deep autoencoders,
# Deep autoencoders are defined by using multiple encoders serially connected.
# Training deep autoencoders by training each autoencoder individually
# (layerwise) and then we train the deep network as a whole for hyperparameter
# tweaking.
# Therefore this class assumes that the autoencoders have been already pretrained.
# GPU support is usually enabled by default, if not see the comments
# below. Input files must be numpy array or saved numpy objects (i.e *.npy),
# hyperparameters and models are saved either as numpy objects (*.npy)
# or as compressed pickle objects. This enables compatibility when using models
# on computers with unknown structure (e.g no GPU support).
###############################################################################

import os

# Removing/Adding comment enables/disables theano GPU support
os.environ[
    'THEANO_FLAGS'] = 'mode=FAST_RUN,device=gpu,floatX=float32,nvcc.flags=-D_FORCE_INLINES'
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
from modeltemplate import Model

MNIST_PATH = '/mnt/disk1/thanasis/autoencoder/'

# Load config files, due to the fact that we need to train each autoencoder
# individually we need to load multiple autoencoder config files
# We assume that the config file for each autoencoder is PREFIX_(POSITION-1).ini
# e.g if prefix=MNIST and we need to load the first autoencoder then the config file
# should be named MNIST_(1-1).ini -> MNIST_0.ini.
# The length parameter is the enumeration of the last config file, therefore
# cp[length] is the config file of the deep neural network

def load_config(input_path, prefix, length):
    CP = []
    for i in xrange(length):
        cp = ConfigParser.ConfigParser()
        cp.read(prefix + '_' + str(i) + '.ini')
        CP.append(cp)
    cp = ConfigParser.ConfigParser()
    cp.read(input_path)
    CP.append(cp)
    return CP

# Logging messages such as loss,loading,etc.

def log(s, label='INFO'):
    sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    sys.stdout.flush()


def load_data(cp, length, train):
    log('Loading data........')
    # If 'input file' parameter not defined then assume MNIST dataset
    if cp[length].get('Experiment', 'inputfile') == '':
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
            prefix = cp[length].get('Experiment', 'prefix')
            np.save(prefix + '_sda_random_perm.npy', p)
        return [X, labels]
    # If 'input file' is specified then load inputfile, our script assumes that
    # the input file will always be a numpy object
    else:
        try:
            X = np.load(cp[length].get('Experiment', 'inputfile'))
        except:
            log('Input file must be a saved numpy object (*.npy)')
        # Will dataset be used for training? then shuffle the dataset
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p]
            prefix = cp[length].get('Experiment', 'prefix')
            np.save(prefix + '_sda_random_perm.npy', p)
        return X
    log('DONE........')

# Initalization and training of the deep neural network

def init(cp, dataset, length):
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    batch_size = int(cp[length].get('NeuralNetwork', 'batchsize'))
    # Get learning decay from config file
    lr_decay = int(cp[length].get('NeuralNetwork', 'lrepochdecay'))
    prefix = cp[length].get('Experiment', 'prefix')
    log(dataset.shape)
    # Create deep neural network
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    # Scalar used for batch training
    index = T.lscalar()
    # Input layer
    input_layer = network = lasagne.layers.InputLayer(shape=(None, int(cp[0].get('NeuralNetwork', 'inputlayer'))),
                                                      input_var=input_var)
    # Layerwise encoders
    # For each autoencoder: if first autoencoder then connect hidden to input layer,
    # otherwise connect hidden to previous hidden layer. Each hidden layer is
    # initialized with their respective pretrained hyperparameters
    for i in xrange(length):
        enc_act = cp[i].get('NeuralNetwork', 'encoderactivation')
        network = lasagne.layers.DenseLayer(incoming=input_layer if i == 0 else network,
                                            num_units=int(
                                                cp[i].get('NeuralNetwork', 'hiddenlayer')),
                                            W=np.load(
                                                prefix + '_' + str(i) + '_W1.npy'),
                                            b=np.load(
                                                prefix + '_' + str(i) + '_b1.npy'),
                                            nonlinearity=relu if enc_act == 'ReLU' else linear)
    # Network status: Input layer->Encoding/Hidden layer 0->Encoding/Hidden layer 1
    # ->...->Encoding/Hidden layer <length>
    encoder_layer = network

    # Layerwise decoders
    # Each decoder is being connected backwards. This means that the <length>
    # decoder gets connected to the <length> encoder.
    # For each autoencoder: connect decoder/output layer to previous layer.
    # Each hidden layer is initialized with their respective pretrained hyperparameters
    for i in reversed(xrange(length)):
        dec_act = cp[i].get('NeuralNetwork', 'decoderactivation')
        network = lasagne.layers.DenseLayer(incoming=network,
                                            num_units=int(
                                                cp[i].get('NeuralNetwork', 'outputlayer')),
                                            W=np.load(
                                                prefix + '_' + str(i) + '_W2.npy'),
                                            b=np.load(
                                                prefix + '_' + str(i) + '_b2.npy'),
                                            nonlinearity=relu if dec_act == 'ReLU' else linear)

    # Print Deep neural network structure for debugging purposes
    print '> Deep neural net Topology'
    print '----------------------------------'
    print lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network))
    print '----------------------------------'

    # Create train function
    learning_rate = T.scalar(name='learning_rate')
    index = T.lscalar()
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.squared_error(
        prediction, input_layer.input_var).mean()
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate)
    train = theano.function(
        inputs=[index, learning_rate], outputs=cost, updates=updates, givens={input_layer.input_var: input_var[index:index + batch_size, :]})
    # Get training epochs from config file
    deep_epochs = int(cp[length].get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp[length].get('NeuralNetwork', 'learningrate'))
    # Start training
    print '> Deep neural net trainining'
    for epoch in xrange(deep_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            loss = train(row, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0] / batch_size)
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss), label='DNT')
        # Decrease learning rate, depending on the decay
        if (epoch % lr_decay == 0 and epoch != 0):
            base_lr = base_lr / 10
        # Save model as object, every 100 epochs
        if (epoch % 100 == 0) and (epoch != 0):
            np.save(prefix + '_sda_model.npy',
                    lasagne.layers.get_all_param_values(network))
    # Saving hyperparameters and model, numpy objects are used for compatibility
    input_layer.input_var = input_var
    np.save(prefix + '_sda_model.npy',
            lasagne.layers.get_all_param_values(network))
    np.save(prefix + '_sda_W1.npy', encoder_layer.W.eval())
    np.save(prefix + '_sda_W2.npy', network.W.eval())
    np.save(prefix + '_sda_b1.npy', encoder_layer.b.eval())
    np.save(prefix + '_sda_b2.npy', network.b.eval())
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    # Log hidden layer shape for debugging purposes
    log(hidden.shape)
    np.save(prefix + '_sda_hidden.npy', hidden)
    np.save(prefix + '_sda_output.npy',
            lasagne.layers.get_output(network).eval())

# Loads hyperparameters from previous pretrained experiment and reconstructs the model.
# Used for model archiving or testing.

def init_pretrained(cp, dataset, length):
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    batch_size = int(cp[length].get('NeuralNetwork', 'batchsize'))
    lr_decay = int(cp[length].get('NeuralNetwork', 'lrepochdecay'))
    prefix = cp[length].get('Experiment', 'prefix')
    log(dataset.shape)
    # Reconstruct deep network
    # Input layer
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    input_layer = network = lasagne.layers.InputLayer(shape=(None, int(cp[0].get('NeuralNetwork', 'inputlayer'))),
                                                      input_var=input_var)
    # Layerwise encoders
    for i in xrange(length):
        enc_act = cp[i].get('NeuralNetwork', 'encoderactivation')
        network = lasagne.layers.DenseLayer(incoming=input_layer if i == 0 else network,
                                            num_units=int(
                                                cp[i].get('NeuralNetwork', 'hiddenlayer')),
                                            W=np.load(
                                                prefix + '_' + str(i) + '_W1.npy'),
                                            b=np.load(
                                                prefix + '_' + str(i) + '_b1.npy'),
                                            nonlinearity=relu if enc_act == 'ReLU' else linear)
    encoder_layer = network

    # Layerwise decoders
    for i in reversed(xrange(length)):
        dec_act = cp[i].get('NeuralNetwork', 'decoderactivation')
        network = lasagne.layers.DenseLayer(incoming=network,
                                            num_units=int(
                                                cp[i].get('NeuralNetwork', 'outputlayer')),
                                            W=np.load(
                                                prefix + '_' + str(i) + '_W2.npy'),
                                            b=np.load(
                                                prefix + '_' + str(i) + '_b2.npy'),
                                            nonlinearity=relu if dec_act == 'ReLU' else linear)
    # Neural network has been reconstructed, yet hyperparameters are initialized
    # with the layerwise hyperparameters, load pretrained deep hyperparameters
    lasagne.layers.set_all_param_values(
        network, np.load(prefix + '_sda_model.npy'))
    # Initalize model template object that is used for saving/archiving purposes.
    # model template is a custom class, see modeltemplate.py for more info
    model = Model(input_layer=input_layer, encoder_layer=encoder_layer,
                  decoder_layer=network, network=network)
    model.save(prefix + '_model.zip')
    # If the function was called for testing purposes, save hidden and output layer
    # results.
    input_layer.input_var = input_var
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    np.save(prefix + '_sda_pretrained_hidden.npy', hidden)
    output = lasagne.layers.get_output(network).eval()
    np.save(prefix + '_sda_pretrained_output.npy', output)

# Control flow
def main(path, length, train, pref):
    cp = load_config(path, pref, length)
    # Check if MNIST dataset was loaded
    try:
        [X, labels] = load_data(cp, length, train)
    except:
        X = load_data(cp, length, train)
    # Check training/testing flag
    if train == 'train':
        init(cp, X, length)
    else:
        init_pretrained(cp, X, length)


from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Training/Testing stacked autoencoders')
    # Configuration file path
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    # Number of pretrained layerwise autoencoders
    parser.add_argument('-n', '--number', required=True, type=int,
                        help='number of autoencoders ')
    # Testing/Training flag
    parser.add_argument('-t', '--train', required=True, type=str,
                        help='training/testing')
    # Prefix of pretrained layerwise autoencoders
    parser.add_argument('-p', '--prefix', required=True, type=str,
                        help='sbufiles prefix')
    opts = parser.parse_args()
    getter = attrgetter('input', 'number', 'train', 'prefix')
    inp, length, train, pref = getter(opts)
    main(inp, length, train, pref)
