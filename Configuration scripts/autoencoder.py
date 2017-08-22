###############################################################################
# This script is used for training/testing and saving shallow autoencoders,
# Shallow autoencoders (denoising) are defined by their simple structure:
# Input layer->Dropout layer->Hidden/Encoding layer->Output/Decoding layer.
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
from modeltemplate import Model

MNIST_PATH = '/mnt/disk1/thanasis/autoencoder/'

# Load autoencoder configuration file


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
            num = cp.get('Experiment', 'num')
            np.save(prefix + '_' + num + 'random_perm.npy', p)
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
            prefix = cp.get('Experiment', 'prefix')
            num = cp.get('Experiment', 'num')
            np.save(prefix + '_' + num + 'random_perm.npy', p)
        return X
    log('DONE........')

# Model initialization and training

def init(cp, dataset):
    # Initialize theano tensors
    log(dataset.shape)
    # Input layer
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                dtype=theano.config.floatX),
                              borrow=True)
    # Scalar used for batch training
    index = T.lscalar()
    # Initialize neural network
    # Get activation functions from config file
    enc_act = cp.get('NeuralNetwork', 'encoderactivation')
    dec_act = cp.get('NeuralNetwork', 'decoderactivation')
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    # Learning decay
    lr_decay = int(cp.get('NeuralNetwork', 'lrepochdecay'))
    # Stacking layers into network
    # Input
    input_layer = network = lasagne.layers.InputLayer(shape=(None, int(cp.get('NeuralNetwork', 'inputlayer'))),
                                                      input_var=input_var)
    # Introduce noise with Dropout
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    # Encoding layer
    encoder_layer = network = lasagne.layers.DenseLayer(incoming=network,
                                                        num_units=int(
                                                            cp.get('NeuralNetwork', 'hiddenlayer')),
                                                        W=lasagne.init.Normal(),
                                                        nonlinearity=relu if enc_act == 'ReLU' else linear)
    # Decoding/Output layer
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'outputlayer')),
                                        W=lasagne.init.Normal(),
                                        nonlinearity=relu if dec_act == 'ReLU' else linear)

    # Create train function
    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.squared_error(
        prediction, input_layer.input_var).mean()
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate)
    train = theano.function(
        inputs=[index, learning_rate], outputs=cost,
        updates=updates, givens={input_layer.input_var: input_var[index:index + batch_size, :]})
    # Get epochs and starting learning rate
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    # Start training
    num = cp.get('Experiment', 'num')
    for epoch in xrange(lw_epochs):
        epoch_loss = 0
        for row in xrange(0, dataset.shape[0], batch_size):
            loss = train(row, base_lr)
            epoch_loss += loss
        epoch_loss = float(epoch_loss) / (dataset.shape[0] / batch_size)
        # Print loss every 10 epochs
        if epoch % 10 == 0:
            log(str(epoch) + ' ' + str(epoch_loss),
                label='LWT-Layer' + str(num))
        # Decrease learning rate, depending on the decay
        if (epoch % lr_decay == 0 and epoch != 0):
            base_lr = base_lr / 10
        # Save model as object, every 100 epochs
        if (epoch % 100 == 0) and (epoch != 0):
            utils.save('autoenc_' + num + '.zip', network)
    # Saving hyperparameters and model, numpy objects are used for compatibility
    input_layer.input_var = input_var
    prefix = cp.get('Experiment', 'prefix')
    np.save(prefix + '_' + num + '_model.npy',
            lasagne.layers.get_all_param_values(network))
    np.save(prefix + '_' + num + '_W1.npy', encoder_layer.W.eval())
    np.save(prefix + '_' + num + '_W2.npy', network.W.eval())
    np.save(prefix + '_' + num + '_b1.npy', encoder_layer.b.eval())
    np.save(prefix + '_' + num + '_b2.npy', network.b.eval())
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    # Log hidden layer shape for debugging purposes
    log(hidden.shape)
    np.save(prefix + '_hidden.npy', hidden)

# Loads hyperparameters from previous pretrained experiment and reconstructs the model.
# Used for model archiving or testing.

def init_pretrained(cp, dataset):
    prefix = cp.get('Experiment', 'prefix')
    num = cp.get('Experiment', 'num')
    # Initialize theano tensors
    log(dataset.shape)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    index = T.lscalar()
    # Reconstruct pretrained neural network
    # Get activation functions from config file
    enc_act = cp.get('NeuralNetwork', 'encoderactivation')
    dec_act = cp.get('NeuralNetwork', 'decoderactivation')
    relu = lasagne.nonlinearities.rectify
    linear = lasagne.nonlinearities.linear
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    # Get learning decay from config file
    lr_decay = int(cp.get('NeuralNetwork', 'lrepochdecay'))
    # Stacking layers into network
    # Input layer
    input_layer = network = lasagne.layers.InputLayer(shape=(None, int(cp.get('NeuralNetwork', 'inputlayer'))),
                                                      input_var=input_var)
    # Introduce noise with DropoutLayer
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    # Encoding/Hidden layer
    encoder_layer = network = lasagne.layers.DenseLayer(incoming=network,
                                                        num_units=int(
                                                            cp.get('NeuralNetwork', 'hiddenlayer')),
                                                        W=np.load(
                                                            prefix + '_' + num + '_W1.npy'),
                                                        b=np.load(
                                                            prefix + '_' + num + '_b1.npy'),
                                                        nonlinearity=relu if enc_act == 'ReLU' else linear)
    # Decoding/Output layer
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'outputlayer')),
                                        W=np.load(prefix + '_' +
                                                  num + '_W2.npy'),
                                        b=np.load(prefix + '_' +
                                                  num + '_b2.npy'),
                                        nonlinearity=relu if dec_act == 'ReLU' else linear)
    # Neural network has been reconstructed, yet hyperparameters are randomly
    # initialized, load pretrained hyperparameters
    lasagne.layers.set_all_param_values(
        network, np.load(prefix + '_' + num + '_model.npy'))
    # Initalize model template object that is used for saving/archiving purposes.
    # model template is a custom class, see modeltemplate.py for more info
    model = Model(input_layer=input_layer, encoder_layer=encoder_layer,
                  decoder_layer=network, network=network)
    # Compressed pickle
    model.save(prefix + '_model.zip')
    # If the function was called for testing purposes, save hidden and output layer
    # results.
    input_layer.input_var = input_var
    hidden = lasagne.layers.get_output(encoder_layer).eval()
    np.save(prefix + '_' + num + '_pretrained_hidden.npy', hidden)
    output = lasagne.layers.get_output(network).eval()
    np.save(prefix + '_' + num + '_pretrained_output.npy', output)

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
    parser = ArgumentParser(description='Training/Testing shallow autoencoders')
    # Configuration file path
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config path file')
    # Training/testing flag
    parser.add_argument('-t', '--train', required=True, type=str,
                        help='training/testing')
    opts = parser.parse_args()
    getter = attrgetter('input', 'train')
    inp, train = getter(opts)
    main(inp, train)
