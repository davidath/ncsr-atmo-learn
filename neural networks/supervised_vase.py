###############################################################################
# This script is used for training/testing and saving a supervised classification
# of the "vase" model, this model expects weather data and dispersions as input.
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
from sklearn.metrics import accuracy_score
from lasagne.regularization import regularize_layer_params, l2
from modeltemplate import Model
import scipy.misc
from sklearn.preprocessing import maxabs_scale,minmax_scale


# Load configuration file

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
    if cp.get('Experiment', 'inputfile') == '':
        log('You have to specify inputfile, MNIST dataset is not available for this model')
        exit(-1)
    else:
        try:
            X = np.load(cp.get('Experiment', 'inputfile'))
            try:
               X = X[cp.get('Experiment','label')]
            except:
               pass
        except:
            log('Input file must be a saved numpy object (*.npy)')
        # Will dataset be used for training? then shuffle the dataset
        if train == 'train':
            p = np.random.permutation(X.shape[0])
            X = X[p]
            prefix = cp.get('Experiment', 'prefix')
            output = cp.get('Experiment','output')
            np.save(output+prefix + '_random_perm.npy', p)
        return X
    log('DONE........')

# Create weather input that is used to get output of the network

def make_weather(cp, dataset):
    # Get Conv layer parameter for the "weather" neural network
    varidx = int(cp.get('Weather', 'varidx'))
    lvlidx = int(cp.get('Weather', 'lvlidx'))
    featurex = int(cp.get('Weather', 'feature_x'))
    featurey = int(cp.get('Weather', 'feature_y'))
    # Retrieve the all weather data from "test" dataset
    dataset = dataset[:, 4]
    dataset = [x for x in dataset]
    dataset = np.array(dataset)
    # Retrieve some variable from the test dataset in specific level
    # VarIdx [0=U,1=V,2=GHT] and LvlIdx [0=500 hPa,1=700 hPa,2=900 hPa]
    # e.g VarIdx=2,LvlIdx=1 -> GHT 700hPa
    dataset = dataset[:, varidx, lvlidx, :, :]
    # Reshape to 2D
    dataset = dataset.reshape(dataset.shape[0],featurex*featurey)
    # Scale
    dataset = minmax_scale(dataset)
    # Reshape back to (sample_size, gridx, gridy)
    dataset = dataset.reshape(dataset.shape[0],featurex,featurey)
    # Log shape for debugging purposes
    log(dataset.shape)
    featurex = int(cp.get('Weather', 'feature_x'))
    featurey = int(cp.get('Weather', 'feature_y'))
    channels = int(cp.get('Weather', 'channels'))
    # Reshape to Conv compatible shape
    dataset = dataset.reshape(dataset.shape[0], channels, featurex, featurey)
    # Create symbolic var for weather input
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    return input_var


# Initialize the "Weather" neural network. Due to the fact that the classification
# models work with weather and dispersion data as input, we construct a small network
# for each input that will later concatenate their inputs.

def init_weather_conv(cp, dataset):
    # make_weather function could be used for the creation of symoblic var of the
    # input, yet we had issues in past experiments with the process of stacking layers
    # and initializing their parameters on individual functions, therefore we decided on
    # initializing every aspect of each network in the same function.

    # Get Conv layer parameter for the "weather" neural network
    featurex = int(cp.get('Weather', 'feature_x'))
    featurey = int(cp.get('Weather', 'feature_y'))
    varidx = int(cp.get('Weather', 'varidx'))
    lvlidx = int(cp.get('Weather', 'lvlidx'))
    # Retrieve the all weather data from "test" dataset
    dataset = dataset[:, 4]
    dataset = [x for x in dataset]
    dataset = np.array(dataset)
    # Retrieve some variable from the test dataset in specific level
    # VarIdx [0=U,1=V,2=GHT] and LvlIdx [0=500 hPa,1=700 hPa,2=900 hPa]
    # e.g VarIdx=2,LvlIdx=1 -> GHT 700hPa
    dataset = dataset[:, varidx, lvlidx, :, :]
    # Reshape to 2D
    dataset = dataset.reshape(dataset.shape[0],featurex*featurey)
    # Scale
    dataset = minmax_scale(dataset)
    # Log shape for debugging purposes
    log(dataset.shape)
    channels = int(cp.get('Weather', 'channels'))
    # Reshape to Conv compatible shape
    dataset = dataset.reshape(dataset.shape[0], channels, featurex, featurey)
    # Create symbolic var for weather input
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    # Get Weather nnet parameters
    def_filters = int(cp.get('Default','filters'))
    def_filter_size = int(cp.get('Default','filter_size'))
    def_stride = int(cp.get('Default','stride'))
    def_padding = int(cp.get('Default','padding'))
    channels = int(cp.get('Weather', 'channels'))
    pool_size = int(cp.get('W1', 'pool'))
    # Layer stacking (Weather Subnetwork #1)
    # ------------------------------------------------------------------------
    # Input layer
    input_layer = network = lasagne.layers.InputLayer(shape=(None, channels, featurex, featurey),
                                                      input_var=input_var)
    # Convolutional layer #1
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=def_filters, filter_size=(
                                             def_filter_size, def_filter_size),
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    # Convolutional layer #2 (by default)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=int(cp.get('W1', 'convfilters')),
                                          filter_size=(
                                             int(cp.get('W1', 'filtersize')), int(cp.get('W1', 'filtersize'))),
                                         stride=int(cp.get('W1', 'stride')),
                                         pad=int(cp.get('W1','pad'))
                                         )
    # Maxpool
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network, pool_size=(pool_size, pool_size))
    # End of layer stacking for Weather Subnetwork #1
    # ########################################################################

    # Layer stacking (Weather Subnetwork #2)
    # ------------------------------------------------------------------------
    # Input layer
    network2 = input_layer
    # Convolutional layer #1
    network2 = lasagne.layers.Conv2DLayer(incoming=network2,
                                         num_filters=def_filters, filter_size=(
                                             def_filter_size, def_filter_size),
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    # Convolutional layer #2 (by default)
    network2 = lasagne.layers.Conv2DLayer(incoming=network2,
                                         num_filters=int(cp.get('W2', 'convfilters')),
                                          filter_size=(
                                             int(cp.get('W2', 'filtersize')), int(cp.get('W2', 'filtersize'))),
                                         stride=int(cp.get('W2', 'stride')),
                                         pad=int(cp.get('W2','pad'))
                                         )
    # Maxpool
    network2 = lasagne.layers.MaxPool2DLayer(
        incoming=network2, pool_size=(pool_size, pool_size))
    # Flatten both subnetworks for concatenation
    network = lasagne.layers.FlattenLayer(network)
    network2 = lasagne.layers.FlattenLayer(network2)
    # Connect subnetworks
    out = lasagne.layers.ConcatLayer(incomings=(network,network2),axis=1)
    # log('Printing Weather Net Structure.......')
    # log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network)))
    return [input_layer, input_var, out]

# Create dispersion input that is used to get output of the network

def make_disp(cp, dataset):
    # Retrieve all dispersions from "test" dataset
    dataset = dataset[:, 3]
    # Get parameters
    featurex = int(cp.get('Dispersion', 'feature_x'))
    featurey = int(cp.get('Dispersion', 'feature_y'))
    channels = int(cp.get('Dispersion', 'channels'))
    # Resize each dispersion, the dispersion are treated as image
    dataset = [scipy.misc.imresize(x, (featurex, featurey)) for x in dataset]
    dataset = np.array(dataset)
    # Reshape to 2D
    dataset = dataset.reshape(dataset.shape[0],featurex*featurey)
    # Scale
    dataset = maxabs_scale(dataset)
    # Reshape to conv compatible shape
    dataset = dataset.reshape(dataset.shape[0], channels, featurex, featurey)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    return input_var

# Initialize the "Dispersion" neural network. Due to the fact that the classification
# models work with weather and dispersion data as input, we construct a small network
# for each input that will later concatenate their inputs.

def init_disp_conv(cp, dataset):
    # make_disp function could be used for the creation of symoblic var of the
    # input, yet we had issues in past experiments with the process of stacking layers
    # and initializing their parameters on individual functions, therefore we decided on
    # initializing every aspect of each network in the same function.

    # Retrieve all dispersions from "test" dataset
    dataset = dataset[:, 3]
    # Get parameters
    featurex = int(cp.get('Dispersion', 'feature_x'))
    featurey = int(cp.get('Dispersion', 'feature_y'))
    channels = int(cp.get('Dispersion', 'channels'))
    pool_size = int(cp.get('D1', 'pool'))
    def_filters = int(cp.get('Default','filters'))
    def_filter_size = int(cp.get('Default','filter_size'))
    def_stride = int(cp.get('Default','stride'))
    def_padding = int(cp.get('Default','padding'))
    # Resize each dispersion, the dispersion are treated as image
    dataset = [scipy.misc.imresize(x, (featurex, featurey)) for x in dataset]
    dataset = np.array(dataset)
    # Reshape to 2D
    dataset = dataset.reshape(dataset.shape[0],featurex*featurey)
    # Scale
    dataset = maxabs_scale(dataset)
    # Reshape to conv compatible shape
    dataset = dataset.reshape(dataset.shape[0], channels, featurex, featurey)
    input_var = theano.shared(name='input_var', value=np.asarray(dataset,
                                                                 dtype=theano.config.floatX),
                              borrow=True)
    # Layer stacking (Dispersion subnetwork #1)
    # ------------------------------------------------------------------------
    # Input layer
    input_layer = network = lasagne.layers.InputLayer(shape=(None, channels, featurex, featurey),
                                                      input_var=input_var)
    # Introduce padding
    pad = network = lasagne.layers.PadLayer(incoming=network,width=((0,1),(1,0)))
    # Convolutional layer #1
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=def_filters, filter_size=(
                                             def_filter_size, def_filter_size),
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    # Convolutional layer #2 (by default)
    network = lasagne.layers.Conv2DLayer(incoming=network,
                                         num_filters=int(cp.get('D1', 'convfilters')), filter_size=(
                                             int(cp.get('D1', 'filtersize')), int(cp.get('D1', 'filtersize'))),
                                         stride=int(cp.get('D1', 'stride')),
                                         pad=int(cp.get('D1','pad')))
    # Maxpool
    network = lasagne.layers.MaxPool2DLayer(
        incoming=network, pool_size=(pool_size, pool_size))
    # End of layer stacking for subnetwork #1
    # ########################################################################

    # Layer stacking (Dispersion subnetwork #2)
    # ------------------------------------------------------------------------
    # Subnetwork #1 till padding
    network2 = pad
    # Convolutional layer #1 (different params)
    network2 = lasagne.layers.Conv2DLayer(incoming=network2,
                                         num_filters=def_filters, filter_size=(
                                             def_filter_size, def_filter_size),
                                         stride=def_stride,
                                         pad=def_padding
                                         )
    # Convolutional layer #2 (different params)
    network2 = lasagne.layers.Conv2DLayer(incoming=network2,
                                         num_filters=int(cp.get('D2', 'convfilters')), filter_size=(
                                             int(cp.get('D2', 'filtersize')), int(cp.get('D2', 'filtersize'))),
                                         stride=int(cp.get('D2', 'stride')),
                                         pad=int(cp.get('D2','pad')))
    # Maxpool
    network2 = lasagne.layers.MaxPool2DLayer(
        incoming=network2, pool_size=(pool_size, pool_size))
    # Flatten both subnetworks
    network = lasagne.layers.FlattenLayer(network)
    network2 = lasagne.layers.FlattenLayer(network2)
    # Connect subnetworks
    out = lasagne.layers.ConcatLayer(incomings=(network,network2),axis=1)
    # log('Printing Dispersion Net Structure.......')
    # log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network)))
    return [input_layer, input_var, out]

# Initalization of the deep neural network where the "Weather" and "Disperion"
# network outputs get concatenated

def init(cp, dataset):
    prefix = cp.get('Experiment','prefix')
    output = cp.get('Experiment','output')
    # Create "Weather" neural network
    [win_layer, win, weather_net] = init_weather_conv(cp, dataset)
    # Create "Dispersion" neural network
    [din_layer, din, disp_net] = init_disp_conv(cp, dataset)
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    # Scalar used for batch training
    index = T.lscalar()
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    # ConcatLayer is used in order to connect the outputs of each network
    concat = network = lasagne.layers.ConcatLayer(
        incomings=(weather_net, disp_net), axis=1)
    # Introduce noise
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    # Fully connected #1, Shape: close to original
    test = network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    # Fully connected #2, Shape: compressing
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    # Fully connected #3, Shape: convert to original
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    # Softmax
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'stationnum')),
                                        nonlinearity=lasagne.nonlinearities.softmax
                                        )
    # Check for pretrained model, the dataset was split into 4 due to the fact that
    # GPU memory couldn't support the parameters of network as well as the dataset.
    try:
        # If pretrained model exists then load it's parameters to this one.
        params = np.load(output+'sharing_model.npy')
        lasagne.layers.set_all_param_values(network,params)
        log('Found pretrained model.....')
        log('Training with pretrained weights......')
    except:
        pass
    # Initialize train function
    lr_decay = int(cp.get('NeuralNetwork','lrdecayepoch'))
    dist_var = T.fmatrix('targets')
    learning_rate = T.scalar(name='learning_rate')
    prediction = lasagne.layers.get_output(network)
    cost = lasagne.objectives.categorical_crossentropy(
        prediction, dist_var).mean()
    # acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
    #               dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(
        network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
        cost, params, learning_rate=learning_rate, momentum=0.9)
    train = theano.function(
        inputs=[index, dist_var, learning_rate], outputs=cost,
        updates=updates, givens={win_layer.input_var: win[index:index + batch_size, :],
                             din_layer.input_var: din[index:index + batch_size, :]})
    # Start training
    # The training of this model is a lengthy procedure that we might want to
    # interrupt before reaching the epoch objective. Introduce model saving when
    # CTRL+C is detected
    try:
        for epoch in xrange(lw_epochs):
            epoch_loss = 0
            for row in xrange(0, dataset.shape[0], batch_size):
                pos = dataset[row:row+batch_size,1]
                x = np.zeros(shape=(len(pos),20),dtype=np.float32)
                for i,arr in enumerate(x):
                    arr[pos[i]] = 1
                loss = train(row, x, base_lr)
                epoch_loss += loss
            epoch_loss = float(epoch_loss) / (dataset.shape[0] * 1.0 / batch_size)
            # Print loss every 10 epochs
            if epoch % 10 == 0:
                log(str(epoch) + ' ' + str(epoch_loss),
                    label='Supervised')
            # Decrease learning rate, depending on the decay
            if (epoch % lr_decay == 0 ) and (epoch != 0):
                base_lr = base_lr / 10.0
            # Save model as object, every 50 epochs
            if (epoch % 50 == 0) and (epoch != 0) :
                np.save(output+prefix + '_model.npy',lasagne.layers.get_all_param_values(network))
        log('Saving......')
        # Saving hyperparameters and model, numpy objects are used for compatibility
        np.save(output+prefix + '_model.npy',lasagne.layers.get_all_param_values(network))
        np.save(output+'sharing_model.npy',lasagne.layers.get_all_param_values(network))
    except KeyboardInterrupt:
        # On CTRL+C detection.....
        #log('Saving......')
        np.save(output+prefix + '_model.npy',lasagne.layers.get_all_param_values(network))
        np.save(output+'sharing_model.npy',lasagne.layers.get_all_param_values(network))
    # Remove comments for testing accuracy

    # win_layer.input_var = make_weather(cp,dataset_test)
    # din_layer.input_var = make_disp(cp,dataset_test)
    # prediction = lasagne.layers.get_output(network).argmax(axis=1).eval()
    # print prediction[0:40]
    # print dataset_test[:,1][0:40]
    # acc = np.mean((prediction==dataset_test[:,1]))
    # print prediction.shape
    # log('ACC:  '+str(acc))
    # print pred
    # acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
    #               dtype=theano.config.floatX)

# Loads hyperparameters from previous pretrained experiment and reconstructs the model.
# Used for model archiving or testing.

def init_pretrained(cp, dataset_test, batch):
    output = cp.get('Experiment','output')
    prefix = cp.get('Experiment','prefix')
    # Create "Weather" neural network
    [win_layer, win, weather_net] = init_weather_conv(cp, dataset_test)
    # Create "Dispersion" neural network
    [din_layer, din, disp_net] = init_disp_conv(cp, dataset_test)
    batch_size = int(cp.get('NeuralNetwork', 'batchsize'))
    index = T.lscalar()
    lw_epochs = int(cp.get('NeuralNetwork', 'maxepochs'))
    base_lr = float(cp.get('NeuralNetwork', 'learningrate'))
    # ConcatLayer is used in order to connect the outputs of each network
    concat = network = lasagne.layers.ConcatLayer(
        incomings=(weather_net, disp_net), axis=1)
    # Introduce noise
    network = lasagne.layers.DropoutLayer(incoming=network,
                                          p=float(cp.get('NeuralNetwork', 'corruptionfactor')))
    # Fully connected #1, Shape: close to original
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden0')),
                                        )
    # Fully connected #2, Shape: compressing
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden1')),
                                        )
    # Fully connected #3, Shape: convert to original
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'hidden2')),
                                        )
    # Softmax
    network = lasagne.layers.DenseLayer(incoming=network,
                                        num_units=int(
                                            cp.get('NeuralNetwork', 'stationnum')),
                                        nonlinearity=lasagne.nonlinearities.softmax
                                        )
    log(lasagne.layers.get_output_shape(lasagne.layers.get_all_layers(network)))
    params = np.load(output+prefix+'_model.npy')
    print params.shape
    # Load pretrained parameters
    lasagne.layers.set_all_param_values(network,params)
    # Save model
    model = Model(input_layer=[win_layer,din_layer],encoder_layer=None,decoder_layer=network,network=network)
    model.save(output+prefix+'_model.zip')

    # Remove comments for testing

    # win_layer.input_var = make_weather(cp,dataset_test)
    # din_layer.input_var = make_disp(cp,dataset_test)
    # prediction = lasagne.layers.get_output(network).eval()
    # max_pred = lasagne.layers.get_output(network).argmax(axis=1).eval()
    # print max_pred[0:40]
    # print dataset_test[:,1][0:40]
    # acc = np.mean((max_pred==dataset_test[:,1]))
    # print prediction.shape
    # log('ACC:  '+str(acc))
    # results = []
    # for i in xrange(dataset_test.shape[0]):
    #     origin = dataset_test[i,1]
    #     raw_preds = prediction[i,:]
    #     scores = [(stat,pred) for stat,pred in enumerate(raw_preds)]
    #     scores = sorted(scores, key=lambda x: x[1], reverse=True)
    #     results.append((origin,raw_preds,scores))
    # results = np.asarray(results,dtype=object)
    # np.save(output+prefix+'_test_results_'+str(batch)+'.npy',results)


# Control flow
def main(path, train):
    cp = load_config(path)
    X = load_data(cp, train)
    if train == 'train':
        init(cp, X)
    else:
        init_pretrained(cp, X)

from operator import attrgetter
from argparse import ArgumentParser
if __name__ == "__main__":
    parser = ArgumentParser(description='Training/Testing for VASE')
    # Configuration file path
    parser.add_argument('-i', '--input', required=True, type=str,
                        help='config file')
    # Trainig/testing flag
    parser.add_argument('-t', '--train', required=True, type=str,
                        help='training/testing')
    opts = parser.parse_args()
    getter = attrgetter('input', 'train')
    inp, train = getter(opts)
    main(inp, train)
