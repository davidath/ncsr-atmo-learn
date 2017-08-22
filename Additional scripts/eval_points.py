#!/usr/bin/env python

import sys
import os
import numpy as np
import modeltemplate
from Clustering import Clustering
from datetime import datetime
from dataset_utils import load_single
from disputil import display_array
from Dataset_transformations import Dataset_transformations
from glob import glob
from scipy.misc import imresize
from scipy.stats import entropy
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from scipy.spatial.distance import correlation
from scipy.spatial.distance import cityblock
import operator
from sklearn import preprocessing


"""
Evaluate a series of random dispersion points agains a model and a clustering 
outcome. The series of random test samples are assumed to be stored in a 
numpy memmap (#samples, data), with a sample specification as given in the 
SAMPLE_SPEC variable, and an overall shape provided in the MM_SHAPE variable.
"""

SAMPLE_SPEC = {'origin': slice(1, 2, None), 
               'disp': slice(2, 27891, None),
               'fk': slice(0, 1, None),
               'weath': slice(27891, 31987, None)}
MM_SHAPE = (22400, 31987)
OR_DISP_SIZE = 251001  # 501*501
STATIONS = ['ALMARAZ',  #0
            'CERNAVODA',#1
            'COFRENTES',#2
            'DOEL',     #3
            'EMSLAND',  #4
            'FORSMARK', #5
            'GARONA',   #6
            'GROHNDE',  #7
            'HEYSHAM',  #8
            'HINKLEY',  #9
            'IGNALINA', #10
            'KHMELNITSKY',#11
            'KOZLODUY', #12
            'KRSKO',    #13
            'LOVIISA',  #14
            'PAKS',     #15
            'RINGHALS', #16
            'SIZEWELL', #17
            'SUKRAINE', #18
            'VANDELLOS']#19

CLUSTERS_FILE = 'GHT_700_clusters_shallow.zip'
# CLUSTERS_FILE = 'GHT_700_raw_kmeans.zip'
# CLUSTERS_FILE = 'GHT_700_raw_density.zip'

MODEL_FILE = 'GHT_700_shallow_model_cpu.zip'  # Should be None for kmeans on raw
# MODEL_FILE = None   # Raw

# DISPERSIONS_DIR = 'ght700_shallow_dispersions_desc2'
DISPERSIONS_DIR = 'ght700_shallow_dispersions'
# DISPERSIONS_DIR = 'kmeans_km2_raw_dispersions'
# DISPERSIONS_DIR = 'kmeans_dense_raw_dispersions'

SPECIES = 'c137'
PAD = 0 #1E-13

def log(s, label='INFO', metadata=False):
    if metadata:
        sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' + str(s) + '\n')
    else:
        sys.stdout.write(str(s) + '\n')
    sys.stdout.flush()

def normalize(samples):
    """
    Normalises the given sample(s), assuming that they are in the form
    (features, samples);
    Returns a new np array of the same shape.
    """
    toret = samples
    # print np.min(toret), np.max(toret), np.std(toret)
    for j in range(0, toret.shape[1]):
        mean = toret[:, j].mean()
        toret[:, j] = np.subtract(toret[:, j], mean)
        toret[:, j] = np.divide(toret[:, j], np.sqrt(np.var(toret[:, j]) + 10))
    # print np.min(toret), np.max(toret), np.std(toret)
    return toret

def reconstruct_date(date_str, dot_nc=False):
    if dot_nc:
        date = datetime.strptime(
            date_str.split('.')[0], '%Y-%m-%d_%H:%M:%S')
    else:
        date = datetime.strptime(date_str, '%Y-%m-%d_%H:%M:%S')
    return datetime.strftime(date, '%y-%m-%d-%H')

def main():
    disp_needs_resizing = False
    target_disp_shape = (501, 501)
    target_disp_length = 501 * 501
    
    datafile = sys.argv[1]
    log('Loading clustering...')
    clustering = load_single(CLUSTERS_FILE)
    log('Done.')
    model = None
    if MODEL_FILE is not None:
        log('Loading model...')
        model = load_single(MODEL_FILE)
        log('Done.')
    else:
        log('Model not given, assumming clustering on raw data.')
    
    # Iterate through the samples
    mm = np.memmap(datafile, dtype='float32', mode='r', shape=MM_SHAPE)
    for s_i, sample in enumerate(mm[4000:]):
        origin_i = int(sample[SAMPLE_SPEC['origin']])  # real origin
        disp = np.array(sample[SAMPLE_SPEC['disp']])
        
        disp = preprocessing.maxabs_scale(disp) * 1000  # scale disp [-1..1)
        disp += PAD
        
        if len(disp) < OR_DISP_SIZE:
            disp_needs_resizing = True
            x = int(np.sqrt(len(disp)))
            target_disp_shape = (x, x)
            target_disp_length = target_disp_shape[0] * target_disp_shape[1]
            # log('Target dispersion shape: ' + str(target_disp_shape))
        assert np.sqrt(len(disp)).is_integer()  # sanity check...
        
        weather = np.array(sample[SAMPLE_SPEC['weath']])
        weather = weather.reshape(4096, 1)
        ds = Dataset_transformations(weather, 1, weather.shape)
        ds.normalize()
        ds._items = ds._items.T

        ds_hidden = None
        if MODEL_FILE is not None:
            h = model.get_hidden(ds._items)
            ds_hidden = Dataset_transformations(h, 1, h.shape)
        else:
            ds_hidden = ds  ## unfortunate naming but...
            assert ds_hidden._items.shape == (1, 4096)
        
        # display_array(ds._items.reshape(64, 64))
        # display_array(h[:,:2500].reshape(50, 50))
        
        # get the closest cluster to the current weather
        cl_order = clustering.centroids_distance(ds_hidden)
        cl_cluster = cl_order[0][0]
        cl_score = cl_order[0][1]
        cluster_date = clustering._desc_date[cl_cluster]
        cl_id = reconstruct_date(cluster_date)
        
        scores = []
        scores_euc = []
        scores_cos = []
        for d in glob(DISPERSIONS_DIR + '/' + cl_id + '/*' + SPECIES + '.npy'):
            origin = d[d.rfind('/') + 1:]
            origin = origin[:origin.find('-')]
            
            cl_dispersion = np.load(d)
            if disp_needs_resizing:
                # resize the 501x501 into whatever is needed
                cl_dispersion = imresize(cl_dispersion, target_disp_shape, mode='F')
            # p real, q model
            # display_array(disp.reshape(167,167))
#             display_array(cl_dispersion)
            
            cl_dispersion = preprocessing.maxabs_scale(cl_dispersion) * 1000
            cl_dispersion += PAD
            
            scor = euclidean(cl_dispersion.reshape(target_disp_length), disp)
            # scor = entropy(disp, cl_dispersion.reshape(target_disp_length))
            scores.append((STATIONS.index(origin), origin, scor))
            
            # Calculate cosine distance:
            scor_euc = cosine(cl_dispersion.reshape(target_disp_length),
                           disp)
            scores_euc.append((STATIONS.index(origin), origin, scor_euc))
            
            scor_cos = correlation(cl_dispersion.reshape(target_disp_length), disp)
            scores_cos.append((STATIONS.index(origin), origin, scor_cos))
            
            assert scor != float('Inf')
            assert scor_euc != float('Inf')
            assert scor_cos != float('Inf')

        scores.sort(key=operator.itemgetter(2))
        scores_euc.sort(key=operator.itemgetter(2))
        scores_cos.sort(key=operator.itemgetter(2))
        # print scores
        # print scores_euc
        # print scores_cos
        
        pos = 0
        pos_euc = 0
        pos_cos = 0
        for i in range(0, len(STATIONS)):
            # print 'Origin', STATIONS.index(origin), 'score...', scores[i][0]
            if origin_i == scores[i][0]:
                pos = i + 1
            if origin_i == scores_euc[i][0]:
                pos_euc = i + 1
            if origin_i == scores_cos[i][0]:
                pos_cos = i + 1
            if pos > 0 and pos_euc > 0 and pos_cos > 0: 
                break
        # log(str(origin_i) + '> ' + str(s_i) + ' ' + str(pos) )
        log(str(origin_i) + '\t' + str(pos) + '\t' + str(pos_euc) + '\t' + str(pos_cos))
        

if __name__ == '__main__':
    main()
