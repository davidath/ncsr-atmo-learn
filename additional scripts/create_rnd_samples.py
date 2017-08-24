#!/usr/bin/env python

"""
Create randomly created training samples based on dispersions. The samples are 
written in a file in the form: fk, rnd_sample, weather.
"""

import sys
import os
from datetime import datetime
import numpy as np
import random
from scipy.ndimage.filters import gaussian_filter
from scipy.misc import imresize
from sklearn import preprocessing

NUM_SAMPLES=5
NUM_POINTS=30
RESIZE_DISPERSION = True
TARGET_SIZE = (167, 167)
GFILTER = True
GFILTER_SIGMA = .9
NORMALISE = False

STATIONS = ['ALMARAZ',  #
            'CERNAVODA',#
            'COFRENTES',#
            'DOEL',#
            'EMSLAND',#
            'FORSMARK',#
            'GARONA',
            'GROHNDE',#
            'HEYSHAM',#
            'HINKLEY',#
            'IGNALINA',
            'KHMELNITSKY',#
            'KOZLODUY',#
            'KRSKO',#
            'LOVIISA', #
            'PAKS',#
            'RINGHALS',#
            'SIZEWELL',#
            'SUKRAINE',#
            'VANDELLOS']#

def calc_mm_spec(dims):
    """
    Calculate and return a memmap spec in the form of a list.
    e.g.: [{'numsamples': slice(0, 500)}, {'disp': slice(0,500), 'weath': slice(501,1000)}]
           [----- 1st dim -------------]  [----------------- 2nd dim --------------------]    
    dims: a list of dimensions. Each dimension is a list of tuples (key, size).
    return: a list of dics containing the spec, a tuple for instantiating the memmap
    """
    spec = []
    dimsizes = []
    
    for d in dims:
        offset = 0
        ddict = {}
        for v in d:
            stop = offset + v[1]
            ddict[v[0]] = slice(offset, stop)
            offset = stop
        dimsizes.append(offset)
        spec.append(ddict)
    return tuple(spec), tuple(dimsizes)

def log(s):
    sys.stderr.write('INFO [' + str(datetime.now()) + '] ' + str(s) + '\n')

def normal(sample):
    return sample  ## TODO implement

def main():
    filenames = []
    # filenames.append(('i131', 'i131_v2.npz'))
    filenames.append(('c137','c137_v2.npz'))
    
    # tmp = calc_mm_spec([[('ns',500)], [('a',250), ('b',200), ('c',500)]])
    # print tmp[0]
    # print tmp[1]
    # sys.exit(1)

    for f in filenames:
        log('Loading ' + f[1] + '...')
        data = np.load(f[1])[f[0]]
        log('Done')
        
        # Prepare the memmap to store the tuples
        totnumsamples = len(data) * NUM_SAMPLES
        weather_size = 4096  # 64**2
        if RESIZE_DISPERSION:
            disp_size = TARGET_SIZE[0] * TARGET_SIZE[1]
        else:
            disp_size = 251001  # 501 * 501
        fk_size = 1
        origin_size = 1
        mm_spec, mm_lengths = calc_mm_spec([ [('num_samples', totnumsamples)], 
            [('fk', fk_size), ('origin', origin_size), ('disp', disp_size), 
             ('weath', weather_size)]])
        print mm_spec
        print mm_lengths

        log('Creating memmap...')
        mm = np.memmap('rnd_samples.dat', dtype='float32', mode='w+', shape=mm_lengths)
        log('Done')
        
        for si, sample in enumerate(data):
            # sample[0]: origin, [1]: origin index, [2]: date, [3]: dispersion, [4]: weather
            origin_index = sample[1]
            if RESIZE_DISPERSION:
                disp = imresize(sample[3], TARGET_SIZE, mode='F')
            else:
                disp = sample[3]
            # weather: GHT700 only - 64*64
            weather = sample[4][2][1].reshape(-1)
            nz = np.nonzero(disp)
            for j in range(NUM_SAMPLES):
                idxs = random.sample(range(len(nz[0])), NUM_POINTS)
                newsample = np.zeros(disp.shape)
                for i in idxs:
                    newsample[nz[0][i]][nz[1][i]] = disp[nz[0][i]][nz[1][i]]
                assert len(np.nonzero(newsample)[0]) == NUM_POINTS
            
                if GFILTER:
                    newsample = gaussian_filter(newsample, GFILTER_SIGMA)
                
                if NORMALISE:
                    newsample = normal(newsample)
                
                tuplindex = si * NUM_SAMPLES + j
                mm[tuplindex][mm_spec[1]['fk']] = [si]
                mm[tuplindex][mm_spec[1]['origin']] = [origin_index]
                mm[tuplindex][mm_spec[1]['disp']] = newsample.reshape(-1)
                mm[tuplindex][mm_spec[1]['weath']] = weather
            
            log('Done: ' + str(si + 1) + '/' + str(len(data)))
        
        log('Flushing...')
        del mm
        print mm_spec
        print mm_lengths
        log('Done')


if __name__ == '__main__':
    main()
