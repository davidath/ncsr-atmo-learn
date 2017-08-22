#!/usr/bin/env python

"""
Create a training set for supervised learning based on convnets.
This presumes a c147.npz-type data file already exists, as it is
created by corr_disp2weather.py.
"""

import sys
import os
import numpy as np
from scipy.misc import imresize
from datetime import datetime
from disputil import display_array
from numpy.random import rand
from sklearn.preprocessing import maxabs_scale
from sklearn.preprocessing import scale
from scipy.ndimage.filters import gaussian_filter

DATAFILE = 'c137_v2.npz'
SPECIES = 'c137'
NUM_OF_SAMPLES = 5  # number of random-noise samples per original sample
REP_NUM = 5
TARGET_DISP_DIMS = (167, 167)
WEATHER_DIST = 0.01  # the weather distortion percentage
# DISPER_DIST = 0.1   # the dispersion distortion percentage
NUM_OF_POINTS = 30

def log(s, label='INFO', metadata=True):
    if metadata:
        sys.stdout.write(label + ' [' + str(datetime.now()) + '] ' +
                         str(s) + '\n')
    else:
        sys.stdout.write(str(s) + '\n')
    sys.stdout.flush()


def main():
    # load the base training data
    log('Loading ' + DATAFILE + '...')
    c137 = np.load(DATAFILE)[SPECIES]
    log('Done.')
    for rep in xrange(REP_NUM):
        training_set = []

        for si, origsmpl in enumerate(c137):
            dispersion = origsmpl[3]  #.reshape((501, 501))
            newdispersion = imresize(dispersion, TARGET_DISP_DIMS, mode='F')
            # newdispersion = newdispersion.reshape(
            #                             TARGET_DISP_DIMS[0]*TARGET_DISP_DIMS[1])
            weather = origsmpl[4]
            origin = origsmpl[0]
            origin_index = origsmpl[1]
            thedate = origsmpl[2]
            w_shape = weather.shape
            U = weather[0].flatten()
            V = weather[1].flatten()
            GHT = weather[2].flatten()
            weather = np.concatenate((maxabs_scale(U),maxabs_scale(V),maxabs_scale(GHT)))
            weather = weather.reshape(w_shape)
            print weather.shape
            newdispersion = maxabs_scale(newdispersion)
            for i in range(NUM_OF_SAMPLES):
                rw = ((rand(3, 3, 64, 64) - 0.5) * 2.0) * WEATHER_DIST
                # rd = (rand(167, 167) - 0.5) * 2.0 * DISPER_DIST
                w = weather + rw * weather
                d = newdispersion
                nzidx_x,nzidx_y = d.nonzero()
                # print np.min(w[0]),np.max(w[0])
                # print np.min(w[1]),np.max(w[1])
                # print np.min(w[2]),np.max(w[2])
                # d[nzidx] = d[nzidx] + rd[nzidx] * d[nzidx]
                ad = np.zeros(shape=(TARGET_DISP_DIMS))
                points = np.random.choice(len(nzidx_x), NUM_OF_POINTS, replace=False)
                ad[nzidx_x[points],nzidx_y[points]] = d[nzidx_x[points],nzidx_y[points]]
                ad = gaussian_filter(ad,0.9)
                # display_array(weather[2][1])
                # display_array(w[2][1])
                # display_array(ad)
                # display_array(d)
            #     break
            # break
                tuple = [origin, origin_index, thedate, ad, w]

                training_set.append(tuple)

            log('Finished sample ' + str(si) + '/' + str(len(c137)))
        training_set = np.array(training_set)
        log('Saving...')
        np.save('supervised_training_c137_'+str(rep), training_set)
        log('Done.')

if __name__ == '__main__':
    main()
