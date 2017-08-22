#!/usr/bin/env python

import sys
import os
from datetime import datetime
import numpy as np
from glob import glob

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
STATIONS_SET = set(STATIONS)

def log(s):
    sys.stderr.write('INFO: ' + str(s) + '\n')
    sys.stderr.flush()

def load_data():
    data = np.load('test_rel_data.npz')['all']
    times = np.load('test_rel_data.npz')['times']
    disps = os.listdir('./test_npy/')
    dispersion_times = {} # {time: [files]}
    for i, s in enumerate(disps):
        st = s.find('-') + 1
        l = s.rfind('-')
        datestr = s[st:l]
        d = datetime.strptime(datestr, '%y-%m-%d-%H')
        if d not in dispersion_times:
            dispersion_times[d] = []
        dispersion_times[d].append(s)
    return dispersion_times, data, times

def main():
    log('Loading...')
    dispersiontimes, weathers, times = load_data()
    # log((len(dispersiontimes), weathers.shape, times.shape))
    # print dispersiontimes.keys()
    # sys.exit(1)
    
    # find which weathers (indexes) correspond to each dispersion
    snaps_per_disp = 13
    disp2weather = {}
    sorted_dds = sorted(dispersiontimes.keys())
    for dd in sorted_dds:
        # assume times is sorted
        tind = np.searchsorted(times, dd)
        windexes = (tind, tind + snaps_per_disp - 1) # both indexes inclusive 
        disp2weather[dd] = (windexes, times[tind:tind+snaps_per_disp])
    
    # calculate average weathers for each dispersion date of interest
    avg_weathers = []
    for dd in sorted_dds:
        windexes = disp2weather[dd][0]
        sindex = windexes[0]
        eindex = windexes[1] + 1
        avgw = np.average(weathers[:,sindex:eindex,:,:,:], 1)
        avg_weathers.append(avgw)
    
    # calculate labels:
    c137 = []
    i131 = []
    dispdir = 'test_npy'
    # origin, dispersion, weather
    for df in glob(dispdir + '/*.npy'):
        bname = os.path.basename(df)
        origin = bname[:bname.find('-')]
        if origin not in STATIONS_SET: continue
        target = None 
        species = bname[bname.rfind('-')+1:bname.rfind('.')]
        # print species
        if species == 'i131':
            target = i131
        elif species == 'c137':
            target = c137
        else: 
            log('ERROR: Unknown species: ' + species) # shouldn't reach this
            system.exit(-1)
        datestr = bname[bname.find('-')+1: bname.rfind('-')]
        d = datetime.strptime(datestr, '%y-%m-%d-%H')
        tind = np.searchsorted(sorted_dds, d)
        weather = avg_weathers[tind]
        dispersion = np.load(df)
        target.append((origin, STATIONS.index(origin), d, dispersion, weather))

    c137 = np.array(c137)
    i131 = np.array(i131)
    
    np.savez_compressed('c137_v2', c137=c137)
    np.savez_compressed('i131_v2', i131=i131)
    log('Done.')

    # Weather data in: dd -> avg_weathers[vars, height, x, y]
    # Dispersion data in: dd -> dispersiontimes[dd] (list of dispersion files)
    # Labels: 
    
if __name__ == '__main__':
    main()