#!/usr/bin/env python

"""
Plot an x,y line graph... 
"""

import sys
import time
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Plot error lines given a log file.')
parser.add_argument('file', metavar='f', type=str, nargs='?',
                   help='a log file to process')
parser.add_argument('label', metavar='l', type=str, nargs='?',
                   help='the label lines of interest start with')
parser.add_argument('xcol', metavar='x', type=int, nargs='?', default=4,
                   help='the column number containing the xs')
parser.add_argument('ycol', metavar='y', type=int, nargs='?', default=5,
                   help='the column number containing the ys')

args = parser.parse_args()

def main():
    xs = []
    ys = []
    for l in open(args.file):
        l = l.strip()
        if l.startswith(args.label):
            toks = l.split()
            xs.append(toks[args.xcol - 1])
            ys.append(toks[args.ycol - 1])
    
    plt.plot(xs, ys)
    plt.show()

if __name__ == '__main__':
    main()