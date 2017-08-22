#!/usr/bin/env python

import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import sys

"""
Displays a histogram. Usage hist.py <filename> [col#]. It expects the file
to contain space-separated numbers.
"""

def sisint(s):
    try: 
        int(s)
        return True
    except ValueError:
        return False
    
def main():
    # fname = sys.argv[1]
    colno = 1
    if len(sys.argv) > 1:
        colno = int(sys.argv[1])
    
    ints = []
    count = 0

    for l in sys.stdin:
        l = l.strip()
        l = l.split()
        if len(l) < colno: continue
        l = l[colno-1]

        if sisint(l):
            count += 1
            ints.append(int(l))
    ints = np.array(ints)
    n, bins, patches = plt.hist(ints, bins=range(1,22), normed=1, facecolor='green', alpha=0.75)
    plt.axis([1, 21, 0, 1])
    plt.show()
    print 'Read', count, 'lines.'
    
if __name__ == '__main__':
    main()
