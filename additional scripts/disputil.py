import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.gridspec as gridspec

def display_array(a):
    # plt.imshow(a, cmap=matplotlib.cm.gray, interpolation='nearest')
    plt.imshow(a, interpolation='nearest')
    plt.show()

def displayz(a, x, y, startind=0, sizex=12, sizey=12, CMAP=None):
    fig = plt.figure(figsize=(sizex, sizey))
    fig.subplots_adjust(hspace=0.01, wspace=0.05)
    for i in range(x * y):
        sub = fig.add_subplot(x, y, i+1)
        # sub.imshow(a[startind+i,:,:], interpolation='nearest')
        if CMAP:
            sub.imshow(a[startind+i,:,:], cmap=CMAP, interpolation='nearest')
        else:
            sub.imshow(a[startind+i,:,:], interpolation='nearest')
    plt.show()

def grid_displayz(a, x, y, startind=0, sizex=12, sizey=12, CMAP=None,out="figure"):
    fig = plt.figure(figsize=(sizex, sizey),frameon=True)
    gs = gridspec.GridSpec(x, y,
         wspace=0.0, hspace=0.0)
    for i in range(x * y):
        sub = plt.subplot(gs[i])
        plt.axis('off')
        sub.set_xticklabels([])
        sub.set_adjustable('box-forced')
        sub.set_yticklabels([])
        # sub.imshow(a[startind+i,:,:], interpolation='nearest')
        if CMAP:
            sub.imshow(a[startind+i,:,:], cmap=CMAP, interpolation='nearest')
        else:
            sub.imshow(a[startind+i,:,:], interpolation='nearest')
    # plt.show()
    fig.savefig(out+".png", bbox_inches='tight', pad_inches=0)
