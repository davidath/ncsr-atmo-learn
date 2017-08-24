import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

def display_array(a):
    # plt.imshow(a, cmap=matplotlib.cm.gray, interpolation='nearest')
    plt.imshow(a, interpolation='nearest')
    plt.show()

def displayz(a, x, y, startind=0, sizex=12, sizey=12):
    fig = plt.figure(figsize=(sizex, sizey))
    fig.subplots_adjust(hspace=0.01, wspace=0.05)
    for i in range(x * y):
        sub = fig.add_subplot(x, y, i+1)
        sub.set_axis_off()
        sub.imshow(a[startind+i,:,:], interpolation='nearest')
