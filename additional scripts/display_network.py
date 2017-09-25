from scipy.misc import imresize

def plot_border(img):
    newimg = np.pad(img,((1,1),(1,1)),'constant',constant_values=(0))
    max_x = newimg.shape[0]-1
    max_y = newimg.shape[1]-1
    newimg[0,:] = np.ones(shape=(newimg[0,:].shape)) * np.max(newimg)
    newimg[max_x,:] = np.ones(shape=(newimg[max_x,:].shape)) * np.max(newimg)
    newimg[:,0] = np.ones(shape=(newimg[:,0].shape)) * np.max(newimg)
    newimg[:,max_y] = np.ones(shape=(newimg[:,max_y].shape)) * np.max(newimg)
    return newimg

def plot_row(images, spacing, CMAP=None):
    images = np.asarray([plot_border(image) for image in images])
    x = images.shape[1]
    y = images.shape[2]
    canvas_y = y*len(images)+spacing*(len(images)-1)
    canvas = np.zeros(shape=(x,canvas_y))
    print canvas.shape
    idx = range(0,canvas.shape[1],y+spacing)
    for pos,i in enumerate(idx):
        canvas[:,i:i+y] += images[pos]
    if CMAP:
      plt.matshow(canvas, cmap=CMAP)
    else:
      plt.matshow(canvas)
    plt.axis('off')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    plt.show()

def plot_grid(images, gridx, gridy, xspace, yspace, CMAP=None, row_space=None, out=None):
    images = np.asarray([plot_border(image) for image in images])
    x = images.shape[1]
    y = images.shape[2]
    canvas_x = x*gridx+xspace*(gridx-1)
    canvas_y = y*gridy+yspace*(gridy-1)
    canvas = np.zeros(shape=(canvas_x,canvas_y))
    print canvas.shape
    print y*gridy+yspace
    print y,gridy,yspace
    idx_y = range(0,canvas.shape[1],y+yspace)
    idx_x = range(0,canvas.shape[0],x+xspace)
    print idx_y
    print idx_x
    for posi,i in enumerate(idx_x):
        for posj,j in enumerate(idx_y):
            canvas[i:i+y,j:j+y] += images[posi+posj]
    if row_space:
        zeros = np.zeros(shape=(row_space,canvas.shape[1]))
        canvas = np.append(zeros,canvas)
        canvas = canvas.reshape(row_space+canvas_x,canvas_y)
    if CMAP:
      plt.matshow(canvas, cmap=CMAP)
    else:
      plt.matshow(canvas)
    plt.axis('off')
    plt.xticks(np.array([]))
    plt.yticks(np.array([]))
    if out:
        fig.savefig(out+".png")
    else:
        plt.show()
