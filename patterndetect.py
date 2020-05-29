import numpy as np

def initialisedataset():
    """This creates the X & Y dataset that we'll use to train the CNN on whether the task is pattern related or not.
    X - 2 channel image, first channel is the input picture. Second channel is the output picutre
    Y - List of booleans stating whether the problem is pattern related or not (this was manually labelled)"""
    import initialdefs
    import math

    task_file, alltasks = initialdefs.starup()

    X = np.array([[np.zeros([32, 32]), np.zeros([32, 32])]])
    Y = [0]

    # make prelim Y's - labels for which problems are patterns. Prelim because we'll make more samples from each problem
    # so we'll only use these to inform us what label we should use
    Yprelim = [0] * 400

    # from manually going through and seeing what tasks were filling in repeating patterns / mosaics
    for i in [16, 60, 73, 109, 241, 286, 304, 312, 350, 399]:
        Yprelim[i] = 1

    for taskno in range(len(alltasks)):
        print(taskno)
        task = alltasks[taskno]
        train = task['train']

        # check the input & output are the same size: if not, don't use (too different, would cause too many problems)
        check = train[0]
        checkinput = np.array(check['input'])
        checkoutput = np.array(check['output'])

        # if they are the same, we can use as sample for the model.
        if checkoutput.shape == checkinput.shape:
            for trainno in range(len(train)):
                # dim0: samples dim1: channels (2: input, out), dim3: x dim4: y
                imagepair = train[trainno]
                imageinput = imagepair['input']
                imageoutput = imagepair['output']
                sz0l = math.floor((32 - np.size(imageinput, 0))/2)  # padding for the left of dimension 0
                sz0r = math.ceil((32 - np.size(imageinput, 0))/2)  # padding for the right of dimension 0
                sz1l = math.floor((32 - np.size(imageinput, 1))/2)  # padding for the left of dimension 1
                sz1r = math.ceil((32 - np.size(imageinput, 1))/2)  # padding for the right of dimension 1
                ippad = np.pad(imageinput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))
                oppad = np.pad(imageoutput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))

                newsample = np.array([[ippad, oppad]])

                X = np.concatenate((X, newsample), axis=0)
                Y.append(Yprelim[taskno])

                # create more images from the rotated versions
                for i in range(3):
                    ippad = np.rot90(ippad)
                    oppad = np.rot90(oppad)

                    newsample = np.array([[ippad, oppad]])

                    X = np.concatenate((X, newsample), axis=0)
                    Y.append(Yprelim[taskno])

                # create more images from the transposed & rotated versions
                ippad = ippad.T
                oppad = oppad.T

                newsample = np.array([[ippad, oppad]])

                X = np.concatenate((X, newsample), axis=0)
                Y.append(Yprelim[taskno])

                for i in range(3):
                    ippad = np.rot90(ippad)
                    oppad = np.rot90(oppad)

                    newsample = np.array([[ippad, oppad]])

                    X = np.concatenate((X, newsample), axis=0)
                    Y.append(Yprelim[taskno])

    X = np.delete(X, 0, axis=0)
    Y.__delitem__(0)

    #  make channel the last dim
    X = np.moveaxis(X, 1, -1)

    return X, Y


def modelbuildtrain(X, Y):
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten

    buildcomplex = True

    if buildcomplex:
        #  build model - complex
        model = Sequential()
        model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1),
                         activation='relu',
                         input_shape=(32, 32, 2)))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
    else:
        # Build the model - simple
        model = Sequential([
            Conv2D(8, 3, input_shape=(32, 32, 2)),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(1, activation='sigmoid'),
        ])

    model.compile(
        'adam',
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )

    # make data
    Xtrain = X[0:5500, :, :, :]
    Xtest = X[5501:, :, :, :]
    Ytrain = Y[0:5500]
    Ytest = Y[5501:]

    # Train the model.
    model.fit(
        Xtrain,
        Ytrain,
        epochs=3,
        validation_data=(Xtest, Ytest)
    )

    return model


def preparetaskformodel(task):
    """takes one image pair and transforms it into the correct format to be presented to model
    imagepair --    an image pair example from a task
    newsample --    a sample to be presented to the model
    """
    import math

    tasktrain = task['train']
    imagepair = tasktrain[0]

    imageinput = np.array(imagepair['input'])
    imageoutput = np.array(imagepair['output'])

    #  check that these are the same size
    if not imageinput.shape == imageoutput.shape:
        #  print('Input and output not the same size so we know its not pattern')
        return 0

    sz0l = math.floor((32 - np.size(imageinput, 0)) / 2)  # padding for the left of dimension 0
    sz0r = math.ceil((32 - np.size(imageinput, 0)) / 2)  # padding for the right of dimension 0
    sz1l = math.floor((32 - np.size(imageinput, 1)) / 2)  # padding for the left of dimension 1
    sz1r = math.ceil((32 - np.size(imageinput, 1)) / 2)  # padding for the right of dimension 1
    ippad = np.pad(imageinput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))
    oppad = np.pad(imageoutput, ((sz0l, sz0r), (sz1l, sz1r)), constant_values=(0, 0))

    newsample = np.array([[ippad, oppad]])

    #  make channel the last dim
    newsample = np.moveaxis(newsample, 1, -1)

    return newsample


# def modelpresrecall(model, X, Y):
#     """gives metrics on the precision and recall of our model designed to label mosaic/symmetry tasks
#     """
#     from sklearn.metrics import classification_report
#
#     #  get precision & recall
#     y_pred = model.predict(X, batch_size=64, verbose=1)
#     y_pred_bool = np.argmax(y_pred, axis=1)
#
#     print(classification_report(Y, y_pred_bool))


def makepredictions(task, model):
    newsample = preparetaskformodel(task)

    if newsample is 0:
        return False
    else:
        prediction = float(model.predict(newsample))
        return prediction > 0.5


def checktranssymmetry(image, repunit):
    """Once a translational repeating unit has been created, this checks whether the repeating unit can be used
    to describe the whole image
    image   --      output image
    repunit --      repeating unit
    return  --      boolean whether repunit creates full pattern or not
    """
    # raster-scan in any possible increments for repeating unit
    for rasterrow in range(1, np.size(repunit, axis=0)):
        for rastercol in range(1, np.size(repunit, axis=1)):
            newrepunit = image[0:rasterrow, 0:rastercol]

            if checktranssymmetry(image, newrepunit):
                #  found it!
                foundsol = 1
                return foundsol, newrepunit

    return 1


def findtranssymmetries(imageoutput):
    """There may be a repeating unit which is translated (raster scanned) across the image. This finds that symmetry
    task    --      full task
    return:
    testout --      output for the test pattern
    cache   --      parameters for how the task was solved
    """

    foundsol = 0

    # create a repeating pattern
    for reprow in range(2, np.size(imageoutput, axis=0) / 2):
        for repcol in range(2, np.size(imageoutput, axis=1) / 2):
            newrepunit = imageoutput[0:reprow, 0:repcol]

            if checktranssymmetry(imageoutput, newrepunit):
                #  found it!
                foundsol = 1
                return foundsol, newrepunit

    return foundsol, newrepunit


def findrotsymmetries(imageoutput):
        """Othe type of possible symmetry is rotational symmetry. This finds any rotational symmetry
    task    --      full task
    return:
    testout --      output for the test pattern
    cache   --      parameters for how the task was solved
    """


def findsymmetries(task):
    """Once a task has been ascertained as a pattern task, this is how to solve it
    task    --      full task
    return:
    testout --      output for the test pattern
    cache   --      parameters for how the task was solved
    """

    import numpy as np

    tasktrain = task['train']
    imagepair = tasktrain[0]

    imageinput = np.array(imagepair['input'])
    imageoutput = np.array(imagepair['output'])

    # find translational symmetries
    foundsol, newrepunit = findtranssymmetries(imageoutput)

    #  find rotational symmetries if no translational symmetries are present
    if foundsol is not 1:
        foundsol, newrepunit = findrotsymmetries(imageoutput)

    #  if a pattern has been found, see if it's the same for the others in the set


#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#  ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ FROM KAGGLE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from skimage import measure
from matplotlib import colors
from numpy.lib.stride_tricks import as_strided


def in_out_diff(t_in, t_out):
    x_in, y_in = t_in.shape
    x_out, y_out = t_out.shape
    diff = np.zeros((max(x_in, x_out), max(y_in, y_out)))
    diff[:x_in, :y_in] -= t_in
    diff[:x_out, :y_out] += t_out
    return diff


def check_symmetric(a):
    try:
        sym = 1
        if np.array_equal(a, a.T):
            sym *= 2  # Check main diagonal symmetric (top left to bottom right)
        if np.array_equal(a, np.flip(a).T):
            sym *= 3  # Check antidiagonal symmetric (top right to bottom left)
        if np.array_equal(a, np.flipud(a)):
            sym *= 5  # Check horizontal symmetric of array
        if np.array_equal(a, np.fliplr(a)):
            sym *= 7  # Check vertical symmetric of array
        return sym
    except:
        return 0


def bbox(a):
    try:
        r = np.any(a, axis=1)
        c = np.any(a, axis=0)
        rmin, rmax = np.where(r)[0][[0, -1]]
        cmin, cmax = np.where(c)[0][[0, -1]]
        return rmin, rmax, cmin, cmax
    except:
        return 0,a.shape[0],0,a.shape[1]


def cmask(t_in):
    cmin = 999
    cm = 0
    for c in range(10):
        t = t_in.copy().astype('int8')
        t[t==c],t[t>0],t[t<0]=-1,0,1
        b = bbox(t)
        a = (b[1]-b[0])*(b[3]-b[2])
        s = (t[b[0]:b[1],b[2]:b[3]]).sum()
        if a>2 and a<cmin and s==a:
            cmin=a
            cm=c
    return cm


def mask_rect(a):
    r,c = a.shape
    m = a.copy().astype('uint8')
    for i in range(r-1):
        for j in range(c-1):
            if m[i,j]==m[i+1,j]==m[i,j+1]==m[i+1,j+1]>=1:m[i,j]=2
            if m[i,j]==m[i+1,j]==1 and m[i,j-1]==2:m[i,j]=2
            if m[i,j]==m[i,j+1]==1 and m[i-1,j]==2:m[i,j]=2
            if m[i,j]==1 and m[i-1,j]==m[i,j-1]==2:m[i,j]=2
    m[m==1]=0
    return (m==2)


def crop_min(t_in):
    try:
        b = np.bincount(t_in.flatten(),minlength=10)
        c = int(np.where(b==np.min(b[np.nonzero(b)]))[0])
        coords = np.argwhere(t_in==c)
        x_min, y_min = coords.min(axis=0)
        x_max, y_max = coords.max(axis=0)
        return t_in[x_min:x_max+1, y_min:y_max+1]
    except:
        return t_in


def call_pred_train(t_in, t_out, pred_func):
    import inspect

    feat = {}
    feat['s_out'] = t_out.shape
    if t_out.shape==t_in.shape:
        diff = in_out_diff(t_in,t_out)
        feat['diff'] = diff
        feat['cm'] = t_in[diff!=0].max()
    else:
        feat['diff'] = (t_in.shape[0]-t_out.shape[0],t_in.shape[1]-t_out.shape[1])
        feat['cm'] = cmask(t_in)
    feat['sym'] = check_symmetric(t_out)
    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]])
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    feat['sizeok'] = len(t_out)==len(t_pred)
    t_pred = np.resize(t_pred,t_out.shape)
    acc = (t_pred==t_out).sum()/t_out.size
    return t_pred, feat, acc


def call_pred_test(t_in, pred_func, feat):
    import inspect

    args = inspect.getargspec(pred_func).args
    if len(args)==1:
        return pred_func(t_in)
    elif len(args)==2:
        t_pred = pred_func(t_in,feat[args[1]])
    elif len(args)==3:
        t_pred = pred_func(t_in,feat[args[1]],feat[args[2]])
    return t_pred


# from: https://www.kaggle.com/nagiss/manual-coding-for-the-first-10-tasks
def get_data(task_filename):
    import json
    with open(task_filename, 'r') as f:
        task = json.load(f)
    return task

# from: https://www.kaggle.com/boliu0/visualizing-all-task-pairs-with-gridlines
cmap = colors.ListedColormap(
    ['#000000', '#0074D9','#FF4136','#2ECC40','#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

num2color = ["black", "blue", "red", "green", "yellow", "gray", "magenta", "orange", "sky", "brown"]
color2num = {c: n for n, c in enumerate(num2color)}


def plot_one(ax, input_matrix, title_text):
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True,which='both',color='lightgrey', linewidth=0.5)
    ax.set_yticks([x-0.5 for x in range(1+len(input_matrix))])
    ax.set_xticks([x-0.5 for x in range(1+len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title_text)


def check_p(task, pred_func):
    import matplotlib.pyplot as plt

    n = len(task["train"]) + len(task["test"])
    fig, axs = plt.subplots(3, n, figsize=(4*n,12), dpi=50)
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    fnum = 0
    t_acc = 0
    t_pred_test_list = []
    for i, t in enumerate(task["train"]):
        t_in, t_out = np.array(t["input"]).astype('uint8'), np.array(t["output"]).astype('uint8')
        t_pred, feat, acc = call_pred_train(t_in, t_out, pred_func)
        plot_one(axs[0,fnum],t_in,f'train-{i} input')
        plot_one(axs[1,fnum],t_out,f'train-{i} output')
        plot_one(axs[2,fnum],t_pred,f'train-{i} pred')
        t_acc+=acc
        fnum += 1
    for i, t in enumerate(task["test"]):
        # removed t_out as there should be no output in the test
        t_in= np.array(t["input"]).astype('uint8')
        t_pred_test = call_pred_test(t_in, pred_func, feat)
        plot_one(axs[0,fnum],t_in,f'test-{i} input')
        #plot_one(axs[1,fnum],t_out,f'test-{i} output')
        plot_one(axs[2,fnum],t_pred_test,f'test-{i} pred')
        t_pred = np.resize(t_pred,t_in.shape) # assume same shape. used to be: t_pred = np.resize(t_pred,t_out.shape)
        # if len(t_out)==1:
        #     acc = int(t_pred==t_out)
        # else:
        #     acc = (t_pred==t_out).sum()/t_out.size
        # t_acc += acc
        # fnum += 1
        t_pred_test_list.append(t_pred_test)
    # plt.show()
    return t_acc/fnum, t_pred_test_list


def get_tile(img ,mask):
    try:
        m,n = img.shape
        a = img.copy().astype('int8')
        a[mask] = -1
        r=c=0
        for x in range(n):
            if np.count_nonzero(a[0:m,x]<0):continue
            for r in range(2,m):
                if 2*r<m and (a[0:r,x]==a[r:2*r,x]).all():break
            if r<m:break
            else: r=0
        for y in range(m):
            if np.count_nonzero(a[y,0:n]<0):continue
            for c in range(2,n):
                if 2*c<n and (a[y,0:c]==a[y,c:2*c]).all():break
            if c<n:break
            else: c=0
        if c>0:
            for x in range(n-c):
                if np.count_nonzero(a[:,x]<0)==0:
                    a[:,x+c]=a[:,x]
                elif np.count_nonzero(a[:,x+c]<0)==0:
                    a[:,x]=a[:,x+c]
        if r>0:
            for y in range(m-r):
                if np.count_nonzero(a[y,:]<0)==0:
                    a[y+r,:]=a[y,:]
                elif np.count_nonzero(a[y+r,:]<0)==0:
                    a[y,:]=a[y+r,:]
        return a[r:2*r,c:2*c]
    except:
        return a[0:1,0:1]


def patch_image(t_in,s_out,cm=0):
    try:
        t = t_in.copy()
        ty,tx=t.shape
        if cm>0:
            m = mask_rect(t==cm)
        else:
            m = (t==cm)
        tile = get_tile(t ,m)
        if tile.size>2 and s_out==t.shape:
            rt = np.tile(tile,(1+ty//tile.shape[0],1+tx//tile.shape[1]))[0:ty,0:tx]
            if (rt[~m]==t[~m]).all():
                return rt
        for i in range(6):
            m = (t==cm)
            t -= cm
            if tx==ty:
                a = np.maximum(t,t.T)
                if (a[~m]==t[~m]).all():t=a.copy()
                a = np.maximum(t,np.flip(t).T)
                if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.flipud(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            a = np.maximum(t,np.fliplr(t))
            if (a[~m]==t[~m]).all():t=a.copy()
            t += cm
            m = (t==cm)
            lms = measure.label(m.astype('uint8'))
            for l in range(1,lms.max()+1):
                lm = np.argwhere(lms==l)
                lm = np.argwhere(lms==l)
                x_min = max(0,lm[:,1].min()-1)
                x_max = min(lm[:,1].max()+2,t.shape[0])
                y_min = max(0,lm[:,0].min()-1)
                y_max = min(lm[:,0].max()+2,t.shape[1])
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                if i==1:
                    sy//=2
                    y_max=y_min+sx
                gap = t[y_min:y_max,x_min:x_max]
                sy,sx=gap.shape
                allst = as_strided(t, shape=(ty,tx,sy,sx),strides=2*t.strides)
                allst = allst.reshape(-1,sy,sx)
                allst = np.array([a for a in allst if np.count_nonzero(a==cm)==0])
                gm = (gap!=cm)
                for a in allst:
                    if sx==sy:
                        fpd = a.T
                        fad = np.flip(a).T
                        if i==1:gm[sy-1,0]=gm[0,sx-1]=False
                        if (fpd[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fpd)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                        if i==1:gm[0,0]=gm[sy-1,sx-1]=False
                        if (fad[gm]==gap[gm]).all():
                            gm = (gap!=cm)
                            np.putmask(gap,~gm,fad)
                            t[y_min:y_max,x_min:x_max] = gap
                            break
                    fud = np.flipud(a)
                    flr = np.fliplr(a)
                    if i==1:gm[sy-1,0]=gm[0,sx-1]=gm[0,0]=gm[sy-1,sx-1]=False
                    if (a[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,a)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (fud[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,fud)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
                    elif (flr[gm]==gap[gm]).all():
                        gm = (gap!=cm)
                        np.putmask(gap,~gm,flr)
                        t[y_min:y_max,x_min:x_max] = gap
                        break
        if s_out==t.shape:
            return t
        else:
            m = (t_in==cm)
            return np.resize(t[m],crop_min(m).shape)
    except:
        return np.resize(t_in, s_out)