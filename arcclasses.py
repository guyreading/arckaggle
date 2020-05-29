import numpy as np
import cv2
import scipy.ndimage
from copy import deepcopy


class SameColourObject:
    width = None
    height = None
    colour = None
    elementarr = None
    distrelotherobjs = None
    positionabsx = None
    positionabsy = None
    holecount = 0
    holes = None

    def findholes(self, obj):
        filledobj = scipy.ndimage.morphology.binary_fill_holes(obj).astype(int)
        holes = filledobj - obj
        holes = np.uint8(holes)

        if np.count_nonzero(holes) > 0:
            #  labels is the shape of the hole. Potentially could do something with this later
            retval, labels = cv2.connectedComponents(holes)
            self.holecount = retval - 1

    def __init__(self, labels, specificcolour, keepasis=False):
        if keepasis:
            validrows = [True] * labels.shape[0]
            validcols = [True] * labels.shape[1]
            firstcol = 0
            firstrow = 0
        else:
            # cut away where object isn't in the image
            validcols = labels.sum(axis=0) > 0  # logical index col
            validrows = labels.sum(axis=1) > 0  # logical index row
            firstcol = min(np.where(validcols)[0])
            firstrow = min(np.where(validrows)[0])
            lastcol = max(np.where(validcols)[0])
            lastrow = max(np.where(validrows)[0])
            validcols[firstcol:lastcol] = True
            validrows[firstrow:lastrow] = True

        # assign the top left corner
        self.positionabsx = int(firstcol)
        self.positionabsy = int(firstrow)

        self.elementarr = labels[np.ix_(validrows, validcols)]

        self.colour = specificcolour

        self.height = np.size(self.elementarr, axis=0)
        self.width = np.size(self.elementarr, axis=1)

        self.findholes(self.elementarr)


class MultiColourObject:
    width = None
    height = None
    elementarr = None
    distrelotherobjs = None
    positionabsx = None
    positionabsy = None
    samecolourobjs = []
    multicolourobjs = []

    def __init__(self, labels):
        """ takes a full-sized image, finds the object in the image, finds the top
        left hand corner of the bounding box around that object within the image and
        saves that to positionAbs, then places the boxed obj into elementarr. 
        """

        # cut away where object isn't in the image
        validcols = labels.sum(axis=0) > 0  # logical index col
        validrows = labels.sum(axis=1) > 0  # logical index row
        firstcol = min(np.where(validcols)[0])
        firstrow = min(np.where(validrows)[0])
        lastcol = max(np.where(validcols)[0])
        lastrow = max(np.where(validrows)[0])
        validcols[firstcol:lastcol] = True
        validrows[firstrow:lastrow] = True

        # assign the top left corner
        self.positionabsx = firstrow
        self.positionabsy = firstcol

        self.elementarr = labels[np.ix_(validrows, validcols)]

        self.height = np.size(self.elementarr, axis=0)
        self.width = np.size(self.elementarr, axis=1)


class SingleImagePair:
    fullinputimg = None
    fulloutputimg = None
    fullpredimg = None
    inputsamecolourobjs = []
    predoutputsamecolobjs = []  # predicted same colour objects
    predoutputmulticolobjs = []
    predoutputcanvas = None  # output canvas
    outputsamecolourobjs = []
    backgroundcol = None

    def gridrefactorobjs(self):
        """looks for periodicity and shape similarity. If something there, refactor all objs so they conform
        with this pattern
        """
        objwidths, objheights, shortestx, shortesty = [], [], [], []
        furtheestleft, furthestup, shortestxt2, shortestyt2 = 100, 100, 100, 100
        for objno, obj1 in enumerate(self.predoutputsamecolobjs):
            if len(self.predoutputsamecolobjs[objno+1:]) != 0:
                for obj2 in self.predoutputsamecolobjs[objno+1:]:
                    shortestxt1 = obj2.positionabsx - obj1.positionabsx
                    shortestyt1 = obj2.positionabsy - obj1.positionabsy

                    if (shortestxt1 > 0) & (shortestxt1 < shortestxt2) & (obj2.positionabsy == obj1.positionabsy):
                        shortestxt2 = shortestxt1

                    if (shortestyt1 > 0) & (shortestyt1 < shortestyt2) & (obj2.positionabsx == obj1.positionabsx):
                        shortestyt2 = shortestyt1

                shortestx = shortestx + [shortestxt2]
                shortesty = shortesty + [shortestyt2]
            objwidths = objwidths + [obj1.elementarr.shape[1]]
            objheights = objheights + [obj1.elementarr.shape[0]]

            if obj1.positionabsx < furtheestleft:
                furtheestleft = obj1.positionabsx
                topleftx = obj1.positionabsx
                toplefty = obj1.positionabsy

            if obj1.positionabsy < furthestup:
                furthestup = obj1.positionabsy
                toplefty = obj1.positionabsy

        mostfreqwidth = max(set(objwidths), key=objwidths.count)
        mostfreqheight = max(set(objheights), key=objheights.count)
        mostfreqx = max(set(shortestx), key=shortestx.count)
        mostfreqy = max(set(shortesty), key=shortesty.count)

        # sense-check at this point
        if (mostfreqwidth >= mostfreqx) or (mostfreqheight >= mostfreqy):
            self.gridapplied = 0
            return

        # use these numbers to set your grid & rep obj size. If you can account for all pixels in each obj: good.
        # start at top left obj
        bwfullimg = (self.fullpredimg != self.backgroundcol) * 1
        pixelscounted = 0
        xpos = topleftx
        ypos = toplefty
        newobjrefactor = []
        rowsize = 0
        objlist = []
        counter = 0
        outputcanvas = np.zeros([bwfullimg.shape[0], bwfullimg.shape[1]])

        while (ypos + mostfreqheight) <= bwfullimg.shape[0]:
            while (xpos + mostfreqwidth) <= bwfullimg.shape[1]:
                bwarr1 = bwfullimg[ypos:ypos+mostfreqheight, xpos:xpos+mostfreqwidth]  # bw array for counting obj pixels
                bwarr = deepcopy(outputcanvas)
                bwarr[ypos:ypos+mostfreqheight, xpos:xpos+mostfreqwidth] = bwarr1
                colarr = self.fullpredimg[ypos:ypos+mostfreqheight, xpos:xpos+mostfreqwidth]  # colarr for making newobj
                pixelscounted = pixelscounted + bwarr.sum()  # count the pixels

                if len(np.delete(np.unique(colarr), np.where(np.unique(colarr) == 0))) == 1:  # rem backcol & chek 1 col left
                    specificcolour = np.delete(np.unique(colarr), np.where(np.unique(colarr) == 0))[0]
                    newobj = SameColourObject(bwarr, specificcolour)
                    newobjrefactor = newobjrefactor + [newobj]
                    objlist = objlist + [counter]
                else:
                    objlist = objlist + [None]
                xpos = xpos + mostfreqx
                counter += 1
            rowsize += 1
            xpos = topleftx
            ypos = ypos + mostfreqy

        objgrid = np.array(objlist).reshape([rowsize, int(counter/rowsize)])

        if pixelscounted == bwfullimg.sum():  # if all objs are accounted for, re-factor objs into grid format
            self.predoutputsamecolobjs = newobjrefactor
            self.gridapplied = 1
            # self.objgrid = objgrid
            print('refactored by seperating by obj grid')
        else:
            self.gridapplied = 0

    def findscobjects(self, side, forgroundcols, keepasis=False):
        """Find same colour objects in either the input or output and place that into the image
        """

        for specificcolour in forgroundcols:
            # process so we can find connected components (individual obs) in image
            if side == 'output':
                cvimg = self.fulloutputimg == specificcolour
            elif side == 'input':
                cvimg = self.fullinputimg == specificcolour

            cvimg = cvimg * 1
            cvimg = np.uint8(cvimg)

            # find individual objects
            retval, labels = cv2.connectedComponents(cvimg)
            for objs in range(1, retval):
                newobj = SameColourObject((labels == objs) * 1, specificcolour, keepasis=keepasis)
                if side == 'output':
                    self.outputsamecolourobjs = self.outputsamecolourobjs + [newobj]
                elif side == 'input':
                    self.inputsamecolourobjs = self.inputsamecolourobjs + [newobj]

            self.predoutputsamecolobjs = deepcopy(self.inputsamecolourobjs)

    def findbackground(self):
        countforcols = []

        # rule 1: the colour needs to be shared in both the input and the output
        uniquesin = np.unique(self.fullinputimg)
        uniquesout = np.unique(self.fulloutputimg)
        uniques = list(set(uniquesin) & set(uniquesout))

        if 0 in uniques:  # just make back background: 99% of the time it is (bad but need to write rest of code!!)
            self.backgroundcol = 0
        elif not uniques:  # empty list (how can this ever happen?! 0 - black - is a col. So...
            self.backgroundcol = None
        else:
            # rule 2: the colour is the dominent colour of the input
            for specificcolour in uniques:
                countforcols.append(np.count_nonzero(self.fullinputimg == specificcolour))

            self.backgroundcol = uniques[countforcols.index(max(countforcols))]

    def extraobjattrs(self):
        # make a list of x co-ords & y co-ords
        xstartcoords = []
        ystartcoords = []

        for obj in self.inputsamecolourobjs:
            xstartcoords = xstartcoords + [obj.positionabsx]
            ystartcoords = ystartcoords + [obj.positionabsy]

        xstartcoords.sort()
        ystartcoords.sort()

        for obj in self.inputsamecolourobjs:
            obj.xorder = next(i for i, x in enumerate(xstartcoords) if x == obj.positionabsx)
            obj.yorder = next(i for i, y in enumerate(ystartcoords) if y == obj.positionabsy)

    def __init__(self, tip, traintest, backgroundcol=None, keepasis=False):
        """Takes a task image pair and populates all
        properties with it
        tip ---   dict, where tip['input'] is the input and
                  taskImgPair['output'] is the output
        """
        # inputs first
        self.fullinputimg = np.array(tip["input"])

        if traintest == 'train':  # need to do this now as it's used later
            self.fulloutputimg = np.array(tip["output"])

        # find unique colours
        inuniques = np.unique(self.fullinputimg)

        # assuming background is the prominent colour: find background & assign the property
        if traintest == 'train':  # if test: we'll need to get it from a train imgpair, can't do it here
            self.findbackground()
        else:
            self.backgroundcol = backgroundcol

        # find all colours other than background colours
        inforgroundcols = inuniques.tolist()
        if self.backgroundcol in inforgroundcols:
            inforgroundcols.remove(self.backgroundcol)

        # find same colour invididual objects in image
        self.findscobjects('input', inforgroundcols, keepasis=keepasis)

        if traintest == 'train':
            outuniques = np.unique(self.fulloutputimg)

            outforgroundcols = outuniques.tolist()
            if self.backgroundcol in outforgroundcols:
                outforgroundcols.remove(self.backgroundcol)

            self.findscobjects('output', outforgroundcols, keepasis=keepasis)

        # add extra attributes
        self.extraobjattrs()

        # create the first predoutputimg
        self.fullpredimg = deepcopy(self.fullinputimg)

        try:
            self.gridrefactorobjs()
        except:
            None


class FullTask:
    trainsglimgprs = []    # list of single image pairs for train set
    testinputimg = []      # list of single image pairs for test set
    testpred = None        # list containing a numpy array for a final prediction, 1 array for each input in test

    def seperatinglinerefactorobjs(self):
        """looks for line(s) which run through the entire input, separating the input into equal sized smaller portions,
        which also equal the size of the output image, suggesting some combo of those input objs to be had. Need to
        do this on a fulltask scale
        """
        for imgpair in self.trainsglimgprs:
            if imgpair.gridapplied:  # the grid obj structure messes with this one
                del imgpair.gridapplied
                return
            else:
                del imgpair.gridapplied

            stillval = 0

            # look for lines
            linesx = []
            linesy = []
            horzorvert = []
            linecol = []
            for obj in imgpair.predoutputsamecolobjs:
                if (obj.height == imgpair.fullinputimg.shape[0]) & (obj.width == 1):
                    # vert line
                    linesx = linesx + [obj.positionabsx]
                    linesy = linesy + [obj.positionabsy]
                    linecol = linecol + [obj.colour]
                    horzorvert = horzorvert + ['vert']

                if (obj.width == imgpair.fullinputimg.shape[1]) & (obj.height == 1):
                    # horz line
                    linesx = linesx + [obj.positionabsx]
                    linesy = linesy + [obj.positionabsy]
                    linecol = linecol + [obj.colour]
                    horzorvert = horzorvert + ['horz']

            if len(horzorvert) > 0:  # there are lines that run through the whole img
                # find the objects created by the seperating lines
                if horzorvert.count(horzorvert[0]) == len(horzorvert):
                    # all the same vals: either all vert or all horz
                    linesx = linesx + [imgpair.fullpredimg.shape[1]]
                    linesy = linesy + [imgpair.fullpredimg.shape[0]]
                    subimgs = []
                    subimgspositions = []
                    colsinobjs = []
                    if horzorvert[0] == 'vert':
                        startx = 0
                        # see if all objs make by splitting full img up are the same
                        for xno in linesx:
                            subimgs = subimgs + [imgpair.fullpredimg[:, startx:xno]]
                            subimgspositions = subimgspositions + [(0, imgpair.fullpredimg.shape[0], startx, xno)]
                            colsinobjs = colsinobjs + list(np.unique(subimgs))
                            startx = xno + 1

                    else:
                        starty = 0
                        # see if all objs make by splitting full img up are the same
                        for yno in linesy:
                            subimgs = subimgs + [imgpair.fullpredimg[starty:yno, :]]
                            subimgspositions = subimgspositions + [(starty, yno, 0, imgpair.fullpredimg.shape[1])]
                            colsinobjs = colsinobjs + list(np.unique(subimgs))
                            starty = yno + 1

                else:
                    # combo of vert & horz lines
                    vertlines = np.array(horzorvert) == 'vert'
                    linesy = np.array(linesy)
                    linesx = np.array(linesx)
                    vertlinesx = np.append(linesx[vertlines], imgpair.fullpredimg.shape[0])
                    horzlinesy = np.append(linesy[vertlines == False], imgpair.fullpredimg.shape[1])
                    startx, starty = 0, 0
                    for vlines in vertlinesx:
                        for hlines in horzlinesy:
                            subimgs = subimgs + [imgpair.fullpredimg[starty:hlines, startx:vlines]]  # last one
                            subimgspositions = subimgspositions + [(starty, hlines, startx, vlines)]
                            colsinobjs = colsinobjs + list(np.unique(subimgs))
                            starty = hlines + 1
                        starty = 0
                        startx = vlines + 1

                # see if the objects can be used for some sort of comparison
                backgroundcol = max(set(colsinobjs), key=colsinobjs.count)

                stillval = 1
                multicolour = 0
                ipbackcanvas = np.zeros([imgpair.fullpredimg.shape[0], imgpair.fullpredimg.shape[1]])
                for objno, eachobj in enumerate(subimgs):
                    if not ((eachobj.shape[0] == imgpair.fulloutputimg.shape[0]) &
                            (eachobj.shape[1] == imgpair.fulloutputimg.shape[1])):
                        stillval = 0

                    if len(np.unique(eachobj)) > 2:
                        multicolour = 1
                    else:
                        cols = np.unique(imgpair.fulloutputimg)
                        col = np.delete(cols, np.argwhere(cols == backgroundcol))
                        imgpair.outputsamecolourobjs = [SameColourObject((imgpair.fulloutputimg == col)*1, col[0], keepasis=True)]
                        newlabel = deepcopy(ipbackcanvas)
                        poss = subimgspositions[objno]
                        newlabel[poss[0]:poss[1], poss[2]:poss[3]] = eachobj
                        cols = np.unique(eachobj)
                        col = np.delete(cols, np.argwhere(cols == backgroundcol))
                        subimgs[objno] = SameColourObject((eachobj == col[0])*1, col[0], keepasis=True)

                if stillval & multicolour:
                    imgpair.predoutputmulticolobjs = subimgs
                elif stillval & (not multicolour):
                    imgpair.predoutputsamecolobjs = subimgs

        if stillval:
            for imgpair in self.testinputimg:
                ipbackcanvas = np.zeros([imgpair.fullpredimg.shape[0], imgpair.fullpredimg.shape[1]])
                subimgstest = deepcopy(subimgs)
                for objno, poss in enumerate(subimgspositions):
                    eachobj = imgpair.fullpredimg[poss[0]:poss[1], poss[2]:poss[3]]
                    newlabel = deepcopy(ipbackcanvas)
                    newlabel[poss[0]:poss[1], poss[2]:poss[3]] = eachobj
                    cols = np.unique(eachobj)
                    col = np.delete(cols, np.argwhere(cols == backgroundcol))
                    subimgstest[objno] = SameColourObject(eachobj, col[0], keepasis=True)

                if stillval & multicolour:
                    imgpair.predoutputmulticolobjs = subimgstest
                elif stillval & (not multicolour):
                    imgpair.predoutputsamecolobjs = subimgstest
                    print('seperating by lines, samecolobj, succeeded')

    def createoutputcanvas(self):
        """creates the 'canvas' for the test output: i.e. size of output & background col
        """
        # see if the output image is a certain scale / size relative to the input
        stillvalid = 1
        for transpose in [0, 1]:
            rm = 0
            for imgpair in self.trainsglimgprs:
                outrow = imgpair.fulloutputimg.shape[0]
                outcol = imgpair.fulloutputimg.shape[1]
                inrow = imgpair.fullinputimg.shape[transpose]
                incol = imgpair.fullinputimg.shape[1 - transpose]

                if rm == 0:  # this is the first image
                    rm = outrow // inrow
                    rc = outrow % inrow
                    cm = outcol // incol
                    cc = outcol % incol
                else:
                    if not (((inrow * rm + rc) == outrow) & ((incol * cm + cc) == outcol)):
                        stillvalid = 0

            if stillvalid:
                for trainortest in [self.trainsglimgprs, self.testinputimg]:
                    for eachtask in trainortest:
                        if eachtask.backgroundcol is None:
                            # set to 0
                            eachtask.backgroundcol = 0

                        inrow = eachtask.fullinputimg.shape[transpose]
                        incol = eachtask.fullinputimg.shape[1 - transpose]
                        eachtask.predoutputcanvas = \
                            np.ones([int(inrow * rm + rc), int(incol * cm + cc)]) * eachtask.backgroundcol
                return

        # see if it's a fixed size:
        stillvalid = 1
        for ii, imgpair in enumerate(self.trainsglimgprs):
            if ii == 0:
                outputshape = imgpair.fulloutputimg.shape
            else:
                if imgpair.fulloutputimg.shape != outputshape:
                    stillvalid = 0

        if stillvalid:
            print('refactored by seperating by line')
            for trainortest in [self.trainsglimgprs, self.testinputimg]:
                for eachtask in trainortest:
                    if eachtask.backgroundcol is None:
                        # set to 0
                        eachtask.backgroundcol = 0

                        eachtask.predoutputcanvas = np.ones([outputshape[0], outputshape[1]])

            return

    def findtestbackground(self):
        backgroundcols = []
        # make a list of background cols for all train sets
        for trainimgpair in self.trainsglimgprs:
            backgroundcols = backgroundcols + [trainimgpair.backgroundcol]

        # if the background is the same in all sets, assign this to the test
        if len(backgroundcols) == backgroundcols.count(backgroundcols[0]):
            return backgroundcols[0]
        else:
            return None

    def __init__(self, task_file, keepasis=False):
        import json

        if isinstance(task_file, str):
            with open(task_file, 'r') as f:
                task = json.load(f)  # tasks is a dict
        else:  # assume we've entered the task from alltasks: dict
            task = task_file

        trainset = task['train']  # trainset is a list
        for ii in range(len(trainset)):
            ntpis = SingleImagePair(trainset[ii], 'train', keepasis=keepasis)
            self.trainsglimgprs = self.trainsglimgprs + [ntpis]

        testset = task['test']  # testnset is a list
        for ii in range(len(testset)):
            backgroundcol = self.findtestbackground()
            ntpis = SingleImagePair(testset[ii], 'test', backgroundcol=backgroundcol, keepasis=keepasis)

            self.testinputimg = self.testinputimg + [ntpis]

            self.createoutputcanvas()

            try:
                self.seperatinglinerefactorobjs()
            except:
                print('seperating objs by line failed')


class FullTaskFromClass(FullTask):
    def __init__(self, fulltask):
        # need to pack back into tips
        for imgpair in fulltask.trainsglimgprs:
            trainset = {'input': imgpair.fullpredimg.astype(int), 'output': imgpair.fulloutputimg}
            ntpis = SingleImagePair(trainset, 'train')
            ntpis.fullinputimg = imgpair.fullinputimg
            self.trainsglimgprs = self.trainsglimgprs + [ntpis]

        for imgpair in fulltask.testinputimg:
            backgroundcol = self.findtestbackground()
            testset = {'input': imgpair.fullpredimg.astype(int), 'output': imgpair.fulloutputimg}
            ntpis = SingleImagePair(testset, 'test', backgroundcol=backgroundcol)
            ntpis.fullinputimg = imgpair.fullinputimg

            self.testinputimg = self.testinputimg + [ntpis]

            self.createoutputcanvas()


# ~~~~~~~~~~~~~~~~~~~~~~~~ PATTERN CLASSES ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
class SinglePatImagePair(SingleImagePair):
    def __init__(self, tip, test_pred_one=0):
        # input - the un-processed input pattern
        self.fullinputimg = np.array(tip["input"])

        # output - the processed input
        if len(test_pred_one) != 1:   # got a testset
            self.fulloutputimg = test_pred_one


class FullPatTask(FullTask):
    def __init__(self, task_file, test_pred_list):
        task = task_file

        trainset = task['train']  # trainset is a list
        for ii in range(len(trainset)):
            ntpis = SinglePatImagePair(trainset[ii])
            self.trainsglimgprs = self.trainsglimgprs + [ntpis]

        testset = task['test']  # testnset is a list
        for ii in range(len(testset)):
            backgroundcol = self.findtestbackground()
            ntpis = SinglePatImagePair(testset[ii], test_pred_one=test_pred_list[ii])

            self.testinputimg = self.testinputimg + [ntpis]