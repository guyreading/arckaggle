import numpy as np
from copy import deepcopy
import arcclasses
import datetime
import time
import xgboost as xgb

# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~ functions used by entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~ entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def colouriner(imgpair):
    """if the input & output is the same other than objects being a different colour, then yes
    imgpair is a SingleImagePair object
    """
    # look for multi-colour objs against a background
    if imgpair.backgroundcol is not None:
        inputbw = (imgpair.fullpredimg == imgpair.backgroundcol) * 1
        outputbw = (imgpair.fulloutputimg == imgpair.backgroundcol) * 1

        if np.array_equal(inputbw, outputbw) & (not np.array_equal(imgpair.fullpredimg, imgpair.fulloutputimg)):
            # this says that the objects are the same... but:
            # is one object a multicolour object?
            # is the background incorrect?

            # easy win:
            if len(imgpair.predoutputsamecolobjs) != len(imgpair.outputsamecolourobjs):
                return 2

    # look for same-colour objs
    if len(imgpair.predoutputsamecolobjs) == len(imgpair.outputsamecolourobjs):
        identitycount = 0
        for objin in imgpair.outputsamecolourobjs:
            for objout in imgpair.predoutputsamecolobjs:
                if (np.array_equal(objin.elementarr, objout.elementarr)) & (objin.positionabsx == objout.positionabsx) \
                        & (objin.positionabsy == objout.positionabsy):
                    identitycount += 1

        if identitycount == len(imgpair.outputsamecolourobjs):
            # all objects in input can be found in output in the same location
            return 1
        else:
            return 2
    else:
        return 0


def zoominer(imgpair, returnsz=1):
    """if the output is a zoomed in version of the input, return 1
    imgpair is a SingleImagePair object
    """
    inimg = imgpair.fullinputimg
    outimg = imgpair.fulloutputimg

    # raster-scan an "outimg" sized image across inimg, see if any of the segments are equal to outimg
    if (inimg.shape[0] > outimg.shape[0]) & (inimg.shape[1] > outimg.shape[1]):
        for ii in range(inimg.shape[0] - outimg.shape[0]):
            for jj in range(inimg.shape[1] - outimg.shape[1]):
                rows = np.repeat(list(range(ii, ii + outimg.shape[0])), outimg.shape[1])
                cols = list(range(jj, jj + outimg.shape[1])) * outimg.shape[0]
                inzoom = inimg[rows, cols]
                if np.array_equal(inzoom, outimg.flatten()):
                    if returnsz == 1:
                        return 1
                    else:
                        return 1, ii, jj

    return 0


def zoomonobject(imgpair):
    for obj in imgpair.inputsamecolourobjs:
        if np.array_equal(obj.elementarr * obj.colour, imgpair.fulloutputimg):
            return 1

    return 0


def objremer(imgpair):
    """finds if an object(s) from the input is removed from the output
    """
    if len(imgpair.inputsamecolourobjs) <= len(imgpair.outputsamecolourobjs):
        return 0

    if len(imgpair.inputsamecolourobjs) > 100:
        return 0

    outobjcount = [0] * len(imgpair.outputsamecolourobjs)
    for outobjno, outobj in enumerate(imgpair.outputsamecolourobjs):
        for inobj in imgpair.inputsamecolourobjs:
            if np.array_equal(inobj.elementarr, outobj.elementarr):
                outobjcount[outobjno] = 1

    if len(outobjcount) == outobjcount.count(1):
        return 1
    else:
        return 0


def listofuniqueshapes(objlist):
    objswiththatshape = []  # list of [list for each shape: which contains obj nos associated with that shape]
    listofshapes = []  # list of np arrays, each of which is a unique shape
    for objno, eachobj in enumerate(objlist):
        newshape = 1
        for shapeno, shape in enumerate(listofshapes):  # list of all unique symbols for this task
            if np.array_equal(eachobj.elementarr, shape):
                newshape = 0
                objswiththatshape[shapeno] = objswiththatshape[shapeno] + [objno]
                break

        if newshape:  # made it to the end of listofshapes, not in there, add it
            listofshapes = listofshapes + [eachobj.elementarr]
            objswiththatshape = objswiththatshape + [[objno]]

    return {'shapes': listofshapes, 'objswithshape': objswiththatshape}


def symbolser(fulltask):
    """look for re-occuring symbols across all the tasks
    """
    # question is: what constitutes a symbol, what constitutes just a normal object?
    # very basic: let's turn everything into a symbol
    listofsymbols = []
    allsymbolnumbers = []  # unique number for each symbol found
    for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
        for eachtest in traintest:
            symbolsinimg = []
            for eachobj in eachtest.predoutputsamecolobjs:
                stillvalid = 1
                counter = 0
                for symbolno, symbol in enumerate(listofsymbols):  # list of all symbols for this task
                    if np.array_equal(eachobj.elementarr, symbol):
                        stillvalid = 0
                        symbolsinimg = symbolsinimg + [symbolno]
                        break

                    counter += 1

                if stillvalid:  # made it to the end of listofsymbols, not in there, add it
                    listofsymbols = listofsymbols + [eachobj.elementarr]

                    # add this symbol number to symbols in this image:
                    allsymbolnumbers = allsymbolnumbers + [counter]
                    symbolsinimg = symbolsinimg + [counter]

            # when we've gone through each object in the set, we should add these to every obj
            for eachobj in eachtest.predoutputsamecolobjs:
                for symbolno, eachsymbol in enumerate(symbolsinimg):
                    setattr(eachobj, 'symbol' + str(symbolno), eachsymbol)

    return fulltask


def booleannoter(fulltask):
    boolnottask = 0
    for imgpair in fulltask.trainsglimgprs:
        toomanyobjs = len(imgpair.predoutputsamecolobjs) > 40
        if not toomanyobjs:
            for objpred in imgpair.predoutputsamecolobjs:
                newelemarr = 1 - objpred.elementarr
                objnottoosmall = newelemarr.sum() > 1
                if objnottoosmall:
                    # remove cols/rows of all zeros
                    validcols = newelemarr.sum(axis=0) > 0  # logical index col
                    validrows = newelemarr.sum(axis=1) > 0  # logical index row
                    firstcol = min(np.where(validcols)[0])
                    firstrow = min(np.where(validrows)[0])
                    lastcol = max(np.where(validcols)[0])
                    lastrow = max(np.where(validrows)[0])
                    validcols[firstcol:lastcol] = True
                    validrows[firstrow:lastrow] = True
                    newelemarr = newelemarr[np.ix_(validrows, validcols)]

                    for objout in imgpair.outputsamecolourobjs:

                        # if any one obj is the same, should apply a not to at least 1 of these objects
                        if np.array_equal(newelemarr, objout.elementarr):
                            boolnottask = 1

    return boolnottask


def booleanlogicer(imgpair):
    """need only 2 input shapes and they both need to be the same size as the output img
    """
    stillvalid = 1
    if len(imgpair.predoutputsamecolobjs) != 2:
        stillvalid = 0

    for obj in imgpair.predoutputsamecolobjs:
        if (obj.elementarr.shape[0] != imgpair.fulloutputimg.shape[0]) or \
                (obj.elementarr.shape[1] != imgpair.fulloutputimg.shape[1]):
            stillvalid = 0

    return stillvalid


def movingobjectser(imgpair):
    """Prelim requirements for moving objs around. check that all objs can be mapped from in to out and they're
    in different locations at the in than they are at the out
    """
    intooutobjs, warnings = linkinobjtooutobj(imgpair)
    if len(warnings) > 0:  # we need 1:1 mapping from in:out, as in test, won't know what to map
        return 0

    inobjs = intooutobjs['inobjs']

    if (len(inobjs) == len(imgpair.predoutputsamecolobjs)) and (len(inobjs) == len(imgpair.outputsamecolourobjs)):
        return 1
    else:
        return 0


# ~~~~~~~~~~~~~~~~~~~~~~~~ rules to apply if passes entry requirements ~~~~~~~~~~~~~~~~~~~~
def accbyinputpixtooutput(fulltask):
    """returns a pixel-wise comparison of matching pixels, comparing input to output
    """
    allaccs = []

    for imgpair in fulltask.trainsglimgprs:
        if imgpair.fullpredimg.shape == imgpair.fulloutputimg.shape:
            # can compare on a pixel-wise basis
            samepix = np.equal(imgpair.fullpredimg, imgpair.fulloutputimg) * 1
            unique, count = np.unique(samepix, return_counts=True)
            if (1 in unique) and (0 in unique):
                oneidx = np.where(unique == 1)[0][0]
                acc = count[oneidx] / (sum(count))
            elif (1 in unique) and (0 not in unique):
                acc = 1
            else:
                acc = 0
        else:
            # should compare on an object-wise basis
            linkedobjs, warning = linkinobjtooutobj(imgpair)  # outobjs in dict are outobjs linked
            # if there are same amount of linked objs to output objs: all objects are accounted for. max acc = 0.9
            acc = 0.9 - np.tanh(abs(len(linkedobjs['outobjs']) - len(imgpair.outputsamecolourobjs)) +
                                abs(len(linkedobjs['outobjs']) - len(imgpair.predoutputsamecolobjs)))

        allaccs = allaccs + [acc]

    acc = sum(allaccs) / len(allaccs)

    return acc


def subtaskvalidation(fulltaskold, fulltasknew, taskname):
    fulltasknew = placeobjsintofullimg(fulltasknew)

    accnew = accbyinputpixtooutput(fulltasknew)
    accold = accbyinputpixtooutput(fulltaskold)
    print('{} - acc before: {}, acc after: {}'.format(taskname, accold, accnew))

    if accnew == 1:
        fulltasknew = placeobjsintofullimg(fulltasknew)
        fulltasknew.testpred = []
        for testimgpairs in fulltasknew.testinputimg:
            fulltasknew.testpred = fulltasknew.testpred + [testimgpairs.fullpredimg]

    if accnew > accold:
        return accnew, fulltasknew
    else:
        return accold, fulltaskold


def findintattrs(fulltask):
    """returns the attributes of the fulltask which are int, as these can be unpacked easily as
    features for an NN
    """
    attrvals = vars(fulltask.trainsglimgprs[0].predoutputsamecolobjs[0])
    samecolobjattrs = list(vars(fulltask.trainsglimgprs[0].predoutputsamecolobjs[0]).keys())
    isint = []

    for attr in samecolobjattrs:
        if isinstance(attrvals[attr], int):
            isint.append(attr)

    return isint


def resultsfrommodel(xtest, ylabels, model):
    """passes xtest through a model to return a set of y predictions
    """
    model, modeltype = model

    if modeltype == 'nn':
        predictions = model.predict(xtest)
        results = np.dot(np.round(predictions), ylabels)  # put them back into an array which holds the colours
    elif modeltype == 'otomap':
        otomap, col = model
        results = []
        for ii in range(xtest.shape[0]):
            xval = str(int(xtest[ii, col]))  # sometimes leave .0 on so need to do int
            results = results + [otomap[xval]]
    elif modeltype == 'xgb':
        predictions = model.predict(xtest)
        predictions2, _ = ylabelstooh(predictions)
        results = np.dot(np.round(predictions2), ylabels)  # put them back into an array which holds the colours

    return results


def createxfeatures(imgpair, objno, isint, maxobjno):
    """creates all the x features for one sample. Many samples in an xsamples, which calls this fun.
    """
    features = [0] * maxobjno * len(isint)
    attrcount = 0

    objlist = imgpair.predoutputsamecolobjs

    # features for this obj go at the beginning
    for attr in isint:
        features[attrcount] = getattr(objlist[objno], attr)
        attrcount += 1

    # make a list of numbers for all objects other than main obj
    otherobjs = list(range(len(objlist)))
    otherobjs.remove(objno)

    # loop through list, each loop looping through attrs like above
    for otherobj in otherobjs:
        for attr in isint:
            features[attrcount] = getattr(objlist[otherobj], attr)
            attrcount += 1

    return np.array(features).reshape(1, maxobjno * len(isint))


def findmaxobjno(fulltask):
    """finds the max no of objs in both train & test so that the correct size for xtrain is given
    """
    maxobjno = 0

    # make the maxobjno size the size of imgpair with the most objs
    for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
        for imgpair in traintest:
            objlist = imgpair.predoutputsamecolobjs

            if len(objlist) > maxobjno:
                maxobjno = len(objlist)

    return maxobjno


def createxsamples(imgpairlist, isint, maxobjno):
    """creates an array which creates x samples to train the NN on.
    Input (imgpairlist): testinputimg or trainsglimgprs
    samples is np array. each row is 1 sample. Each col is 1 feature.
    """
    for imgpair in imgpairlist:
        objlist = imgpair.predoutputsamecolobjs

        for objno in range(len(objlist)):
            if 'samples' in locals():
                samples = np.vstack((samples, createxfeatures(imgpair, objno, isint, maxobjno)))
            else:
                samples = createxfeatures(imgpair, objno, isint, maxobjno)

    # remove cols which don't have any variation
    # colsallsame = np.where(np.all(samples == samples[0, :], axis=0))[0]

    # if colsallsame.size != 0:
    #     samples = np.delete(samples, colsallsame, 1)

    return samples


def makesimplemodel(xtrain, ytrain, ylabels, isint, xtest):
    from keras.models import Sequential
    from keras.layers import Dense
    from sklearn.metrics import accuracy_score

    # try a one-to-one mapping first
    revertedy = np.dot(ytrain, ylabels)

    symbolcols = [i for i, name in enumerate(isint) if not name.find('symbol')]
    acc = 0

    for allcols in [symbolcols, range(len(isint))]:
        for cols in allcols:

            otomap = {}
            otomaprev = {}  # make sure there ins't just a list of unique xvals mapping to same val of y's
            for rows in range(xtrain.shape[0]):
                xval = str(xtrain[rows, cols])
                yval = str(revertedy[rows])
                if xval in otomap:
                    if (otomap[xval] != revertedy[rows]) or (otomaprev[yval] != xtrain[rows, cols]):
                        break
                else:
                    otomap[xval] = revertedy[rows]
                    otomaprev[yval] = xtrain[rows, cols]

                if rows == xtrain.shape[0] - 1:
                    acc = 1
                    break

            if acc:
                break
        if acc:
            break

    if acc == 1:  # first if acc == 1: need to now check that this oto mapping is still valid for test
        for rows in range(xtest.shape[0]):
            if not str(xtest[rows, cols]) in otomap.keys():
                # oto map will err
                acc = 0

    if acc == 1:
        model = otomap, cols  # the correct mapping & the col from x train that it was from
        modeltype = 'otomap'
        print(isint[cols])
    else:
        yclasses = np.size(ytrain, axis=1)
        usexgboost = 1

        if usexgboost:
            model = xgb.XGBClassifier(max_depth=3, eta=1, reg_lambda=5)
            model.fit(xtrain, revertedy)
            prediction = model.predict(xtrain)
            prediction2, _ = ylabelstooh(prediction)
            ypred = np.dot(np.round(prediction2), ylabels)
            acc = accuracy_score(revertedy, ypred)
            modeltype = 'xgb'
        else:
            # Neural network
            model = Sequential()
            model.add(Dense(16, input_dim=np.size(xtrain, axis=1), activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(12, activation='relu'))
            model.add(Dense(yclasses, activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            model.fit(xtrain, ytrain, epochs=800, verbose=0)

            # evaluate the keras model
            _, acc = model.evaluate(xtrain, ytrain)

            modeltype = 'nn'

    return (model, modeltype), acc


def ylabelstooh(ylabel):
    from sklearn.preprocessing import OneHotEncoder

    # turn into a ohe
    ohe = OneHotEncoder()
    ylabel = np.array(ylabel)
    ylabelunique = np.unique(ylabel)
    ylabel = ylabel.reshape(-1, 1)
    y = ohe.fit_transform(ylabel).toarray()

    return y, ylabelunique


def createcolysamples(imgpairlist):
    ylabel = []
    for imgpair in imgpairlist:
        inobjs = imgpair.predoutputsamecolobjs
        outobjs = imgpair.outputsamecolourobjs

        for inobj in inobjs:
            xpos = inobj.positionabsx
            ypos = inobj.positionabsy

            # the same object in output might not be in the same index so will need to find it
            for outobj in outobjs:
                if (outobj.positionabsx == xpos) & (outobj.positionabsy == ypos):
                    break

            ylabel = ylabel + [outobj.colour]

    y, ylabelunique = ylabelstooh(ylabel)

    return y, ylabelunique


def makecolourpredictions(fulltask, model, isint, ylabels, maxobjno):
    altfulltask = deepcopy(fulltask)

    for traintest in [altfulltask.trainsglimgprs, altfulltask.testinputimg]:
        for testno, eachtest in enumerate(traintest):
            # find predictions from model
            xtest = createxsamples([eachtest], isint, maxobjno)
            # print(xtest.shape)

            results = resultsfrommodel(xtest, ylabels, model)

            # put the predictions into a final image
            for objno in range(len(eachtest.predoutputsamecolobjs)):
                eachtest.predoutputsamecolobjs[objno].colour = results[objno]

    return altfulltask


def placeobjsintosingleimgpair(imgpair):
    # make a blank canvas
    outputimg = np.zeros([imgpair.fullinputimg.shape[0], imgpair.fullinputimg.shape[1]])

    for objno, obj in enumerate(imgpair.inputsamecolourobjs):
        rowsalt = np.repeat(list(range(obj.positionabsx, obj.positionabsx + obj.height)), obj.width)
        colsalt = list(range(obj.positionabsy, obj.positionabsy + obj.width)) * obj.height
        vals = obj.elementarr.flatten() * obj.colour

        outputimg[rowsalt, colsalt] = vals

    return outputimg


def placeobjsintofullimg(fulltask):
    """takes objects from predoutputsamecolobjs and places them back into a fullpredimg image.
    We do this after every sub-task so that we can see if we've completed the task.
    """
    fulltask.testpred = []

    for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
        for testno, eachtest in enumerate(traintest):
            outputimg = np.copy(eachtest.predoutputcanvas)
            # need another incase obj locations are out of bounds for new canvas (need to move objs around for new
            # canvas, still, but still wanna keep previous changes)
            outputimgorig = np.zeros([eachtest.fullpredimg.shape[0], eachtest.fullpredimg.shape[1]])
            returnorigimgsize = 0

            for objno, obj in enumerate(eachtest.predoutputsamecolobjs):
                # colsalt = list(np.repeat(list(range(obj.positionabsx, obj.positionabsx + obj.height)), obj.width))
                # rowsalt = list(range(obj.positionabsy, obj.positionabsy + obj.width)) * obj.height
                colsalt = list(range(obj.positionabsx, obj.positionabsx + obj.width)) * obj.height
                rowsalt = list(np.repeat(list(range(obj.positionabsy, obj.positionabsy + obj.height)), obj.width))

                vals = list(obj.elementarr.flatten() * obj.colour)

                # remove values which are 0, so the background val is maintained  rowsalt
                indices = [i for i, x in enumerate(vals) if x == obj.colour]
                rowsalt = [d for (i, d) in enumerate(rowsalt) if i in indices]
                colsalt = [d for (i, d) in enumerate(colsalt) if i in indices]
                vals = [d for (i, d) in enumerate(vals) if i in indices]

                outputimgorig[rowsalt, colsalt] = vals
                try:
                    outputimg[rowsalt, colsalt] = vals
                except IndexError:
                    # got an err therefore we should return the other canvas
                    returnorigimgsize = 1

            if returnorigimgsize:
                eachtest.fullpredimg = outputimgorig
            else:
                eachtest.fullpredimg = outputimg

    return fulltask


def colourchange(fulltaskin):
    print('colourchange accessed')
    fulltask = deepcopy(fulltaskin)

    # unroll all the features & samples & put into array
    isint = findintattrs(fulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(fulltask)
    xtrain = createxsamples(fulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    # print(xtrain.shape)
    ytrain, ylabels = createcolysamples(fulltask.trainsglimgprs)  # makes y_set. one-hot np.
    # print(ytrain.shape)

    # special case: only 1 y output type
    ytrain2 = ytrain.tolist()
    if ytrain2.count(ytrain2[0]) == len(ytrain2):  # all 1's & will err out
        for traintest in [fulltask.trainsglimgprs, fulltask.testinputimg]:
            for imgno, imgpair in enumerate(traintest):
                for objno, obj in enumerate(imgpair.predoutputsamecolobjs):
                    obj.colour = fulltask.trainsglimgprs[0].outputsamecolourobjs[0].colour
        fulltask = placeobjsintofullimg(fulltask)
        acc, fulltaskfinal = subtaskvalidation(fulltaskin, fulltask, 'colourchange')
        return acc, fulltaskfinal

    # make & train a model on prorperties from train set
    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    # use trained NN to predict colours for test set(s) + change the colours
    fulltask = makecolourpredictions(fulltask, model, isint, ylabels, maxobjno)

    # find acc of new iteration
    acc, fulltaskfinal = subtaskvalidation(fulltaskin, fulltask, 'colourchange')

    return acc, fulltaskfinal


def multicolourchange(fulltask):
    """this could be so complicated & could come in so many guises, I wouldn't know where to start with a
    hard-coded solutions. I'm just going to throw a NN at it.
    """
    print('multicolourchange accessed')
    acc = 0
    return acc, fulltask


def createyzoomsamples(imgpairlist):
    """Outputs a list of booleans. 1 if the object in question was what was zoomed in on. 0 if not
    """
    ytrain = []

    for imgpair in imgpairlist:
        for obj in imgpair.predoutputsamecolobjs:
            if np.array_equal(obj.elementarr * obj.colour, imgpair.fulloutputimg):
                ytrain = ytrain + [1]
            else:
                ytrain = ytrain + [0]

    y, ylabelunique = ylabelstooh(ytrain)

    return y


def makezoomobjpredictions(fulltask, model, isint, maxobjno):
    fulltask.testpred = []

    for testno, eachtest in enumerate(fulltask.testinputimg):
        # find predictions from model
        xtest = createxsamples([eachtest], isint, maxobjno)
        ylabels = [0, 1]

        results = resultsfrommodel(xtest, ylabels, model)

        objno = np.where(results == 1)[0][0]

        predobj = fulltask.testinputimg[testno].predoutputsamecolobjs[objno].elementarr * \
                  fulltask.testinputimg[testno].predoutputsamecolobjs[objno].colour

        fulltask.testpred = fulltask.testpred + [predobj]

    return fulltask


def zoomspecialrulecheck(fulltask):
    objsinimg = []
    for imgpairno in range(len(fulltask.trainsglimgprs)):
        objsinimg = objsinimg + [len(fulltask.trainsglimgprs[imgpairno].predoutputsamecolobjs)]

    if (len(objsinimg) == objsinimg.count(objsinimg[0])) & (objsinimg[0] == 1):
        return 1
    else:
        return 0


def zoomobjrules(fulltask):
    print('zoomobjrules accessed')

    # special, easy rule if there's only one obj to choose from:
    if zoomspecialrulecheck(fulltask):
        fulltask.testpred = []
        for istrain, traintest in enumerate([fulltask.trainsglimgprs, fulltask.testinputimg]):
            for testno, eachtest in enumerate(traintest):
                colour = eachtest.predoutputsamecolobjs[0].colour
                objimg = eachtest.predoutputsamecolobjs[0].elementarr
                eachtest.fullpredimg = objimg * colour
                if not istrain:
                    fulltask.testpred = fulltask.testpred + [objimg * colour]

        return 1, fulltask

    isint = findintattrs(fulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(fulltask)
    xtrain = createxsamples(fulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    ytrain = createyzoomsamples(fulltask.trainsglimgprs)

    # train NN on prorperties from train set
    ylabels = [0, 1]  # boolean of "was this zoomed in on"
    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    # use trained NN to predict colours for test set(s)
    fulltask = makezoomobjpredictions(fulltask, model, isint, maxobjno)

    return acc, fulltask


def zoomnoobjrules(fulltask):
    acc = 0
    return acc, fulltask


def zoomrules(fulltask):
    """looks for rules determining what section to zoom in on, in the input, and why
    """
    print('zoomrules accessed')
    if checkforallimages(fulltask, zoomonobject):
        acc, fulltask = zoomobjrules(fulltask)  # zoomed on a specific object we have in input
    else:
        acc, fulltask = zoomnoobjrules(fulltask)  # zoomed on a specific area in input but not exactly around an obj

    return acc, fulltask


def createyobjremsamples(imgpairlist):
    """outputs a list of booleans. 1 for if the input img exists in the output. 0 if not
    """
    ytrain = []

    for imgpair in imgpairlist:
        for inobj in imgpair.predoutputsamecolobjs:
            objexists = 0
            for outobj in imgpair.outputsamecolourobjs:
                if np.array_equal(inobj.elementarr, outobj.elementarr):
                    objexists = 1

            ytrain = ytrain + [objexists]

    y, ylabelunique = ylabelstooh(ytrain)

    return y


def makeobjrempredictions(fulltask, model, isint, maxobjno):
    """predict which objects are to be removed in the train & test input image(s) & remove them, then check
    the prediction with the train images
    """
    # make a copy of the fulltask - gonna make some deletions
    newfulltask = deepcopy(fulltask)

    # remove the objects from inputsamecolobjs in train
    for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
        for testno, eachtest in enumerate(traintest):
            xtrain = createxsamples([eachtest], isint, maxobjno)  # np arr with samples=rows, features=cols
            # print('xtrain no {} has {} rows and {} cols'.format(testno, xtrain.shape[0], xtrain.shape[1]))
            ylabels = [0, 1]

            results = resultsfrommodel(xtrain, ylabels, model)

            # this is the first obj manipulation task
            if len(eachtest.predoutputsamecolobjs) == 0:
                eachtest.predoutputsamecolobjs = deepcopy(eachtest.fullinputimg)

            objs = eachtest.predoutputsamecolobjs

            noofobjs = len(objs)
            for objno in range(noofobjs - 1, -1, -1):  # go backwards as if we del, all idxs will shift down one
                if results[objno] == 1:  # del this
                    del (objs[objno])

    # let's see if that's been positive
    acc, finalfulltask = subtaskvalidation(fulltask, newfulltask, 'objrem')

    return acc, finalfulltask


def objremrules(fulltask):
    """Looks for rules determining what objects are removed from the input and why
    """
    print('object remove rules accessed')
    isint = findintattrs(fulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(fulltask)
    xtrain = createxsamples(fulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    ytrain = createyobjremsamples(fulltask.trainsglimgprs)

    # train NN on prorperties from train set
    ylabels = [1, 0]  # objects we want to del
    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    # use trained NN to predict colours for test set(s)
    acc, fulltask = makeobjrempredictions(fulltask, model, isint, maxobjno)

    return acc, fulltask


def linkinobjtooutobj(imgpair, maptype='oto'):
    """for each in object, see it exists as an output. If so, return it's array list number. Return
    the output object's x position and y position"""
    warning = ''

    inshapesall = listofuniqueshapes(imgpair.predoutputsamecolobjs)
    outshapesall = listofuniqueshapes(imgpair.outputsamecolourobjs)
    inshapes = inshapesall['shapes']
    outshapes = outshapesall['shapes']
    inobjswshapes = inshapesall['objswithshape']
    outobjswshapes = outshapesall['objswithshape']
    if len(inobjswshapes) != len(outobjswshapes):
        warning = 'different shapes in as out /n'

    inobj = []
    outobj = []

    for inshapeno, eachinshape in enumerate(inshapes):
        for outshapeno, eachoutshape in enumerate(outshapes):
            if np.array_equal(eachinshape, eachoutshape):  # got a matching pair, make sure they're okay
                if len(inobjswshapes[inshapeno]) == len(outobjswshapes[outshapeno]):  # same no of ins to outs:
                    for objno in range(len(inobjswshapes[inshapeno])):
                        if maptype == 'oto':  # one to one
                            # just assign x in to x out so we get a 1:1 mapping
                            inobj = inobj + [inobjswshapes[inshapeno][objno]]
                            outobj = outobj + [outobjswshapes[outshapeno][objno]]

                        if maptype == 'otm':  # one to many
                            None  # for now

                else:
                    warning = warning + 'different number of ins to outs /n'

    if len(inobj) != len(imgpair.predoutputsamecolobjs):
        warning = warning + 'not all in shapes accounted for: need to remove some first'

    return {'inobjs': inobj, 'outobjs': outobj}, warning


def createymovesamples(imgpair, axis):
    positionlist = []
    for obj in imgpair.outputsamecolourobjs:
        positionlist = positionlist + [getattr(obj, 'positionabs' + axis)]

    return positionlist


def creatingmovingobjsxset(imgpairlist, isint, maxobjno, traintestno, xyaxis):
    for imgpairno, imgpair in enumerate(imgpairlist):
        intooutobjs, warnings = linkinobjtooutobj(imgpair)
        if len(warnings) > 0:  # we need 1:1 mapping from in:out, as in test, won't know what to map
            return 0, 0

        inobjs = intooutobjs['inobjs']
        outobjs = intooutobjs['outobjs']

        xtrainraw = createxsamples([imgpair], isint, maxobjno)  # np array with samples=rows, features=cols

        if imgpairno == 0:
            xtrain = xtrainraw[inobjs[0], :]
            for objno in inobjs[1:]:
                xtrain = np.vstack([xtrain, xtrainraw[objno, :]])

            if traintestno == 0:
                ytrainraw = createymovesamples(imgpair, xyaxis)
                ytrainraw2 = [ytrainraw[outobjs[0]]]
                for objno in outobjs[1:]:
                    ytrainraw2 = ytrainraw2 + [ytrainraw[objno]]
            else:
                ytrainraw2 = 0
        else:
            for objno in inobjs:
                xtrain = np.vstack([xtrain, xtrainraw[objno, :]])

            if traintestno == 0:
                ytrainraw = createymovesamples(imgpair, xyaxis)
                for objno in outobjs:
                    ytrainraw2 = ytrainraw2 + [ytrainraw[objno]]

    return xtrain, ytrainraw2


def movingobjects(fulltask):
    """looks to determine rules for where to move each object if they need moving
    """
    print('movingobjects accessed')
    newfulltask = deepcopy(fulltask)

    isint = findintattrs(newfulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(newfulltask)

    for xyaxis in ['x', 'y']:
        xtrain, ytrainraw2 = creatingmovingobjsxset(newfulltask.trainsglimgprs, isint, maxobjno, 0, xyaxis)
        xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)

        # train
        ytrain, ylabels = ylabelstooh(ytrainraw2)
        model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

        for traintestno, traintest in enumerate([newfulltask.trainsglimgprs, newfulltask.testinputimg]):
            for imgpairno, imgpair in enumerate(traintest):
                # now make predictions with the model
                xset = createxsamples([imgpair], isint, maxobjno)
                results = resultsfrommodel(xset, ylabels, model)

                # assign the new val
                for objno in range(len(imgpair.predoutputsamecolobjs)):
                    setattr(imgpair.predoutputsamecolobjs[objno], 'positionabs' + xyaxis, int(results[objno]))

    acc, finalfulltask = subtaskvalidation(fulltask, newfulltask, 'moveobjs')

    return acc, finalfulltask


def booleannot(fulltask):
    """applies a boolean not to each object in turn. If accuracy goes up, keep the not
    """
    print('booleannot accessed')
    newfulltask = deepcopy(fulltask)

    # make the y's
    boolnotobjs = []
    for imgpair in newfulltask.trainsglimgprs:
        for obj in imgpair.predoutputsamecolobjs:
            newelemarr = 1 - obj.elementarr

            # check this elemarr against output img
            y1 = obj.positionabsy
            y2 = obj.positionabsy + obj.elementarr.shape[0]
            x1 = obj.positionabsx
            x2 = obj.positionabsx + obj.elementarr.shape[1]
            outputimg = (imgpair.fulloutputimg[y1:y2, x1:x2] != imgpair.backgroundcol) * 1

            # add to list saying if it is or ins't a not
            boolnotobjs = boolnotobjs + [np.array_equal(newelemarr, outputimg)]

    isint = findintattrs(newfulltask)  # finds all attribues which are ints & can be used as features
    maxobjno = findmaxobjno(newfulltask)

    xtrain = createxsamples(newfulltask.trainsglimgprs, isint, maxobjno)  # np array with samples=rows, features=cols
    ytrain, ylabels = ylabelstooh(boolnotobjs)

    xtest = createxsamples(fulltask.testinputimg, isint, maxobjno)
    model, acc = makesimplemodel(xtrain, ytrain, ylabels, isint, xtest)

    for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
        for imgpair in traintest:
            xtrain = createxsamples([imgpair], isint, maxobjno)
            results = resultsfrommodel(xtrain, ylabels, model)
            for objno, obj in enumerate(imgpair.predoutputsamecolobjs):
                if results[objno]:
                    obj.elementarr = 1 - obj.elementarr

    # turn this into a new class as we might have gained/lost objects
    newfulltask = placeobjsintofullimg(newfulltask)
    newfulltask = arcclasses.FullTaskFromClass(newfulltask)

    acc, finalfulltask = subtaskvalidation(fulltask, newfulltask, 'booleannot')

    return acc, finalfulltask


def booltests(newfulltask, test):
    for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
        for imgpair in traintest:
            if test == 0:  # logical and
                taskname = 'logical and'
                logicalarr = np.logical_and(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr) * 1
                imgpair.predoutputsamecolobjs = [arcclasses.SameColourObject(logicalarr, 1, keepasis=True)]
                imgpair.fullpredimg = logicalarr
            elif test == 1:  # logical or
                taskname = 'logical or'
                logicalarr = np.logical_or(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr) * 1
                imgpair.predoutputsamecolobjs = [arcclasses.SameColourObject(logicalarr, 1, keepasis=True)]
                imgpair.fullpredimg = logicalarr
            elif test == 2:  # logical nand
                taskname = 'logical nand'
                logicalarr = np.logical_not(np.logical_or(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr)) * 1
                imgpair.predoutputsamecolobjs = [arcclasses.SameColourObject(logicalarr, 1, keepasis=True)]
                imgpair.fullpredimg = logicalarr
            elif test == 3:  # logical nor
                taskname = 'logical nor'
                logicalarr = np.logical_not(np.logical_or(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr)) * 1
                imgpair.predoutputsamecolobjs = [arcclasses.SameColourObject(logicalarr, 1, keepasis=True)]
                imgpair.fullpredimg = logicalarr
            elif test == 4:  # logical xor
                taskname = 'logical xor'
                logicalarr = np.logical_xor(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr) * 1
                imgpair.predoutputsamecolobjs = [arcclasses.SameColourObject(logicalarr, 1, keepasis=True)]
                imgpair.fullpredimg = logicalarr
            elif test == 5:  # logical xnor
                taskname = 'logical xnor'
                logicalarr = np.logical_not(np.logical_xor(imgpair.predoutputsamecolobjs[0].elementarr, imgpair.predoutputsamecolobjs[1].elementarr)) * 1
                imgpair.predoutputsamecolobjs = [arcclasses.SameColourObject(logicalarr, 1, keepasis=True)]
                imgpair.fullpredimg = logicalarr

    # elif test == 6:
    #     for traintest in [newfulltask.trainsglimgprs, newfulltask.testinputimg]:
    #         for imgpair in traintest:
    return newfulltask, taskname


def booleanlogic(fulltask):
    print('boolean logic accessed')
    for imgpair in fulltask.trainsglimgprs:
        imgpair.fullpredimg = imgpair.predoutputsamecolobjs[0].elementarr
    accold = accbyinputpixtooutput(fulltask)
    accbest = accold
    for test in range(5):
        newfulltask = deepcopy(fulltask)
        newfulltask, taskname = booltests(newfulltask, test)
        accnew, fulltasknew = subtaskvalidation(fulltask, newfulltask, taskname)
        if accnew > accbest:
            accbest = accnew
            toptest = test

    if accbest > accold:
        newfulltask = deepcopy(fulltask)
        newfulltask, taskname = booltests(newfulltask, toptest)

        # to re-order the class
        accnew, newfulltask = subtaskvalidation(fulltask, newfulltask, taskname)
        return accbest, newfulltask
    else:
        # no luck, return the old, unchanged fulltask
        return 0, fulltask


# ~~~~~~~~~~~~~~~~~~~~~~~~~~ looping through entry requirements ~~~~~~~~~~~~~~~~~~~~~~~~~
def checkforallimages(fulltask, function):
    truthlist = []
    for ii in range(len(fulltask.trainsglimgprs)):
        truthlist.append(function(fulltask.trainsglimgprs[ii]))

    if truthlist.count(truthlist[0]) == len(truthlist):  # all are the same
        return truthlist[0]
    else:
        return 0


def timeout_handler(signum, frame):
    raise TimeoutException


class TimeoutException(Exception):
    pass


def findnextrule(fulltask, subtaskdonelist, symbols=True):
    """This loops through all rule entry requirements and looks for a rule which satisfies requirements. If requirements
    are met, it then looks for rules to define that type of behaviour. e.g. see if objects just need to be coloured in
    (entry requirements). If so: what determines the rule of colouring in?
    fulltask is a FullTask: if it has been input as an argument to second or greater depth calls for findnextrule, this
    may be adjusted from original fulltask.
    subtaskdonelist is a list of subtasks done, in case we keep recognising a task as needing that transform done on it,
    so we don't end up in an endless loop
    """
    startprocess = datetime.datetime.now()
    if (0 not in subtaskdonelist) & symbols:
        fulltask = symbolser(fulltask)
        subtaskdonelist = subtaskdonelist + [0]

    if (checkforallimages(fulltask, booleanlogicer) == 1) & (6 not in subtaskdonelist):
        acc, fulltask = booleanlogic(fulltask)
        subtaskdonelist = subtaskdonelist + [6]
    elif (checkforallimages(fulltask, colouriner) == 2) & (2 not in subtaskdonelist):
        # do multicolour stuff
        acc, fulltask = multicolourchange(fulltask)
        subtaskdonelist = subtaskdonelist + [2]
    elif (checkforallimages(fulltask, zoominer) == 1) & (3 not in subtaskdonelist):
        acc, fulltask = zoomrules(fulltask)
        subtaskdonelist = subtaskdonelist + [3]
    elif (checkforallimages(fulltask, objremer) == 1) & (4 not in subtaskdonelist):
        acc, fulltask = objremrules(fulltask)
        subtaskdonelist = subtaskdonelist + [4]
    elif (booleannoter(fulltask) == 1) & (5 not in subtaskdonelist):
        acc, fulltask = booleannot(fulltask)
        subtaskdonelist = subtaskdonelist + [5]
    elif (checkforallimages(fulltask, colouriner) == 1) & (1 not in subtaskdonelist):
        # do colour stuff
        acc, fulltask = colourchange(fulltask)
        subtaskdonelist = subtaskdonelist + [1]
    elif (checkforallimages(fulltask, movingobjectser) == 1) & (7 not in subtaskdonelist):
        acc, fulltask = movingobjects(fulltask)
        subtaskdonelist = subtaskdonelist + [7]
    else:
        # no more rules to apply
        acc = 0
        return acc, fulltask

    endprocess = datetime.datetime.now()
    print('Time spent on this process was: {}'.format(endprocess - startprocess))

    if acc == 1:
        for testno, onetestpred in enumerate(fulltask.testpred):
            if isinstance(onetestpred, list):
                fulltask.testpred[testno] = [int(x) for x in onetestpred]
            else:  # assume numpy array
                fulltask.testpred[testno] = onetestpred.astype(int)
                fulltask.testpred[testno] = fulltask.testpred[testno].tolist()

        return acc, fulltask
    else:
        # go again to see if we can find the next step
        acc, fulltask = findnextrule(fulltask, subtaskdonelist, symbols)
        return acc, fulltask
