import arcclasses
import patterndetect
import numpy as np
import arcrules
import misckagglesolutions


def solutionflattener(pred):
    """The following python code converts a 2d list or numpy array pred into the correct format:
    """
    if not isinstance(pred, list):  # then it'll be a numpy array
        pred = pred.tolist()

    str_pred = str([row for row in pred])
    str_pred = str_pred.replace(', ', '')
    str_pred = str_pred.replace('[[', '|')
    str_pred = str_pred.replace('][', '|')
    str_pred = str_pred.replace(']]', '|')
    return str_pred


def writesolution(t_pred_test_list, fl, taskname):
    for eachtest in range(len(t_pred_test_list)):
        str_pred = solutionflattener(t_pred_test_list[eachtest])
        fl.write(taskname + '_' + str(eachtest) + ',' + str_pred + '\n')


def singlesolution(task, patmodel):
    acc = 0
    patterntask = patterndetect.makepredictions(task, patmodel)

    if patterntask:
        # t_pred_test_list is a list containing numpy array, 1 element for each input in test
        acc, t_pred_test_list = patterndetect.check_p(task, patterndetect.patch_image)

        if acc != 1:
            None  # make a pattern class here and see if we can get results from that
        else:
            sol = 'pattern'

    if acc != 1:
        subtaskdonelist = []
        taskclass = arcclasses.FullTask(task)
        try:
            acc, fulltask = arcrules.findnextrule(taskclass, subtaskdonelist)
            t_pred_test_list = fulltask.testpred

            if acc == 1:
                sol = 'arcrule'
        except Exception as inst:
            print('errored with symbols')
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)
            acc = 0

    if acc != 1:
        subtaskdonelist = []
        try:
            taskclass = arcclasses.FullTask(task)  # reload as symbols will stay
            acc, fulltask = arcrules.findnextrule(taskclass, subtaskdonelist, symbols=False)
            t_pred_test_list = fulltask.testpred

            if acc == 1:
                sol = 'arcrule'
            else:
                fulltask.testpred = []
                for testno, onetestpred in enumerate(fulltask.testinputimg):
                    if isinstance(onetestpred.fullpredimg, list):
                        fulltask.testpred = fulltask.testpred + [int(x) for x in onetestpred.fullpredimg]
                    else:  # assume numpy array
                        ontestpredlist = onetestpred.fullpredimg.astype(int)
                        fulltask.testpred = fulltask.testpred + [ontestpredlist.tolist()]
        except Exception as inst:
            print('errored without symbols')
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)
            acc = 0

    if acc != 1:
        try:
            a = misckagglesolutions.toplevel1(task)
        except Exception as inst:
            print('misckaggle errored')
            print(type(inst))  # the exception instance
            print(inst.args)  # arguments stored in .args
            print(inst)
            a = -1
            acc = 0

        if a != -1:
            print('misc kaggle: {} to 1'.format(acc))
            acc = 1
            t_pred_test_list = [a]
            fulltask.testinputimg[0].fullpredimg = np.array(t_pred_test_list[0])
            sol = 'misckaggle'
        else:
            sol = 'arcrule'

    return fulltask, t_pred_test_list, sol


def solutions(alltasks, tasknames, patmodel):
    """patmodel is the trained model used to detect pattern problems (and potentially other kind of problems in the
    future
    """

    subloc = 'C:/Users/User1/Documents/Programming/Abstract reasoning Kaggle/' \
             'abstraction-and-reasoning-challenge/submission.csv'

    fl = open(subloc, "w")
    fl.write("output_id,output\n")

    score = []

    for ii in range(len(alltasks)):
        acc = 0
        task = alltasks[ii]
        taskname = tasknames[ii]
        patterntask = patterndetect.makepredictions(task, patmodel)
        # print(str(ii) + ', ' + taskname)

        if patterntask:
            # t_pred_test_list is a list containing numpy array, 1 element for each input in test
            acc, t_pred_test_list = patterndetect.check_p(task, patterndetect.patch_image)

            if acc != 1:
                None  # make a pattern class here and see if we can get results from that

        if acc != 1:
            print(ii)
            subtaskdonelist = []
            taskclass = arcclasses.FullTask(task)
            try:
                acc, fulltask = arcrules.findnextrule(taskclass, subtaskdonelist)
                t_pred_test_list = fulltask.testpred

            except Exception as inst:
                print('errored with symbols')
                print(type(inst))  # the exception instance
                print(inst.args)  # arguments stored in .args
                print(inst)
                acc = 0

        if acc != 1:
            print(ii)
            subtaskdonelist = []
            try:
                taskclass = arcclasses.FullTask(task)  # reload as symbols will stay
                acc, fulltask = arcrules.findnextrule(taskclass, subtaskdonelist, symbols=False)
                t_pred_test_list = fulltask.testpred

                if acc == 1:
                    sol = 'arcrule'
                else:
                    fulltask.testpred = []
                    for testno, onetestpred in enumerate(fulltask.testinputimg):
                        if isinstance(onetestpred.fullpredimg, list):
                            fulltask.testpred = fulltask.testpred + [int(x) for x in onetestpred.fullpredimg]
                        else:  # assume numpy array
                            ontestpredlist = onetestpred.fullpredimg.astype(int)
                            fulltask.testpred = fulltask.testpred + [ontestpredlist.tolist()]
            except Exception as inst:
                print('errored without symbols')
                print(type(inst))  # the exception instance
                print(inst.args)  # arguments stored in .args
                print(inst)
                acc = 0

        if acc != 1:
            try:
                a = misckagglesolutions.toplevel1(task)
            except Exception as inst:
                print('misckaggle errored')
                print(type(inst))  # the exception instance
                print(inst.args)  # arguments stored in .args
                print(inst)
                a = -1
                acc = 0

            if a != -1:
                print('misc kaggle: {} to 1'.format(acc))
                acc = 1
                t_pred_test_list = [a]
                sol = 'misckaggle'

        if ('t_pred_test_list' in locals()) and (t_pred_test_list is not None) and (len(t_pred_test_list) != 0):
            writesolution(t_pred_test_list, fl, taskname)
            if acc == 1:
                score = score + [1]
            else:
                score = score + [0]
        else:  # just add an incorrect placeholder solution
            for eachtest in range(len(task['test'])):
                fl.write(taskname + '_' + str(eachtest) + ',' + '|000000000|00000000|' + '\n')
                score = score + [0]

    print(score.count(1) / len(score))


def runfullscript(dataset='train'):
    import cloudpickle
    import initialdefs
    alltasks, tasknames = initialdefs.startup(dataset)

    f = open('C:/Users/User1/Documents/Programming/Abstract reasoning Kaggle/patternmodel.pckl', 'rb')
    model = cloudpickle.load(f)

    solutions(alltasks, tasknames, model)


def testatask(task):
    import cloudpickle
    import plottingfuns

    f = open('C:/Users/User1/Documents/Programming/Abstract reasoning Kaggle/patternmodel.pckl', 'rb')
    model = cloudpickle.load(f)

    #taskclass = arcclasses.FullTask(task)
    try:
        fulltask, t_pred_test_list, sol = singlesolution(task, model)
    except Exception as inst:
        print(type(inst))  # the exception instance
        print(inst.args)  # arguments stored in .args
        print(inst)
        acc = 0

    # if acc != 1:
    #     taskclass = arcclasses.FullTask(task)
    #     acc, taskclass = arcrules.findnextrule(taskclass, [], symbols=False)  # trying again without symbols

    if (not sol == 'pattern') or (not sol == 'misckaggle'):
        plottingfuns.plot_task(fulltask)