def startup(dataset='train', printsample=False):
    # This Python 3 environment comes with many helpful analytics libraries installed
    # It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
    # For example, here's several helpful packages to load in

    import numpy as np  # linear algebra
    import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

    # Input data files are available in the "../input/" directory.
    # For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

    import os

    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    import json
    from pathlib import Path

    import numpy as np

    import plottingfuns as arcplot

    data_path = Path('C:/Users/User1/Documents/Programming/Abstract reasoning Kaggle/abstraction-and-reasoning-challenge')

    # this is here in case we need it but we don't need to really print out every file in the repo!
    # for dirname, _, filenames in os.walk(data_path):
    #     print(dirname)

    from pathlib import Path

    # Any results you write to the current directory are saved as output.

    training_path = data_path / 'training'
    evaluation_path = data_path / 'evaluation'
    test_path = data_path / 'test'

    training_tasks = sorted(os.listdir(training_path))
    evaluation_tasks = sorted(os.listdir(evaluation_path))
    test_tasks = sorted(os.listdir(test_path))

    if printsample:  # plot task 9
        i = 9
        task_file = str(training_path / training_tasks[i])

        with open(task_file, 'r') as f:
            task = json.load(f)

        print(i)
        print(training_tasks[i])
        arcplot.plot_task(task)
        arcplot.print_numpy_arr(task)

    alltasks = []
    tasknames = []

    if dataset is 'train':
        for i in range(len(training_tasks)):
            task_file = str(training_path / training_tasks[i])
            tasknames.append(training_tasks[i].split('.')[0])

            with open(task_file, 'r') as f:
                nexttask = json.load(f)
                alltasks.append(nexttask)
    elif dataset is 'eval':
        for i in range(len(evaluation_tasks)):
            task_file = str(evaluation_path / evaluation_tasks[i])
            tasknames.append(evaluation_tasks[i].split('.')[0])

            with open(task_file, 'r') as f:
                nexttask = json.load(f)
                alltasks.append(nexttask)
    elif dataset is 'test':
        for i in range(len(test_tasks)):
            task_file = str(test_path / test_tasks[i])
            tasknames.append(test_tasks[i].split('.')[0])

            with open(task_file, 'r') as f:
                nexttask = json.load(f)
                alltasks.append(nexttask)
    else:
        print('dataset assigned to non-existent string')
        alltasks = 0

    return alltasks, tasknames
