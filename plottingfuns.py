import matplotlib.pyplot as plt
from matplotlib import colors


def plot_one(ax, task, i, traintest, input_or_output):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    input_matrix = task[traintest][i][input_or_output]
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(traintest + ' ' + input_or_output)


def plot_one_class(ax, task, i, traintest, inoutpred):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)

    if traintest == 'train':
        if inoutpred == 'input':
            input_matrix = task.trainsglimgprs[i].fullinputimg
        elif inoutpred == 'output':
            input_matrix = task.trainsglimgprs[i].fulloutputimg
        else:
            if task.trainsglimgprs[i].fullpredimg is None:
                return

            input_matrix = task.trainsglimgprs[i].fullpredimg
    else:
        if inoutpred == 'input':
            input_matrix = task.testinputimg[i].fullinputimg
        elif inoutpred == 'pred':
            if task.testinputimg[i].fullpredimg is None:
                return

            input_matrix = task.testinputimg[i].fullpredimg

        else:
            return

    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=0.5)
    ax.set_yticks([x - 0.5 for x in range(1 + len(input_matrix))])
    ax.set_xticks([x - 0.5 for x in range(1 + len(input_matrix[0]))])
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(traintest + ' ' + inoutpred)


def plot_task(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    import arcclasses
    if isinstance(task, arcclasses.FullTask) or isinstance(task, arcclasses.FullTaskFromClass):
        plotarcclass(task)
        return

    num_train = len(task['train'])
    fig, axs = plt.subplots(2, num_train, figsize=(3 * num_train, 3 * 2))
    for i in range(num_train):
        plot_one(axs[0, i], task, i, 'train', 'input')
        plot_one(axs[1, i], task, i, 'train', 'output')
    plt.tight_layout()
    plt.show()

    num_test = len(task['test'])
    fig, axs = plt.subplots(2, num_test, figsize=(3 * num_test, 3 * 2))
    if num_test == 1:
        plot_one(axs[0], task, 0, 'test', 'input')
        plot_one(axs[1], task, 0, 'test', 'output')
    else:
        for i in range(num_test):
            plot_one(axs[0, i], task, i, 'test', 'input')
            plot_one(axs[1, i], task, i, 'test', 'output')
    plt.tight_layout()
    plt.show()


def print_numpy_arr(task):
    num_train = len(task['train'])
    for i in range(num_train):
        print()


def plotarcclass(task):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    num_train = len(task.trainsglimgprs)
    fig, axs = plt.subplots(3, num_train, figsize=(3 * num_train, 3 * 2))
    for i in range(num_train):
        plot_one_class(axs[0, i], task, i, 'train', 'input')
        plot_one_class(axs[1, i], task, i, 'train', 'output')
        plot_one_class(axs[2, i], task, i, 'train', 'pred')
    plt.tight_layout()
    plt.show()

    num_test = len(task.testinputimg)
    fig, axs = plt.subplots(3, num_test, figsize=(3 * num_test, 3 * 2))
    if num_test == 1:
        plot_one_class(axs[0], task, 0, 'test', 'input')
        plot_one_class(axs[1], task, 0, 'test', 'output')
        plot_one_class(axs[2], task, 0, 'test', 'pred')
    else:
        for i in range(num_test):
            plot_one_class(axs[0, i], task, i, 'test', 'input')
            plot_one_class(axs[1, i], task, i, 'test', 'output')
            plot_one_class(axs[2, i], task, i, 'test', 'pred')
    plt.tight_layout()
    plt.show()
