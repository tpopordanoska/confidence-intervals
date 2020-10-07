import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os


def plot_and_save_bounds(lower, upper, empirical, title, path):
    fig, ax = plt.subplots(1, 3, figsize=(15, 7))
    plt.rcParams.update({'font.size': 16})
    # fig.suptitle(title)

    plot_matrix(lower, ax[0], 'Lower bound')
    plot_matrix(empirical, ax[1], 'Empirical value', False)
    plot_matrix(upper, ax[2], "Upper bound")

    plt.savefig(os.path.join(path, f'{title}.pdf'), bbox_inches='tight')
    plt.show()


def plot_matrix(matrix, ax, title, is_bound=True):
    if is_bound:
        cmap = mpl.cm.Blues(np.linspace(0, 1, 20))
    else:
        cmap = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[0:10, :-1])
    ax.matshow(matrix, cmap=cmap) # plt.get_cmap('Blues'))
    ax.set_title(title)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            c = np.around(matrix[j, i].item(), decimals=3)
            ax.text(i, j, str(c), va='center', ha='center', fontsize=24)


def plot_bounds_comparison(lower1, upper1, lower2, upper2, emp, title):
    fig, ax = plt.subplots(2, 3, figsize=(17, 14))
    fig.suptitle(title, size=18)
    plot_matrix(lower1, ax[0, 0], 'Lower')
    plot_matrix(emp, ax[0, 1], 'Empirical')
    plot_matrix(upper1, ax[0, 2], "Upper")
    plot_matrix(lower2, ax[1, 0], 'Lower')
    plot_matrix(emp, ax[1, 1], 'Empirical')
    plot_matrix(upper2, ax[1, 2], "Upper")

    plt.show()
