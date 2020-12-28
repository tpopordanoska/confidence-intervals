import matplotlib as mpl
import matplotlib.pyplot as plt

from cieg.experiments.methods.pmatrix import *

FONTSIZE_BIG = 16
FONTSIZE_MEDIUM = 12
FONTSIZE_SMALL = 10


def set_fontsize(experiment):
    if experiment.name == 'higgs_third':
        plt.rcParams.update({'font.size': FONTSIZE_SMALL})
        return FONTSIZE_SMALL

    plt.rcParams.update({'font.size': FONTSIZE_BIG})
    return FONTSIZE_BIG


def plot_pmatrix_bounds(path, method, experiment):
    set_fontsize(experiment)

    lower, upper, empirical = load_pmatrix_bounds(path, method)

    plot_and_save_pmatrix(lower, path, f"lb_pmatrix_{experiment.name}_{method}.pdf", experiment)
    plot_and_save_pmatrix(empirical, path, f'emp_pmatrix_{experiment.name}_{method}.pdf', experiment, False)
    plot_and_save_pmatrix(upper, path, f"ub_pmatrix_{experiment.name}_{method}.pdf", experiment)


def plot_eig_bounds(path, method, experiment):
    set_fontsize(experiment)

    eigvals_lower, eigvals_upper, eigvals_emp, eigvects_lower, eigvects_upper, eigvects_emp = load_eig_bounds(path, method)
    plot_and_save_eigvals(eigvals_lower, path, f"eigvals_lower_{experiment.name}_{method}.pdf", experiment)
    plot_and_save_eigvals(eigvals_emp, path, f'eigvals_emp_{experiment.name}_{method}.pdf', experiment, False)
    plot_and_save_eigvals(eigvals_upper, path, f"eigvals_upper_{experiment.name}_{method}.pdf", experiment)

    plot_and_save_eigvects(eigvects_lower, path, f"eigvects_lower_{experiment.name}_{method}.pdf", experiment)
    plot_and_save_eigvects(eigvects_emp, path, f'eigvects_emp_{experiment.name}_{method}.pdf', experiment, False)
    plot_and_save_eigvects(eigvects_upper, path, f"eigvects_upper_{experiment.name}_{method}.pdf", experiment)


def plot_and_save_eigvals(array, path, plot_name, experiment, is_bound=True):
    FONTSIZE = set_fontsize(experiment)

    fig, ax = plt.subplots(1, 1, figsize=(3, 2))
    if is_bound:
        cmap = mpl.cm.Blues(np.linspace(0, 1, 20))
    else:
        cmap = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[0:10, :-1])
    data = np.reshape(array, (1, len(array)))
    ax.imshow(data, cmap=cmap)
    ax.set_xticklabels("")
    ax.set_yticklabels("")

    for i in range(len(array)):
        c = np.around(array[i].item(), decimals=2)
        ax.text(i, 0, str(c), va='center', ha='center', fontsize=FONTSIZE)

    plt.savefig(os.path.join(path, plot_name), bbox_inches='tight')
    plt.show()


def plot_and_save_eigvects(matrix, path, plot_name, experiment, is_bound=True):
    FONTSIZE = set_fontsize(experiment)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if is_bound:
        cmap = mpl.cm.Blues(np.linspace(0, 1, 20))
    else:
        cmap = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[0:10, :-1])
    ax.matshow(matrix, cmap=cmap)  # plt.get_cmap('Blues'))
    ax.set_xticklabels("")
    ax.set_yticklabels("")
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            c = np.around(matrix[j, i].item(), decimals=2)
            ax.text(i, j, str(c), va='center', ha='center', fontsize=FONTSIZE)

    plt.savefig(os.path.join(path, plot_name), bbox_inches='tight')
    plt.show()


def plot_and_save_pmatrix(matrix, path, plot_name, experiment, is_bound=True):
    FONTSIZE = set_fontsize(experiment)

    fig, ax = plt.subplots(1, 1, figsize=(7, 4))
    if is_bound:
        cmap = mpl.cm.Blues(np.linspace(0, 1, 20))
    else:
        cmap = mpl.cm.Reds(np.linspace(0, 1, 20))
    cmap = mpl.colors.ListedColormap(cmap[0:10, :-1])
    ax.matshow(matrix, cmap=cmap)  # plt.get_cmap('Blues'))
    ax.set_xticklabels([''] + experiment.column_names)
    ax.set_yticklabels([''] + experiment.column_names)
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            c = np.around(matrix[j, i].item(), decimals=2)
            ax.text(i, j, str(c), va='center', ha='center', fontsize=FONTSIZE)

    plt.savefig(os.path.join(path, plot_name), bbox_inches='tight')
    plt.show()
