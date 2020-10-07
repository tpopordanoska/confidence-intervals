import os
from datetime import datetime as dt

import matplotlib.pyplot as plt
from scipy.stats import norm as normal_dist
from tqdm import tqdm

from cieg.eigenvectors import *
from cieg.experiments.methods import *
from cieg.experiments.utils import preprocess
from cieg.utils.covariance import cov

n_simulation_rounds = 100  # Number of simulations in total
N = 500000  # Number of samples
d = 5  # Number of dimensions
eps = 1e-9  # Threshold for entries of PM to be zero
n_alphas = 100  # How many alphas to check
alphas = np.linspace(0, 0.999, n_alphas)
np.random.seed(42)


def main():
    run_and_save_results()
    plot_results()


def run_and_save_results():
    fpr_laplace_fisher = []
    fpr_gauss_fisher = []
    fpr_laplace_our = []
    fpr_gauss_our = []
    simulation_round = 0

    while simulation_round < n_simulation_rounds:
        # Generating a random adjacency matrix
        # Must be symmetric
        threshold = np.random.choice([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        adj = 1. * (np.random.rand(d, d) > threshold)
        adj *= (1 - np.eye(d))
        adj = np.tril(adj) + np.triu(adj.T, 1)
        # Converting adjacency matrix to precision matrix by making it PD
        S, V = np.linalg.eigh(adj)
        S = np.maximum(np.diag(S), np.eye(adj.shape[0]) + abs(np.random.rand()))
        precision_matrix = V @ S @ V.T
        idx_zeros = (np.abs(precision_matrix) <= eps).ravel()
        # We want a precision matrix with zeros
        if idx_zeros.sum() == 0:
            continue
        # Now we can create a random covariance
        cov_random = np.linalg.inv(precision_matrix)
        # Generating the data from the multivariate Laplace distribution
        # Look at Kotz. Samuel; Kozubowski, Tomasz J.; Podgorski, Krzysztof (2001).
        # The Laplace Distribution and Generalizations. Birkhauser., page 249
        mu = np.zeros((d, 1))
        W = -1 * np.log(np.random.rand(d, N))
        NN = np.random.multivariate_normal(np.zeros((d,)), cov_random, size=N).T
        x_laplace = mu * W + np.sqrt(W) * NN

        fpr_laplace_fisher.append(assess_fisher_test(x_laplace, alphas, idx_zeros))
        fpr_gauss_fisher.append(assess_fisher_test(NN, alphas, idx_zeros))
        fpr_laplace_our.append(assess_our(x_laplace, alphas, idx_zeros))
        fpr_gauss_our.append(assess_our(NN, alphas, idx_zeros))
        simulation_round += 1

    print(f'Total rounds: {len(fpr_laplace_fisher)}')

    path = create_folders()
    np.save(os.path.join(path, 'fpr_laplace_fisher.npy'), fpr_laplace_fisher)
    np.save(os.path.join(path, 'fpr_gauss_fisher.npy'), fpr_gauss_fisher)
    np.save(os.path.join(path, 'fpr_laplace_our.npy'), fpr_laplace_our)
    np.save(os.path.join(path, 'fpr_gauss_our.npy'), fpr_gauss_our)


def assess_fisher_test(data, alphas, idx_zeros):
    # Estimating the partial correlations and z-transforms for them
    prec_emp = np.linalg.inv(np.cov(data))
    z = np.zeros(prec_emp.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            # We do not need to compute the transfor on diagonals, it is infinity
            if i != j:
                partial_corr = -prec_emp[i, j] / np.sqrt(prec_emp[i, i] * prec_emp[j, j])
                z[i, j] = 0.5 * np.log((1 + partial_corr) / (1 - partial_corr))
            else:
                z[i, j] = np.inf

    # Computing the statistic for Fisher's test
    statistic = np.sqrt(data.shape[1] - (z.shape[0] - 2) - 3) * np.abs(z)
    fprs = []
    for alpha in alphas:
        # Doing the test
        pred_non_zero = statistic.ravel() > normal_dist.ppf(1 - alpha / 2)
        if idx_zeros.sum() > 0:
            fpr = pred_non_zero[idx_zeros].sum() / idx_zeros.sum()
        else:
            fpr = 0
        fprs.append(fpr)
    return fprs


def assess_our(data, alphas, idx_zeros):
    data = torch.tensor(data).float()
    data = preprocess(data)
    sigma = cov(data)
    eig = get_eig(sigma)
    fprs = []
    for alpha in tqdm(alphas):
        eps = get_eps(data, alpha)
        # print(f"Eps: {eps}")

        eigvals_lower, eigvals_upper = get_eigenvalue_bounds(eig.eigenvalues, eps)
        eigvects_lower, eigvects_upper = get_eigenvector_bounds(eps, eig, sigma)

        _, _, pred_non_zero = pmatrix_bounds(eigvals_lower, eigvals_upper,  eigvects_lower, eigvects_upper, sigma, eig)
        pred_non_zero = pred_non_zero.data.numpy().ravel()

        if idx_zeros.sum() > 0:
            fpr = pred_non_zero[idx_zeros].sum() / idx_zeros.sum()
        else:
            fpr = 0
        fprs.append(fpr)
    return fprs


def create_folders():
    # Create a folder 'results'
    path_results = os.path.join(os.getcwd(), "results")
    if not os.path.exists(path_results):
        try:
            os.mkdir(path_results)
        except OSError:
            print(f"Creation of the directory {path_results} failed")

    # Create a separate folder for each time running the experiment
    path = os.path.join(path_results, dt.now().strftime('%Y-%m-%d_%H-%M-%S.%f'))
    try:
        os.mkdir(path)
    except OSError:
        print("Creation of the directory %s failed" % path)

    return path


def plot_results(folder_name=None):
    path_results = os.path.join(os.getcwd(), "results")
    if not folder_name:
        folder_name = os.listdir(path_results)[-1]
    path = os.path.join(path_results, folder_name)

    laplace_fisher = np.load(os.path.join(path, 'fpr_laplace_fisher.npy'))
    gauss_fisher = np.load(os.path.join(path, 'fpr_gauss_fisher.npy'))
    laplace_our = np.load(os.path.join(path, 'fpr_laplace_our.npy'))
    gauss_our = np.load(os.path.join(path, 'fpr_gauss_our.npy'))

    laplace_fisher_mean, laplace_fisher_std = get_mean_and_std(laplace_fisher)
    gauss_fisher_mean, gauss_fisher_std = get_mean_and_std(gauss_fisher)
    laplace_our_mean, laplace_our_std = get_mean_and_std(laplace_our)
    gauss_our_mean, gauss_our_std = get_mean_and_std(gauss_our)

    create_and_save_plot(alphas,
                         laplace_fisher_mean,
                         laplace_fisher_std,
                         laplace_our_mean,
                         laplace_our_std,
                         "Laplace distribution",
                         path)

    create_and_save_plot(alphas,
                         gauss_fisher_mean,
                         gauss_fisher_std,
                         gauss_our_mean,
                         gauss_our_std,
                         "Gaussian distribution",
                         path)


def get_mean_and_std(data):
    return np.mean(data, 0), np.std(data, 0) / np.sqrt(n_simulation_rounds)


def create_and_save_plot(alphas, mean_f, std_f, mean_our, std_our, title, path):
    plt.rcParams.update({'font.size': 16})
    plt.figure(figsize=(6, 6))
    plt.plot(alphas, mean_f, 'r', label='Fisher test')
    plt.fill_between(alphas, mean_f - std_f, mean_f + std_f, color='r', alpha=0.25, linewidth=0)
    plt.plot(alphas, mean_our, 'b', label='Our method')
    plt.fill_between(alphas, mean_our - std_our, mean_our + std_our, color='b', alpha=0.25, linewidth=0)
    plt.plot([0, 1], [0, 1], color='black', linestyle='--')
    plt.title(title)
    plt.xlabel('Significance')
    plt.ylabel('False Positive Rate')
    plt.legend()
    plt.savefig(os.path.join(path, f'{title}.pdf'), bbox_inches='tight')


if __name__ == '__main__':
    main()
