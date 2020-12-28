import time

import numpy as np
import torch
from scipy import linalg
from sklearn.utils import check_random_state


def resample_pmatrix(data, n_iterations, rng):
    """
    Get bounds on the precision matrix using bootstrap.

    :param data: A pandas dataframe containing the data
    :param n_iterations: The number of iterations to run the bootstrap
    :param rng: Random seed

    :return: Lower and upper bounds as torch tensors
    """
    stats = []
    for i in range(n_iterations):
        sample = bootstrap(data, rng)
        sigma = sample.cov()
        prec_emp = linalg.inv(sigma)
        stats.append(prec_emp.flatten())

    lower, upper = pmatrix_conf_interval(stats)
    lower = torch.tensor(np.reshape(lower, (data.shape[1], data.shape[1])))
    upper = torch.tensor(np.reshape(upper, (data.shape[1], data.shape[1])))

    return lower, upper


def pmatrix_conf_interval(stats, alpha=95):
    ordered_pmatrix = []
    for i in range(len(stats[0])):
        sublist = [elem[i] for elem in stats]
        sublist.sort()
        ordered_pmatrix.append(sublist)

    # Alternatively: return percentile(ordered_pmatrix, alpha)
    return np.percentile(ordered_pmatrix, (100-alpha)/2, axis=1), \
           np.percentile(ordered_pmatrix, alpha + ((100-alpha)/2), axis=1)


def resample_eigendecomposition(data, n_iterations, rng, method='bootstrap', frac=1):
    stats = []
    # print(f"Calculating {n_iterations} samples...")

    for i in range(n_iterations):
        if method == 'permutation':
            sample = permutation(data, rng, frac)
        elif method == 'monte carlo':
            sample = monte_carlo(data, rng, frac)
        elif method == 'bootstrap':
            sample = bootstrap(data, rng, frac)

        sigma = sample.cov()
        eigenvalues, eigenvectors = linalg.eigh(sigma)
        # Sort the eigenvalues and corresponding eigenvectos
        idx = eigenvalues.argsort()
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        stats.append([eigenvalues, eigenvectors])

    eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper = conf_interval(stats)

    eigvals_lower = torch.tensor(eigvals_lower)
    eigvals_upper = torch.tensor(eigvals_upper)
    eigvects_lower = torch.tensor(eigvects_lower)
    eigvects_upper = torch.tensor(eigvects_upper)

    return eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper


def monte_carlo(data, rng, frac):
    return data.sample(frac=frac, replace=False, random_state=rng)


def permutation(data, rng, frac=1):
    return data.sample(frac=frac, replace=False, random_state=rng)


def bootstrap(data, rng, frac=1):
    return data.sample(frac=frac, replace=True, random_state=rng)


def conf_interval(stats, alpha=95):
    ordered_eigenvectors = []
    ordered_eigenvalues = []
    for i in range(len(stats[0][0])):
        eigenvals = np.array([elem[0][i] for elem in stats])
        eigenvects = np.array([elem[1][:, i] for elem in stats])
        idx_sort = eigenvals.argsort()
        ordered_eigenvalues.append(eigenvals[idx_sort])
        ordered_eigenvectors.append(eigenvects[idx_sort, :])

    eigvals_lower, eigvals_upper = percentile(ordered_eigenvalues, alpha)
    eigvects_lower, eigvects_upper = percentile(ordered_eigenvectors, alpha)

    return eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper


def percentile(data, alpha):
    perc_lower = (100-alpha)/2
    perc_upper = alpha + ((100 - alpha) / 2)
    idx_lower = int(np.ceil((len(data[0]) * perc_lower) / 100)) - 1
    idx_upper = int(np.ceil((len(data[0]) * perc_upper) / 100)) - 1

    return [elem[idx_lower] for elem in data],  [elem[idx_upper] for elem in data]


def time_data_splits(X, n_iter):
    for i in np.linspace(0.1, 1, 10):
        start = time.time()
        resample_eigendecomposition(data=X, n_iterations=n_iter, rng=check_random_state(0), method='bootstrap', frac=i)
        end = time.time()
        print(f"Time needed for {n_iter} iterations and {i*100}% of the data: {end - start}")


def time_iterations(X):
    for i in range(1, 6):
        start = time.time()
        resample_eigendecomposition(data=X, n_iterations=10 ** i, rng=check_random_state(0), method='bootstrap', frac=1)
        end = time.time()
        print(f"Time needed for {10**i} iterations: {end - start}")
