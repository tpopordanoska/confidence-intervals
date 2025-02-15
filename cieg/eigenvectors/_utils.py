from collections import defaultdict

import numpy as np
import torch

from cieg.utils import minor
from cieg.utils.covariance import covCov_estimator


def get_eig_bounds(matrix, eps):
    eig = get_eig(matrix)
    v_lower, v_upper = get_eigenvector_bounds(eps, eig, matrix)

    return v_lower, v_upper


def get_eps(X, alpha=0.05):
    cov_cov = covCov_estimator(X)
    eig = get_eig(cov_cov)
    lambda_max = torch.max(eig.eigenvalues)
    normal = torch.distributions.Normal(0, 1)

    return normal.icdf(torch.tensor(1 - alpha / 2)) * torch.sqrt(2 * lambda_max)


def get_eig(matrix):
    return torch.symeig(matrix, eigenvectors=True)


def get_eigenvalue_bounds(lambdas, eps):
    lambdas_lower = lambdas - eps
    lambdas_lower[torch.lt(lambdas_lower, 0)] = 0
    lambdas_upper = lambdas + eps

    return lambdas_lower, lambdas_upper


def cieg(X, sigma, eig, alpha=0.05):
    eps = get_eps(X, alpha)
    print(f"Eps: {eps}")

    eigvals_lower, eigvals_upper = get_eigenvalue_bounds(eig.eigenvalues, eps)
    eigvects_lower, eigvects_upper = get_eigenvector_bounds(eps, eig, sigma)

    return eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper


def get_eigenvector_bounds(eps, eig, matrix):
    # calculate the absolute value of the pairwise distances between eigenvalues
    pd_lambdas = torch.abs(torch.ger(eig.eigenvalues, torch.ones_like(eig.eigenvalues)) -
                           torch.ger(torch.ones_like(eig.eigenvalues), eig.eigenvalues))

    # make sure that diagonal is not included when testing whether non-trivial upper bounds
    # based on eigenvalue-eigenvector identity are possible
    ind = np.diag_indices(pd_lambdas.shape[0])
    pd_lambdas[ind[0], ind[1]] = (2 * eps) + torch.ones(pd_lambdas.shape[0])
    # the entries equal to "True" are the rows/cols for which a non-trivial upper bound is possible
    # (13) Proposition 1
    ubind = torch.min(pd_lambdas, 0).values > 2 * eps

    # compute the lower and upper bounds on the product in the denominator of
    # the eigenvalue-eigenvector identity
    prod_lower = torch.prod(pd_lambdas - 2 * eps, 0)  # lower bound
    prod_lower[ubind == False] = 0
    pd_lambdas[ind[0], ind[1]] = (-2 * eps) + torch.ones(pd_lambdas.shape[0])
    prod_upper = torch.prod(pd_lambdas + 2 * eps, 0)  # upper bound

    lbind = np.zeros(matrix.size())
    lambdas = eig.eigenvalues

    prod_mj_lower = torch.zeros_like(matrix)
    prod_mj_upper = float("Inf") * torch.ones_like(matrix)
    p = lambdas.size()[0]
    for j in range(p):
        Mj = minor(matrix, j)
        mj_lambdas = torch.symeig(Mj, eigenvectors=False, upper=True, out=None).eigenvalues
        pd_l_mj_l = torch.abs(
            torch.ger(lambdas, torch.ones_like(mj_lambdas)) - torch.ger(torch.ones_like(lambdas), mj_lambdas))
        lbind[j] = torch.min(pd_l_mj_l, 1).values > 2 * eps
        prod_mj_upper[j] = torch.prod(pd_l_mj_l + 2 * eps, 1)
        prod_mj_lower[j] = torch.prod(pd_l_mj_l - 2 * eps, 1)
    prod_mj_lower = torch.transpose(torch.abs(prod_mj_lower) * lbind, 0, 1)
    prod_mj_upper = torch.transpose(prod_mj_upper, 0, 1)

    v_lower = torch.transpose(torch.sqrt(prod_mj_lower / torch.ger(prod_upper, torch.ones_like(prod_upper))), 0, 1)\
        .type(torch.DoubleTensor)
    v_upper = torch.transpose(torch.sqrt(prod_mj_upper / torch.ger(prod_lower, torch.ones_like(prod_lower))), 0, 1)\
        .type(torch.DoubleTensor)

    # enforce trivial bounds due to orthonormality
    v_lower, v_upper = enforce_normality(v_lower, v_upper)

    # now get the signed version of the upper and lower bounds on V
    #  cases:
    #  1) Vhat positive, lbind==1: don't change the upper and lower bounds
    #  3) Vhat negative, lbind==1: v_upper = -v_lower, v_lower = -v_upper
    #  3) lbind==0: v_lower = -v_upper : in this case, we do not have a non-trivial lower bound on the absolute value
    # case 2:
    tmpinds = np.where((lbind == True) * (torch.sign(eig.eigenvectors).numpy() == -1))
    tmp = -v_lower[tmpinds[0], tmpinds[1]]
    v_lower[tmpinds[0], tmpinds[1]] = -v_upper[tmpinds[0], tmpinds[1]]
    v_upper[tmpinds[0], tmpinds[1]] = tmp
    # case 3:
    tmpinds = np.where(lbind == False)
    v_lower[tmpinds[0], tmpinds[1]] = -v_upper[tmpinds[0], tmpinds[1]]

    print("========= Before orthogonality constraints: =========")
    print(v_lower)
    print(v_upper)

    v_lower, v_upper = enforce_orthogonality(v_lower, v_upper)
    print("========= After orthogonality constraints: =========")
    print(v_lower)
    print(v_upper)

    return v_lower, v_upper


def enforce_normality(v_lower, v_upper):
    v_upper = torch.clamp(v_upper, min=0.0, max=1.0)
    # We can compute tighter bounds due to L2 norm of vectors equal to one.
    l2_upper = v_lower * v_lower
    l2_upper = torch.sqrt(1 - (torch.ones_like(l2_upper).matmul(l2_upper) - l2_upper))
    v_upper = torch.min(v_upper, l2_upper)

    # We can also computer tighter lower bounds due to L2 norm of vectors equal to one.
    l2_lower = v_upper * v_upper
    l2_lower = torch.sqrt(torch.max(torch.zeros_like(l2_lower), 1.0 - (torch.ones_like(l2_lower).matmul(l2_lower) - l2_lower)))
    v_lower = torch.max(v_lower, l2_lower)

    return v_lower, v_upper


def enforce_orthogonality(v_lower, v_upper):
    mu_dict, nu_dict = get_bounds_on_sum(v_lower, v_upper)
    alpha = v_upper
    beta = v_lower
    p = v_lower.shape[0]
    for l in range(p):
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                mu = get_from_dict((i, j, l), mu_dict)
                nu = get_from_dict((i, j, l), nu_dict)
                # case 1
                if alpha[l, j] < 0:
                    if beta[l, i] >= 0:
                        temp_mu = - mu / beta[l, j]
                        temp_nu = - nu / beta[l, j]
                    elif alpha[l, i] < 0:
                        temp_mu = - mu / alpha[l, j]
                        temp_nu = - nu / alpha[l, j]
                    else:
                        temp_mu = torch.min(-mu / alpha[l, j], -mu / beta[l, j])
                        temp_nu = torch.max(-nu / alpha[l, j], -nu / beta[l, j])
                    beta[l, i] = torch.max(beta[l, i], temp_mu)
                    alpha[l, i] = torch.min(alpha[l, i], temp_nu)
                # case 2
                elif beta[l, j] > 0:
                    if beta[l, i] >= 0:
                        temp_nu = - nu / alpha[l, j]
                        temp_mu = - mu / alpha[l, j]
                    elif alpha[l, i] < 0:
                        temp_nu = - nu / beta[l, j]
                        temp_mu = - mu / beta[l, j]
                    else:
                        temp_mu = torch.min(-mu / alpha[l, j], -mu / beta[l, j])
                        temp_nu = torch.max(-nu / alpha[l, j], -nu / beta[l, j])
                    beta[l, i] = torch.max(beta[l, i], temp_nu)
                    alpha[l, i] = torch.min(alpha[l, i], temp_mu)
                # case 3
                elif alpha[l, j] > 0 and beta[l, j] < 0:
                    if beta[l, i] >= 0:
                        temp_mu = - mu / beta[l, j]
                        temp_nu = - nu / alpha[l, j]
                        beta[l, i] = torch.max(torch.tensor([beta[l, i], temp_mu, temp_nu]))
                    elif alpha[l, i] < 0:
                        temp_mu = - mu / alpha[l, j]
                        temp_nu = - nu / beta[l, j]
                        alpha[l, i] = torch.min(torch.tensor([alpha[l, i], temp_mu, temp_nu]))
                # case 4a
                elif beta[l, j] == 0:
                    if alpha[l, j] == 0:
                        continue
                    if mu > 0:
                        temp_up = - mu / alpha[l, j]
                        alpha[l, i] = torch.min(alpha[l, i], temp_up)
                    if nu < 0:
                        temp_low = - nu / alpha[l, j]
                        beta[l, i] = torch.max(beta[l, i], temp_low)
                # case 4b
                elif alpha[l, j] == 0:
                    if beta[l, j] == 0:
                        continue
                    if mu > 0:
                        temp_up = - mu / beta[l, j]
                        alpha[l, i] = torch.min(alpha[l, i], temp_up)
                    if nu < 0:
                        temp_low = - nu / alpha[l, j]
                        beta[l, i] = torch.max(beta[l, i], temp_low)

    return beta, alpha


def get_bounds_on_sum(v_lower, v_upper):
    mu_dict = defaultdict(int)
    nu_dict = defaultdict(int)
    p = v_lower.shape[0]

    for l in range(p):
        for i in range(p - 1):
            for j in range(i + 1, p):
                for k in range(p):
                    if k == l:
                        continue

                    low_low = v_lower[k, i] * v_lower[k, j]
                    low_up = v_lower[k, i] * v_upper[k, j]
                    up_low = v_upper[k, i] * v_lower[k, j]
                    up_up = v_upper[k, i] * v_upper[k, j]

                    mu_dict[(i, j, l)] += torch.min(torch.tensor([low_low, low_up, up_low, up_up]))
                    nu_dict[(i, j, l)] += torch.max(torch.tensor([low_low, low_up, up_low, up_up]))

    return mu_dict, nu_dict


def get_from_dict(key, dictionary):
    i, j, l = key
    if key in dictionary:
        return dictionary[key]
    elif (j, i, l) in dictionary:
        return dictionary[j, i, l]
    else:
        print(f"No element with key {i, j, l} found in dictionary.")


def print_eig_bounds(eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper, eig):
    print(f"Eigvals lower: {eigvals_lower}")
    print(f"Eigvals:       {eig.eigenvalues}")
    print(f"Eigvals upper: {eigvals_upper}")
    print('\n')
    print(f"Eigvects lower: {eigvects_lower}")
    print(f"Eigvects:       {eig.eigenvectors}")
    print(f"Eigvects upper: {eigvects_upper}")
    print('\n')
    print(f"Eigvects lower difference {eig.eigenvectors - eigvects_lower}")
    print(f"Eigvects upper difference {eigvects_upper - eig.eigenvectors}")
    print('\n')


def check_eig_bounds(eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper, eig):
    print("Verifying bounds")
    assert torch.all(eigvals_lower.lt(eig.eigenvalues))
    assert torch.all(eigvals_upper.gt(eig.eigenvalues))
    assert torch.all(eigvects_lower.lt(eig.eigenvectors))
    assert torch.all(eigvects_upper.gt(eig.eigenvectors))
