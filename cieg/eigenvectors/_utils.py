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

    return v_lower, v_upper


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


def check_bounds(eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper, eig):
    print("Verifying bounds")
    assert torch.all(eigvals_lower.lt(eig.eigenvalues))
    assert torch.all(eigvals_upper.gt(eig.eigenvalues))
    assert torch.all(eigvects_lower.lt(eig.eigenvectors))
    assert torch.all(eigvects_upper.gt(eig.eigenvectors))
