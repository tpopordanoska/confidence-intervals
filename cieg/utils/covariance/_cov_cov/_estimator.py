# Author: Wacha Bounliphone - wacha.bounliphone@centralesupelec.fr
# Code refactoring: Aleksei Tiulpin, Teodora Popordanoska 2020
# Copyright (c) 2016
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# If you use this software in your research, please cite:%
# Bounliphone, W. &  Blaschko, M.B. (2016).
# A U-statistic Approach to Hypothesis Testing for Structure Discovery in
# Undirected Graphical Models
#
# -------------------------------------------------------------------
# covCov  is the full covariance between the elements of \hat_{Sigma}
# covCov = cov[\hat_{Sigma}]
# -------------------------------------------------------------------

import torch

from ._cov_cov_cases import case1, case2, case3, case4, case5, case6, case7
from ._indices import indice_for_all_cases, matching_indices


def covCov_estimator(X=None):
    ind_uptri, ind_ijkl, ind_qr = matching_indices(X.size(0))
    ind_c1, ind_c2, ind_c3, ind_c4, ind_c5, ind_c6, ind_c7 = indice_for_all_cases(ind_ijkl)

    ind_ijkl = ind_ijkl.long() - 1
    covCovTheo = torch.zeros(ind_ijkl.size(0))

    # %%%%% CASE 1:
    for indice in range(ind_c1.size(0)):
        Xi = X[ind_ijkl[ind_c1[indice]][0][0], :]
        Xj = X[ind_ijkl[ind_c1[indice]][0][1], :]
        Xk = X[ind_ijkl[ind_c1[indice]][0][2], :]
        Xl = X[ind_ijkl[ind_c1[indice]][0][3], :]
        covCovTheo[ind_c1[indice]] = case1(Xi, Xj, Xk, Xl)

    # %%%%% CASE 2:
    for indice in range(ind_c2.size(0)):
        case2_a = torch.unique(ind_ijkl[ind_c2[indice]]).squeeze()
        assert case2_a.size(0) == 2
        case2_l = torch.zeros(2)
        case2_l[0] = torch.sum(ind_ijkl[ind_c2[indice]] == case2_a[0])
        case2_l[1] = torch.sum(ind_ijkl[ind_c2[indice]] == case2_a[1])
        assert torch.max(case2_l) == 2
        Xi = X[case2_a[0], :]
        Xk = X[case2_a[1], :]
        covCovTheo[ind_c2[indice]] = case2(Xi, Xk)

    # %%%%% CASE 3:
    for indice in range(ind_c3.size(0)):
        case3_a = torch.unique(ind_ijkl[ind_c3[indice]]).squeeze()
        assert case3_a.size(0) == 3
        case3_l = torch.zeros(3)
        case3_l[0] = torch.sum(ind_ijkl[ind_c3[indice]] == case3_a[0])
        case3_l[1] = torch.sum(ind_ijkl[ind_c3[indice]] == case3_a[1])
        case3_l[2] = torch.sum(ind_ijkl[ind_c3[indice]] == case3_a[2])
        assert torch.max(case3_l) == 2
        case3_ind = torch.zeros(3)
        case3_ind[0] = case3_a[case3_l == 2]
        case3_ind[1:] = case3_a[(case3_l == 1).nonzero().squeeze()]
        case3_ind = case3_ind.long()
        Xi = X[case3_ind[0], :]
        Xk = X[case3_ind[1], :]
        Xl = X[case3_ind[2], :]
        covCovTheo[ind_c3[indice]] = case3(Xi, Xk, Xl)

    # %%%%% CASE 4:
    for indice in range(ind_c4.size(0)):
        case4_a = torch.unique(ind_ijkl[ind_c4[indice]]).squeeze()
        assert case4_a.size(0) == 3
        case4_l = torch.zeros(3)
        case4_l[0] = torch.sum(ind_ijkl[ind_c4[indice]] == case4_a[0])
        case4_l[1] = torch.sum(ind_ijkl[ind_c4[indice]] == case4_a[1])
        case4_l[2] = torch.sum(ind_ijkl[ind_c4[indice]] == case4_a[2])
        assert torch.max(case3_l) == 2;
        case4_ind = torch.zeros(3);
        case4_ind[0] = case4_a[case4_l == 2];
        case4_ind[1:] = case4_a[(case4_l == 1).nonzero().squeeze()];
        case4_ind = case4_ind.long()
        Xi = X[case4_ind[0], :]
        Xj = X[case4_ind[1], :]
        Xl = X[case4_ind[2], :]
        covCovTheo[ind_c4[indice]] = case4(Xi, Xj, Xl)

    # %%%%% CASE 5
    for indice in range(ind_c5.size(0)):
        Xi = X[ind_ijkl[ind_c5[indice], 0], :].squeeze()
        Xj = X[ind_ijkl[ind_c5[indice], 1], :].squeeze()
        covCovTheo[ind_c5[indice]] = case5(Xi, Xj)

    # %%%%% CASE 6
    for indice in range(ind_c6.size(0)):
        case6_a = torch.unique(ind_ijkl[ind_c6[indice]]).squeeze()
        assert case6_a.size(0) == 2
        case6_l = torch.zeros(2)
        case6_l[0] = torch.sum(ind_ijkl[ind_c6[indice], :] == case6_a[0])
        case6_l[1] = torch.sum(ind_ijkl[ind_c6[indice], :] == case6_a[1])
        assert torch.max(case6_l) == 3
        case6_ind = torch.zeros(2)
        case6_ind[0] = case6_a[case6_l == 3]
        case6_ind[1] = case6_a[case6_l == 1]
        case6_ind = case6_ind.long()
        Xi = X[case6_ind[0], :]
        Xl = X[case6_ind[1], :]
        covCovTheo[ind_c6[indice]] = case6(Xi, Xl)

    # %%%%% CASE 7
    for indice in range(ind_c7.size(0)):
        Xi = X[ind_ijkl[ind_c7[indice], 0], :].squeeze()
        covCovTheo[ind_c7[indice]] = case7(Xi)

    n = X.size(0)
    # transforme into a matrix
    covCovSize = torch.max(torch.max(ind_qr).squeeze()).squeeze()
    covCovSize = covCovSize.int()
    covCov = torch.zeros(covCovSize, covCovSize)
    ind_qr = ind_qr - 1
    for k in range(ind_qr.size(0)):
        i = ind_qr[k][0].int()
        j = ind_qr[k][1].int()
        covCov[i][j] = covCovTheo[k]
        covCov[j][i] = covCovTheo[k]

    # assert torch.min(torch.diag(covCov)) >= 0

    #    # should be a positive definite matrix
    #    assert torch.min(torch.eig(covCov).eigenvalues()) >= 0
    #    # trace > max eigenvalue
    #    thetrace = torch.trace(covCov)
    #    max_l_covcov = torch.max(torch.eig(covCov).eigenvalues())
    #    assert thetrace >= max_l_covcov

    return covCov
