# Python translation: Matthew Blaschko 2020
# Code refactoring: Aleksei Tiulpin, Teodora Popordanoska 2020
# Author: Wacha Bounliphone - wacha.bounliphone@centralesupelec.fr
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

import scipy
import torch


def case1(Xi=None, Xj=None, Xk=None, Xl=None):
    n = Xi.size(0)

    # zeta1

    t1_1 = torch.mean(Xi * Xj * Xk * Xl)
    t1_2 = -torch.mean(Xi) * torch.mean(Xj * Xk * Xl)
    t1_3 = -torch.mean(Xj) * torch.mean(Xi * Xk * Xl)
    t1_4 = -torch.mean(Xk) * torch.mean(Xi * Xj * Xl)
    t1_5 = torch.mean(Xi) * torch.mean(Xk) * torch.mean(Xj * Xl)
    t1_6 = torch.mean(Xj) * torch.mean(Xk) * torch.mean(Xi * Xl)
    t1_7 = -torch.mean(Xi * Xj * Xk) * torch.mean(Xl)
    t1_8 = torch.mean(Xi) * torch.mean(Xl) * torch.mean(Xj * Xk)
    t1_9 = torch.mean(Xj) * torch.mean(Xl) * torch.mean(Xi * Xk)

    t2_1 = (torch.mean(Xi * Xj) - 2 * torch.mean(Xi) * torch.mean(Xj))
    t2_2 = (torch.mean(Xk * Xl) - 2 * torch.mean(Xk) * torch.mean(Xl))

    zeta1 = (t1_1 + t1_2 + t1_3 + t1_4 + t1_5 + t1_6 + t1_7 + t1_8 + t1_9) - t2_1 * t2_2
    zeta1 = (1. / 4.) * zeta1

    # final term
    constante = 1. / scipy.special.binom(n, 2)
    mycovariance = constante * (2. * (n - 2.) * zeta1)

    return mycovariance


def case2(Xi=None, Xk=None):
    n = Xi.size(0)

    t1_1 = torch.mean(Xi * Xi * Xk * Xk)
    t1_2 = -2. * torch.mean(Xi * Xk * Xk) * torch.mean(Xi)
    t1_3 = -2. * torch.mean(Xi * Xi * Xk) * torch.mean(Xk)
    t1_4 = 4. * torch.mean(Xi * Xk) * torch.mean(Xi) * torch.mean(Xk)

    t2_1 = torch.mean(Xi * Xi) - 2. * (torch.mean(Xi) ** 2)
    t2_2 = torch.mean(Xk * Xk) - 2. * (torch.mean(Xk) ** 2)

    zeta1 = (t1_1 + t1_2 + t1_3 + t1_4) - t2_1 * t2_2
    zeta1 = (1. / 4.) * zeta1

    # final_term
    constante = 1. / scipy.special.binom(n, 2)
    mycovariance = (constante * (2. * (n - 2.) * zeta1))

    return mycovariance


def case3(Xi=None, Xk=None, Xl=None):
    n = Xi.size(0)

    # zeta1
    zeta1_t1 = torch.mean(Xi * Xi * Xk * Xl)
    zeta1_t2 = -2. * torch.mean(Xi) * torch.mean(Xi * Xk * Xl)
    zeta1_t3 = -torch.mean(Xk) * torch.mean(Xi * Xi * Xl)
    zeta1_t4 = 2. * torch.mean(Xi * Xl) * torch.mean(Xi) * torch.mean(Xk)
    zeta1_t5 = -torch.mean(Xi * Xi * Xk) * torch.mean(Xl)
    zeta1_t6 = 2. * torch.mean(Xi * Xk) * torch.mean(Xi) * torch.mean(Xl)

    zeta1_t7 = torch.mean(Xi * Xi) - 2. * (torch.mean(Xi) ** 2)
    zeta1_t8 = torch.mean(Xk * Xl) - 2. * torch.mean(Xk) * torch.mean(Xl)

    zeta1 = zeta1_t1 + zeta1_t2 + zeta1_t3 + zeta1_t4 + zeta1_t5 + zeta1_t6 - (zeta1_t7 * zeta1_t8)
    zeta1 = (1. / 4.) * zeta1

    # final_term
    constante = 1. / scipy.special.binom(n, 2)
    mycovariance = (constante * (2. * (n - 2.) * zeta1))

    return mycovariance


def case4(Xi=None, Xj=None, Xl=None):
    n = Xi.size(0)

    # zeta1

    t1_1 = torch.mean(Xi * Xi * Xj * Xl)
    t1_2 = -torch.mean(Xi) * torch.mean(Xj * Xi * Xl)
    t1_3 = -torch.mean(Xj) * torch.mean(Xi * Xi * Xl)
    t1_4 = -torch.mean(Xi) * torch.mean(Xi * Xj * Xl)
    t1_5 = (torch.mean(Xi) ** 2) * torch.mean(Xj * Xl)
    t1_6 = torch.mean(Xj) * torch.mean(Xi) * torch.mean(Xi * Xl)
    t1_7 = -torch.mean(Xi * Xi * Xj) * torch.mean(Xl)
    t1_8 = torch.mean(Xi) * torch.mean(Xl) * torch.mean(Xj * Xi)
    t1_9 = torch.mean(Xj) * torch.mean(Xl) * torch.mean(Xi * Xi)

    t2_1 = (torch.mean(Xi * Xj) - 2. * torch.mean(Xi) * torch.mean(Xj))
    t2_2 = (torch.mean(Xi * Xl) - 2. * torch.mean(Xi) * torch.mean(Xl))

    zeta1 = (t1_1 + t1_2 + t1_3 + t1_4 + t1_5 + t1_6 + t1_7 + t1_8 + t1_9) - t2_1 * t2_2
    zeta1 = (1. / 4.) * zeta1

    # final_term
    constante = 1. / scipy.special.binom(n, 2)
    mycovariance = (constante * (2. * (n - 2.) * zeta1))
    return mycovariance


def case5(X=None, Y=None):
    n = X.size(0)
    meanX = torch.mean(X)
    meanY = torch.mean(Y)

    # zeta1
    # E[(h_1)^2]
    zeta1_t1 = torch.mean(X * Y * X * Y)
    zeta1_t2 = -2. * meanX * torch.mean(X * Y * Y)
    zeta1_t3 = (meanX ** 2) * torch.mean(Y * Y)
    zeta1_t4 = -2. * meanY * torch.mean(X * X * Y)
    zeta1_t5 = 2. * meanX * meanY * torch.mean(X * Y)
    zeta1_t6 = (meanY ** 2) * torch.mean(X * X)

    # E[(h_1)]^2
    zeta1_t7 = torch.mean(X * Y)
    zeta1_t8 = -2. * meanY * meanX

    zeta1 = zeta1_t1 + zeta1_t2 + zeta1_t3 + zeta1_t4 + zeta1_t5 + zeta1_t6 - ((zeta1_t7 + zeta1_t8) ** 2)
    zeta1 = (1. / 4.) * zeta1

    constante = 1. / scipy.special.binom(n, 2)
    mycovariance = constante * (2. * (n - 2.) * zeta1)

    return mycovariance


def case6(Xi=None, Xl=None):
    n = Xi.size(0)

    # zeta1
    t1_1 = torch.mean(Xi * Xi * Xi * Xl)
    t1_2 = -3. * torch.mean(Xi) * torch.mean(Xi * Xi * Xl)
    t1_3 = 2. * torch.mean(Xi * Xl) * (torch.mean(Xi) ** 2)
    t1_4 = -torch.mean(Xl) * torch.mean(Xi * Xi * Xi)
    t1_5 = 2. * torch.mean(Xi * Xi) * torch.mean(Xi) * torch.mean(Xl)

    t2_1 = torch.mean(Xi * Xi) - 2. * (torch.mean(Xi) ** 2)
    t2_2 = torch.mean(Xi * Xl) - 2. * torch.mean(Xi) * torch.mean(Xl)

    zeta1 = (t1_1 + t1_2 + t1_3 + t1_4 + t1_5) - t2_1 * t2_2
    zeta1 = (1. / 4.) * zeta1

    # final_term
    constante = 1. / scipy.special.binom(n, 2)
    mycovariance = (constante * (2. * (n - 2.) * zeta1))
    return mycovariance


def case7(Xi=None):
    n = Xi.size(0)

    zeta1_t1 = torch.mean(Xi * Xi * Xi * Xi)
    zeta1_t2 = -4. * torch.mean(Xi) * torch.mean(Xi * Xi * Xi)
    zeta1_t3 = 4. * torch.mean(Xi * Xi) * (torch.mean(Xi) ** 2)
    zeta1_t4 = torch.mean(Xi * Xi) - 2. * (torch.mean(Xi) ** 2)

    zeta1 = zeta1_t1 + zeta1_t2 + zeta1_t3 - (zeta1_t4 ** 2)
    zeta1 = (1. / 4.) * zeta1

    constante = 1. / scipy.special.binom(n, 2)
    mycovariance = constante * (2. * (n - 2.) * zeta1)

    return mycovariance
