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


import torch


def indice_for_all_cases(ind_ijkl=None):
    """
    Returns the indices for all the 7 cases of cov-cov
    Args:
        ind_ijkl:

    Returns:

    """
    # indice_for_all_cases
    # %%%%% CASE 1:  i \ne j,k,l$; $j \ne k,l$; $k \ne l$
    ind_c1 = ((ind_ijkl[:, 0] != ind_ijkl[:, 1]) * (ind_ijkl[:, 0] != ind_ijkl[:, 2]) * (
            ind_ijkl[:, 0] != ind_ijkl[:, 3]) * (ind_ijkl[:, 1] != ind_ijkl[:, 2]) * (
                      ind_ijkl[:, 1] != ind_ijkl[:, 3]) * (ind_ijkl[:, 2] != ind_ijkl[:, 3])).nonzero()

    # %%%%% CASE 2:  $i=j$; $j \ne k,l$; $k = l$
    ind_c2 = ((ind_ijkl[:, 0] == ind_ijkl[:, 1]) * (ind_ijkl[:, 1] != ind_ijkl[:, 2]) * (
            ind_ijkl[:, 1] != ind_ijkl[:, 3]) * (ind_ijkl[:, 2] == ind_ijkl[:, 3])).nonzero()

    # %%%%% CASE 3: i=j$; $i \ne k,l $; $j \ne k,l$; $k \ne l$
    # % ou par sym; k==l; $k \ne i,j $; $l \ne i,j$; $i \ne j$
    ind_c3_1 = (ind_ijkl[:, 0] == ind_ijkl[:, 1]) * (ind_ijkl[:, 1] != ind_ijkl[:, 2]) * (
            ind_ijkl[:, 1] != ind_ijkl[:, 3]) * (ind_ijkl[:, 2] != ind_ijkl[:, 3])
    ind_c3_2 = (ind_ijkl[:, 2] == ind_ijkl[:, 3]) * (ind_ijkl[:, 2] != ind_ijkl[:, 0]) * (
            ind_ijkl[:, 2] != ind_ijkl[:, 1]) * (ind_ijkl[:, 0] != ind_ijkl[:, 1])
    ind_c3 = torch.logical_or(ind_c3_1, ind_c3_2).nonzero()

    # %%%%% CASE 4: NOK $i=k$; $j \ne i,k,l$; $k \ne l$
    ind_c4_1 = (ind_ijkl[:, 0] == ind_ijkl[:, 2]) * (ind_ijkl[:, 1] != ind_ijkl[:, 0]) * (
            ind_ijkl[:, 1] != ind_ijkl[:, 2]) * (ind_ijkl[:, 1] != ind_ijkl[:, 3]) * (
                       ind_ijkl[:, 2] != ind_ijkl[:, 3])
    ind_c4_2 = (ind_ijkl[:, 0] == ind_ijkl[:, 3]) * (ind_ijkl[:, 1] != ind_ijkl[:, 0]) * (
            ind_ijkl[:, 1] != ind_ijkl[:, 2]) * (ind_ijkl[:, 1] != ind_ijkl[:, 3]) * (
                       ind_ijkl[:, 2] != ind_ijkl[:, 3])
    ind_c4_3 = (ind_ijkl[:, 1] == ind_ijkl[:, 2]) * (ind_ijkl[:, 0] != ind_ijkl[:, 1]) * (
            ind_ijkl[:, 0] != ind_ijkl[:, 2]) * (ind_ijkl[:, 0] != ind_ijkl[:, 3]) * (
                       ind_ijkl[:, 2] != ind_ijkl[:, 3])
    ind_c4_4 = (ind_ijkl[:, 1] == ind_ijkl[:, 3]) * (ind_ijkl[:, 0] != ind_ijkl[:, 1]) * (
            ind_ijkl[:, 0] != ind_ijkl[:, 2]) * (ind_ijkl[:, 0] != ind_ijkl[:, 3]) * (
                       ind_ijkl[:, 2] != ind_ijkl[:, 3])
    ind_c4 = torch.logical_or(torch.logical_or(ind_c4_1, ind_c4_2), torch.logical_or(ind_c4_3, ind_c4_4)).nonzero()

    # %%%%% CASE 5  $i=k$; $i \ne j$; $j=l$
    ind_c5_1 = (ind_ijkl[:, 0] == ind_ijkl[:, 2]) * (ind_ijkl[:, 0] != ind_ijkl[:, 1]) * (
            ind_ijkl[:, 1] == ind_ijkl[:, 3])
    ind_c5_2 = (ind_ijkl[:, 0] == ind_ijkl[:, 3]) * (ind_ijkl[:, 0] != ind_ijkl[:, 1]) * (
            ind_ijkl[:, 1] == ind_ijkl[:, 2])
    ind_c5 = torch.logical_or(ind_c5_1, ind_c5_2).nonzero()

    # %%%%% CASE 6: $i=j=k$; $i \ne l$
    ind_c6_1 = (ind_ijkl[:, 0] == ind_ijkl[:, 1]) * (ind_ijkl[:, 0] == ind_ijkl[:, 2]) * (
            ind_ijkl[:, 0] != ind_ijkl[:, 3])
    ind_c6_2 = (ind_ijkl[:, 0] == ind_ijkl[:, 1]) * (ind_ijkl[:, 0] == ind_ijkl[:, 3]) * (
            ind_ijkl[:, 0] != ind_ijkl[:, 2])
    ind_c6_3 = (ind_ijkl[:, 2] == ind_ijkl[:, 0]) * (ind_ijkl[:, 2] == ind_ijkl[:, 3]) * (
            ind_ijkl[:, 2] != ind_ijkl[:, 1])
    ind_c6_4 = (ind_ijkl[:, 2] == ind_ijkl[:, 1]) * (ind_ijkl[:, 2] == ind_ijkl[:, 3]) * (
            ind_ijkl[:, 2] != ind_ijkl[:, 0])
    ind_c6 = torch.logical_or(torch.logical_or(ind_c6_1, ind_c6_2), torch.logical_or(ind_c6_3, ind_c6_4)).nonzero()

    # %%%%% CASE 7 :  $i=j,k,l$
    ind_c7 = ((ind_ijkl[:, 0] == ind_ijkl[:, 1]) * (ind_ijkl[:, 0] == ind_ijkl[:, 2]) * (
            ind_ijkl[:, 0] == ind_ijkl[:, 3])).nonzero()

    return ind_c1, ind_c2, ind_c3, ind_c4, ind_c5, ind_c6, ind_c7


def matching_indices(p=None):
    """
    ind_uptri indice of the upper triang
    ind_ijkl is a matrix with ((p*(p+1))/2)*((p*(p+1))/2) rows and 4 columns of the indice of ijkl
    ind_qr is a matrix matrix ((p*(p+1))/2 rows and 2 colums of indice of the upper triangular
    """

    ind_uptri = torch.empty(0, 2)
    rows = torch.ger(torch.arange(1., p + 1), torch.ones(p))
    cols = torch.transpose(rows, 0, 1)
    for i in range(p):
        ind_uptri = torch.cat([ind_uptri, torch.transpose(torch.stack([rows[i, i:p], cols[i, i:p]]), 0, 1)], 0)

    ind_ijkl = torch.empty(0, 4)
    ind_qr = torch.empty(0, 2)

    for q in range(1, ind_uptri.size(0) + 1):
        for r in range(q, ind_uptri.size(0) + 1):
            ind_ijkl = torch.cat([ind_ijkl, torch.cat([ind_uptri[q - 1, :], ind_uptri[r - 1, :]], 0).unsqueeze(0)], 0)
            ind_qr = torch.cat([ind_qr, torch.FloatTensor([q, r]).unsqueeze(0)], 0)

    return ind_uptri, ind_ijkl, ind_qr
