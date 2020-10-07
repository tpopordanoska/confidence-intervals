import os
from datetime import datetime as dt

import torch


def minor(m: torch.Tensor, i: int) -> torch.Tensor:
    """Construct a matrix minor of m by deleting the ith row and column.
    (c) Matthew Blaschko, 2020

    Args:
        m (torch.Tensor): Matrix
        i (int): Index of the minor

    Returns:
        result (torch.Tensor): The Minor
    """
    inds = torch.arange(m.size()[0])
    inds = torch.cat([inds[:i], inds[i+1:]])
    return torch.index_select(torch.index_select(m, 0, inds), 1, inds)


def create_folders():
    """
    Create folder results if it doesn't exist and create another folder for the current experiment

    :return: The path to the folder where the results from the current experiment will be saved.
    """
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
