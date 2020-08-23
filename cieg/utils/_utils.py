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