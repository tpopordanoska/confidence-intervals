import torch
from sklearn.preprocessing import StandardScaler


def preprocess(X):
    stds = torch.std(X, 1)
    stds[stds == 0] = 1
    stds = 1. / stds
    X = torch.diag(stds).matmul(X)
    p = X.size(0)
    d = torch.diag(torch.arange(1, p + 1)) / (0.5 * p)
    X = d.matmul(X)

    return X


def reverse(lower, upper, X):
    p = X.size(0)
    d = torch.diag(torch.arange(1, p + 1)) / (0.5 * p)
    stds = torch.std(X, 1)
    stds[stds == 0] = 1
    stds = 1. / stds

    lower = torch.inverse(d).matmul(lower.float())
    lower = torch.inverse(torch.diag(stds)).matmul(lower)
    upper = torch.inverse(d).matmul(upper.float())
    upper = torch.inverse(torch.diag(stds)).matmul(upper)

    return lower, upper


# Only to check the logic
def reverse_check(X_processed, X_original):
    p = X_original.size(0)
    d = torch.diag(torch.arange(1, p + 1)) / (0.5 * p)
    X_old = torch.inverse(d).matmul(X_processed)

    stds = torch.std(X_original, 1)
    stds[stds == 0] = 1
    stds = 1. / stds

    X_reversed = torch.inverse(torch.diag(stds)).matmul(X_old)

    print(X_reversed == X_original)


def standard_scale(X):
    scaler = StandardScaler()
    scaler.fit(X).transform(X)

    return X
