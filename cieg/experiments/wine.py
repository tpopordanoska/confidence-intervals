from sklearn.datasets import load_wine

from cieg.experiments.utils import *
from .experiment import Experiment


class Wine(Experiment):
    def __init__(self):
        dataset = load_wine()

        X = torch.transpose(torch.tensor(dataset.data).float(), 0, 1)
        super().__init__(X[4:7, :], name="Wine")
