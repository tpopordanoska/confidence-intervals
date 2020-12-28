import pandas as pd
import torch

from .experiment import Experiment


class Census(Experiment):

    def __init__(self, **kwargs):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/census1990-mld/USCensus1990.data.txt"
        self.load_dataset_if_not_exists('data', url=url)
        X_input = pd.read_csv('data/USCensus1990.data.txt', usecols=kwargs.pop('usecols'))

        X = torch.transpose(torch.tensor(X_input.values).float(), 0, 1)
        super().__init__(X, name="Census")
