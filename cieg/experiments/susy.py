import pandas as pd
import torch

from .experiment import Experiment


class Susy(Experiment):

    def __init__(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00279/SUSY.csv.gz"
        self.load_dataset_if_not_exists('data', url=url)
        dataset = pd.read_csv("data/SUSY.csv.gz", compression='gzip', delimiter=',', header=None)

        X = torch.transpose(torch.tensor(dataset.values).float(), 0, 1)
        super().__init__(X[:4, :], name="Susy")
