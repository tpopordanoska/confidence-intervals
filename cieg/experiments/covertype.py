import pandas as pd
import torch

from .experiment import Experiment


class Covertype(Experiment):

    def __init__(self):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz"
        self.load_dataset_if_not_exists('data', url=url)
        # See covtype.info for description of the attributes
        dataset = pd.read_csv("data/covtype.data.gz", compression='gzip', delimiter=',', header=None)
        X = torch.transpose(torch.tensor(dataset.values).float(), 0, 1)

        super().__init__(X[4:8, :], name="Covertype")
