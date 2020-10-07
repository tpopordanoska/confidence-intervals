import pandas as pd
import torch

from .experiment import Experiment


class Higgs(Experiment):

    def __init__(self, **kwargs):
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
        columns = ['class', 'lepton pT', 'lepton eta', 'lepton phi', 'missing energy magnitude', 'missing energy phi',
                   'jet 1 pt', 'jet 1 eta', 'jet 1 phi', 'jet 1 b-tag', 'jet 2 pt', 'jet 2 eta', 'jet 2 phi',
                   'jet 2 b-tag', 'jet 3 pt', 'jet 3 eta', 'jet 3 phi', 'jet 3 b-tag', 'jet 4 pt', 'jet 4 eta',
                   'jet 4 phi', 'jet 4 b-tag', 'm_jj', 'm_jjj', 'm_lv', 'm_jlv', 'm_bb', 'm_wbb', 'm_wwbb']

        self.load_dataset_if_not_exists('data', url=url)
        dataset = pd.read_csv("data/HIGGS.csv.gz",
                              compression='gzip',
                              delimiter=',',
                              header=None,
                              names=columns,
                              usecols=kwargs.pop('usecols'))

        X = torch.transpose(torch.tensor(dataset.values).float(), 0, 1)
        super().__init__(X, kwargs.pop("name"))
