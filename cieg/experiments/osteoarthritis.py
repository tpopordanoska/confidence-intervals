import pandas as pd

from cieg.experiments.utils import *
from .experiment import Experiment


class Osteoarthritis(Experiment):

    def __init__(self, **kwargs):
        cols = kwargs.pop("usecols")
        print(f"Features used: {cols}")

        # Read data
        X_pd = pd.read_csv('data/oai_most_bl_aleksei_sep20_w_dataset_col.csv', usecols=cols)
        if 'Side' in cols:
            X_pd['Side'] = X_pd['Side'].map({"R": 0, "L": 1})
        X_pd = X_pd.dropna()
        X = torch.transpose(torch.tensor(X_pd.values).float(), 0, 1)

        cols = [x.upper() for x in cols]
        super().__init__(X, "oa", column_names=cols)
