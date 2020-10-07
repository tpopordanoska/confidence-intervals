import pandas as pd

from cieg.utils import *
from .experiment import Experiment


class Osteoarthritis(Experiment):

    def __init__(self, **kwargs):
        columns_to_read = kwargs.pop("usecols")
        print(f"Features used: {columns_to_read}")

        # Read data
        X_pd = pd.read_csv('data/oai_most_bl_aleksei_sep20_w_dataset_col.csv', usecols=columns_to_read)
        if 'Side' in columns_to_read:
            X_pd['Side'] = X_pd['Side'].map({"R": 0, "L": 1})
        X_pd = X_pd.dropna()
        X = torch.transpose(torch.tensor(X_pd.values).float(), 0, 1)

        super().__init__(X, "Osteoarthritis")
