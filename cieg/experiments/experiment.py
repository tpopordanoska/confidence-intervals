import pandas as pd
import requests

from cieg.eigenvectors import *
from cieg.experiments.methods import *
from cieg.experiments.utils import *
from cieg.utils.covariance import cov
from cieg.utils.draw import *


class Experiment:
    def __init__(self, X, name):
        self.X = X
        self.name = name

    @staticmethod
    def load_dataset_if_not_exists(path, url):
        if not os.path.exists(path):
            os.mkdir(path)

        filename = os.path.join(path, os.path.basename(url))
        if not os.path.exists(filename):
            print("Loading dataset")
            data = requests.get(url).content
            with open(filename, "wb") as file:
                file.write(data)

    def run(self, path):
        X = preprocess(self.X)
        X_pd = pd.DataFrame(torch.transpose(X, 0, 1).data.numpy(), dtype='float64')

        sigma = cov(X)
        emp_prec = torch.inverse(sigma)
        eig = get_eig(sigma)

        print("----------- RESAMPLING - bootstrap pmatrix -----------")
        pmatrix_lower, pmatrix_upper = resample_pmatrix(X_pd, n_iterations=100, rng=check_random_state(0))
        print_pmatrix_bounds(pmatrix_lower, pmatrix_upper, emp_prec)
        check_pmatrix_bounds(pmatrix_lower, pmatrix_upper, emp_prec)
        plot_and_save_bounds(pmatrix_lower,
                             pmatrix_upper,
                             emp_prec,
                             f"Bounds on the precision matrix using resampling on {self.name}",
                             path)

        print("----------- OUR METHOD -----------")
        # Bounds on eigendecomposition
        eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper = cieg(X, sigma, eig)
        print_eig_bounds(eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper, eig)
        check_eig_bounds(eigvals_lower, eigvals_upper, eigvects_lower, eigvects_upper, eig)

        # Bounds on precision matrix
        pmatrix_lower, pmatrix_upper, _ = pmatrix_bounds(eigvals_lower,
                                                         eigvals_upper,
                                                         eigvects_lower,
                                                         eigvects_upper,
                                                         sigma, eig)
        print_pmatrix_bounds(pmatrix_lower, pmatrix_upper, emp_prec)
        check_pmatrix_bounds(pmatrix_lower, pmatrix_upper, emp_prec)
        plot_and_save_bounds(pmatrix_lower,
                             pmatrix_upper,
                             emp_prec,
                             f"Bounds on the precision matrix using our method on {self.name}",
                             path)
