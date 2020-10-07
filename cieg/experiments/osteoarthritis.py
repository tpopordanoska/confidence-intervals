import warnings

import pandas as pd

from cieg.eigenvectors import *
from cieg.experiments.methods import *
from cieg.experiments.utils import *
from cieg.utils import *
from cieg.utils.covariance import cov

warnings.filterwarnings(
    "ignore", category=UserWarning
)

path = create_folders()

columns_to_read = ['Side', 'WOMAC', 'OSTM']  # OSTM OSFM
print(f"Features used: {columns_to_read}")

# Read data
X_pd = pd.read_csv('../data/oai_most_bl_aleksei_sep20_w_dataset_col.csv', usecols=columns_to_read)
if 'Side' in columns_to_read:
    X_pd['Side'] = X_pd['Side'].map({"R": 0, "L": 1})
X_pd = X_pd.dropna()
X = torch.transpose(torch.tensor(X_pd.values).float(), 0, 1)
print(f"X shape: {X.shape}")

X = preprocess(X)
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
                     "Osteoarthritis bounds on the precision matrix using resampling",
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
                     "Osteoarthritis bounds on the precision matrix using our method",
                     path)
