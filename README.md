# Distribution-Independent Confidence intervals for the Eigendecomposition of Covariance Matrices

This is the code that accompanies the paper [Distribution-Independent Confidence Intervals for the Eigendecomposition of Covariance Matrices via the Eigenvalue-Eigenvector Identity](https://lirias.kuleuven.be/handle/123456789/679096), accepted at [ICML 2021 Workshop on Distribution-Free Uncertainty Quantification](https://sites.google.com/berkeley.edu/dfuq21/home?authuser=0).

## Installation

```
conda env create -f conda_env_<YOUR_OS>.yml
conda activate confidence_intervals
pip install -e .
```

## Experiments 

To run the experiments, use the run.py script in cieg/experiments. 


## Reference
If you found this code useful, please cite:

```tex
@incollection{Popordanoska2021b,
year={2021},
booktitle={ICML Workshop on Distribution Free Uncertainty Quantification},
title={Distribution-Independent Confidence Intervals for the Eigendecomposition of Covariance Matrices via the Eigenvalue-Eigenvector Identity},
author={Teodora Popordanoska and Aleksei Tiulpin and Wacha Bounliphone and Blaschko, Matthew B.},
}
```

## License

Everything is licensed under the [MIT License](https://opensource.org/licenses/MIT).


## Acknowledgements

We acknowledge funding from the Flemish Government under the "Onderzoeksprogramma Artificiële Intelligentie (AI) Vlaanderen" programme. 
