# Autoencoder and Topological Data Analysis for Dynamical-System Data

This repository contains a cleaned Python pipeline for analyzing high-dimensional ODE trajectory data using machine learning, dimensionality reduction, topological data analysis, and regression.

## What the code does

The pipeline loads numerical simulation data from a CSV file, selects numerical features, scales the data, and applies complementary analysis methods:

1. Neural autoencoder compression into a two-dimensional or three-dimensional latent space.
2. PCA and t-SNE projections for comparison with the learned latent representation.
3. Vietoris-Rips persistent homology using GUDHI to compute persistence intervals and Betti-number summaries.
4. Mapper graph construction using Isomap, UMAP, DBSCAN, and KeplerMapper.
5. Polynomial-kernel support vector regression to test whether one coordinate or feature can be predicted from the remaining variables.

The goal is to study whether high-dimensional dynamical-system trajectories have lower-dimensional geometric structure, visible latent organization, or persistent topological features such as connected components, loops, and higher-dimensional holes.

## Main file

- `autoencoder_tda_pipeline.py`: Combined and cleaned version of the autoencoder, PCA, t-SNE, Mapper, persistent-homology, Betti-number, and SVR scripts.

## Example usage

```bash
python autoencoder_tda_pipeline.py --csv ode10d_results.csv --feature-mode last --n-features 10 --latent-dim 2 --epochs 100
```

For a three-dimensional latent space:

```bash
python autoencoder_tda_pipeline.py --csv ode10d_results.csv --feature-mode last --n-features 10 --latent-dim 3 --epochs 100
```

## Outputs

The script creates an `outputs/` folder containing reduced embeddings, plots, Betti-number summaries, persistence interval counts, Mapper summaries, and SVR evaluation results when the required packages are installed.

## Requirements

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## References and attribution

This code is project-specific and was written to analyze dynamical-system simulation data. It uses open-source scientific-computing and machine-learning libraries. Please cite the relevant tools when using or adapting this workflow:

- Abadi et al., TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems, arXiv:1603.04467, 2016.
- Pedregosa et al., Scikit-learn: Machine Learning in Python, Journal of Machine Learning Research, 12, 2825-2830, 2011.
- McInnes, Healy, and Melville, UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction, arXiv:1802.03426, 2018.
- van Veen et al., Kepler Mapper: A flexible Python implementation of the Mapper algorithm, Journal of Open Source Software, 4(42), 1315, 2019.
- The GUDHI Project, GUDHI User and Reference Manual, GUDHI Editorial Board.

The repository does not include external library source code. Dependencies should be installed through their official package managers.
