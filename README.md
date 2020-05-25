[![DOI](https://zenodo.org/badge/121650199.svg)](https://zenodo.org/badge/latestdoi/121650199)

# TranspirationEstimationAlgorithm
A method of estimating transpiration directly from carbon, water, and energy fluxes, such as eddy covariance datasets.

For a brief introduction of how to use the code, please see the tutorial [here](tutorial.ipynb).

For an extensive overview of how the method works, please see the associated publication:

Nelson, Jacob A., Nuno Carvalhais, Matthias Cuntz, Nicolas Delpierre, Jürgen Knauer, Jérome Ogée, Mirco Migliavacca, Markus Reichstein, and Martin Jung. Coupling Water and Carbon Fluxes to Constrain Estimates of Transpiration: The TEA Algorithm. Journal of Geophysical Research: Biogeosciences, December 21, 2018. https://doi.org/10.1029/2018JG004727.

## Installation

Requirements:
- numpy
- scipy
- xarray
- scikit-learn

Can be installed via anaconda:

conda install -c jnelson18 transpiration_estimation_algorithm

Additionally the ipynb can be run interactively:
- open a terminal
- download both the tutorial.ipynb and TestData into the same directory
- with jupyter installed (conda install jupyter) run:

    jupyter notebook tutorial.ipynb
