# FRISBHEE - FRIedmann Solver for Black Hole Evaporation in the Early-universe

## Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner       

This package provides the solution of the Friedmann - Boltzmann equations for Primordial Black Holes + SM radiation + BSM Models.
We consider the collapse of density fluctuations as the PBH formation mechanism.

#### Dark Radiation

The main class in "SolFBEqs_DR.py" returns the full evolution of the PBH, SM and Dark Radiation comoving energy densities,
together with the evolution of the PBH mass and spin as function of the $\log_{10}$ @ scale factor.

The example script "DNeff.py" containts the final determination of DNeff depending on the model parameters.

#### Dark Matter

The main class in "Omega_h2_onlyDM.py" returns the relic abundance in the case of a purely-interacting Dark Matter produced from BH evaporation.
"Omega_h2_FI.py" computes the relic abundance for a Freeze-In scenario, together with the DM produced from the evaporation.

The scripts "Example_onlyDM.py" and "Example_FI.py" contain examples on how to use the aforementioned classes. 
The notebooks "Example_onlyDM.ipynb" and "Example_FI.ipynb" contain the same example as in the python scripts

If using this code, please cite:
- arXiv:2107.00013, arXiv:2107.00016, arXiv:2207.09462
