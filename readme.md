# FRISBHEE - FRIedmann Solver for Black Hole Evaporation in the Early-universe

## Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner       

This package provides the solution of the Friedmann - Boltzmann equations for Primordial Black Holes + SM radiation + BSM Models.
We consider the collapse of density fluctuations as the PBH formation mechanism.
We provide codes for monochromatic and extended mass and spin distributions.


#### Dark Radiation

The main classes in the folder DNeff return the full evolution of the PBH, SM and Dark Radiation comoving energy densities,
together with the evolution of the PBH mass and spin as function of the $\log_{10}$ @ scale factor.
The program "DNeff_Mono.py" assumes a monochromatic distribution, "DNeff_MassDist.py" considers mass distributions only --valid only for Schwarzschild PBHs--, 
and "DNeff_SpinMassDist.py" considers extended distributions in both mass and spin.

The example script "ex_DNeff_Mono.py" containts the final determination of DNeff depending on the model parameters for the monochromatic scenario.
"ex_DNeff_MassDist.py" and "ex_DNeff_SpinMassDist.py" determine DNeff for extended mass and mass & spin distributions, respectively.

#### Dark Matter

The main classes in in the folder Dark_Matter contain the determination the relic abundance in the case of Dark Matter produced from BH evaporation for 
monochromatic and extended distributions.
The program "SolFBEqs_Mono.py" assumes a monochromatic distribution, "SolFBEqsMassDist.py" considers mass distributions only --valid only for Schwarzschild PBHs--, 
and "SolFBEqs_SpinMassDist.py" considers extended distributions in both mass and spin.

"Omega_h2_FI.py" computes the relic abundance for a Freeze-In scenario, together with the DM produced from the evaporation.
This code is only valid for monochromatic distributions.

The scripts "Example_DM_MassDist.py", "Example_DM_SpinMassDist.py" and "Example_FI.py" contain examples on how to use the aforementioned classes. 
The notebooks "Example_Dist.ipynb" and "Example_FI.ipynb" contain the same example as in the python scripts

#### Required Modules

We use Pathos (https://pathos.readthedocs.io/en/latest/pathos.html) for parallelisation, and tqdm (https://pypi.org/project/tqdm/) for progress meter. 
These should be installed in order to FRISBHEE to run.

#### Credits

If using this code, please cite:
- arXiv:2107.00013, arXiv:2107.00016, arXiv:2207.09462, arXiv:2212.03878
