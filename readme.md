# FRISBHEE - FRIedmann Solver for Black Hole Evaporation in the Early-universe

## Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner       

This package provides the solution of the Friedmann - Boltzmann equations for Primordial Black Holes + SM radiation + BSM Models.
We consider the collapse of density fluctuations as the PBH formation mechanism.
We provide codes for monochromatic and extended mass and spin distributions.

To run the codes, type:
```
python3 -m folder.file
```
where folder corresponds to the folder where the file is. Note that it is not necessary to type the .py

### SM evolution including determination of Bekenstein-Hawking \mathtt{BH} and von Neumann Hawking radiation entropies

In the folder ```smcase```, we include the program ```soleqs.py``` that computes the evolution of PBH parameter, specifically mass and spin, SM radiation comoving densities together with the evolution of the \mathtt{BH} and von Neumann Hawking radiation entropies. 
The example script to use the main class is called ```ex_sol.py```.
If the user wants to stop the evolution at the Page time, they would need to change the boolean variable to
```
end_evol_at_Pagetime = True
```


### Dark Radiation

The main classes in the folder ```dneff``` return the full evolution of the PBH, SM and Dark Radiation comoving energy densities,
together with the evolution of the PBH mass and spin as function of the $\log_{10}$ @ scale factor.
The program ```mono.py``` assumes a monochromatic distribution, ```massdist.py``` considers mass distributions only --valid only for Schwarzschild PBHs--, and ```spinmassdist.py``` considers extended distributions in both mass and spin.

The example script ```ex_mono.py``` containts the final determination of DNeff depending on the model parameters for the monochromatic scenario.
```ex_massDist.py``` and ```ex_spinmassdist.py``` determine DNeff for extended mass and mass & spin distributions, respectively.

### Dark Matter

The main classes in in the folder ```dm``` contain the determination the relic abundance in the case of Dark Matter produced from BH evaporation for monochromatic and extended distributions.
The program ```mono.py``` assumes a monochromatic distribution, ```massdist.py``` considers mass distributions only --valid only for Schwarzschild PBHs--, and ```spinmassdist.py``` considers extended distributions in both mass and spin.

```freeze_in.py``` computes the relic abundance for a Freeze-In scenario, together with the DM produced from the evaporation.
This code is only valid for monochromatic distributions.

The scripts ```ex_massdist.py```, ```ex_spinmassdist.py``` and ```ex_FI.py``` contain examples on how to use the aforementioned classes. 
The notebooks ```Example_Dist.ipynb``` and ```Example_FI.ipynb``` contain the same example as in the python scripts

#### Required Modules

We use Pathos (https://pathos.readthedocs.io/en/latest/pathos.html) for parallelisation, and tqdm (https://pypi.org/project/tqdm/) for progress meter. 
These should be installed in order to FRISBHEE to run.

#### Credits

If using this code, please cite:
- arXiv:2107.00013, arXiv:2107.00016, arXiv:2207.09462, arXiv:2212.03878
