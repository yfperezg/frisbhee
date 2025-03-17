###################################################################################################
#                                                                                                 #
#                               Primordial Black Hole Evaporation                                 #
#                         Dark Matter Production from Hawking Radiation                           #
#                                 Considering Mass Distributions                                  #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                                   Based on: arXiv:2212.XXXXX                                    #
#                                                                                                 #
###################################################################################################

#======================================================#
#                                                      #
#                     Example script                   #  
#                                                      #
#======================================================#

import sys, os
path, filename = os.path.split(os.path.realpath(__file__))
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta
from scipy.special import kn
import mpmath
from mpmath import polylog
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.optimize import root, toms748

from src import bhprop as bh

from dm.mono import FBEqs_Sol                   # Monochromatic Scenario

import time

#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

asi  = 0.  # PBH initial rotation a_star factor
mDM  = -1. # Log10 @ Dark Matter mass in GeV
sDM  = 2.  # Dark Matter spin

Dic_sDR = {0.:'scl', 0.5:'fer', 1.:'vec', 2.:'gra'}

#------------------------------------------------------------------------------------------------------#
#          We call the solver, and save the arrays containing the full evolution of the PBH,           #
#    SM and DR comoving energy densities, together with the evolution of the PBH mass and spin         #
#                              as function of the log_10 @ scale factor.                               #
#------------------------------------------------------------------------------------------------------#

#+++++++++++++++++++++++++++++#
#        Monochromatic        #
#+++++++++++++++++++++++++++++#

start = time.time()

MBH_arr = []
bra_arr = []

Mi = -1.0 # Initial Log10@ PBH mass, in g
DM = 0.2  # Step on Log10@PBH mass
Db = 1.0  # Step on beta at formation

while Mi < 9.:
        
    bi = -25. # Initial value of beta at formation
    
    bi_arr  = np.array([])
    Oh2_arr = np.array([])
    
    Oh2_val = 0.
    
    while Oh2_val <= 1.:
        
        #+++++++++++++++++++++++++++++++++++++++++#
        #             Monochromatic case          #
        #+++++++++++++++++++++++++++++++++++++++++#
        
        Oh2 = FBEqs_Sol(Mi, asi, bi, mDM, sDM)

        x, t, MBH, ast, SBH, SRad, Rad, PBH, TUn, NDBE, Tev  = Oh2.Solt()
        
        nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3             # Initial photon number density
        
        rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3
        
        T0 = 2.34865e-13  # Temperature today in GeV
        
        Oh2_val  = NDBE[-1] * nphi * 10.**(-3.*x[-1]) * 10.**mDM * (bh.gstarS(T0)/bh.gstarS(Tev))*(T0/Tev)**3*(1/rc)

        bc = np.log10(Tev/TUn[0])
        
        print(bi, Oh2_val, bc)
        
        bi_arr = np.append(bi_arr, bi)
        Oh2_arr = np.append(Oh2_arr, Oh2_val)
        
        if Oh2_val <= 1.e-10:
            bi += 4.*Db
        elif Oh2_val <= 1.e-5:
            bi += 2.*Db
        elif Oh2_val <= 1.e-2:
            bi += Db
        else:
            bi += Db/10.
            
        if bi >= bc + 2.:
            break

    if Oh2_arr[-1] > 0.12:

        Oh2int = interp1d(bi_arr, Oh2_arr)
            
        def bi_r(bi): return 0.12 - Oh2int(bi)
        
        bi_ra = toms748(bi_r, bi_arr[0], bi_arr[-1])
        print(Mi, bi_ra, bc)

        MBH_arr = np.append(MBH_arr, Mi)
        bra_arr = np.append(bra_arr, bi_ra)
        
        if bi_ra < bc - 1.:
            Mi += DM
        else:
            Mi += DM/10.
        
    else:
        break

#===========================#
#           Plots           #
#===========================#

fig, ax = plt.subplots(figsize=(6.5,6.25), sharex=True, layout='constrained')

ax.plot(MBH_arr, bra_arr)
ax.set_ylim(-25, -4)
ax.set_xlim(-1, 9)
plt.show()


np.savetxt(path+"/data/scan_Oh2_beta_ra_s="+Dic_sDR[sDM]+"_mDM=10^"+str(mDM)+"GeV.txt", np.array([MBH_arr, bra_arr]).T)


