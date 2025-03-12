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

import sys
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

from src import bhprop as bh

from dm.mono import FBEqs_Sol                   # Monochromatic Scenario

import time

#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

Mi   = 5.  # Peak mass in g at formation  --> Taken here as a parameter
asi  = 0.99999  # PBH initial rotation a_star factor
bi   = -3. # Initial PBH fraction 
mDM  = 1.  # Log10 @ Dark Matter mass
sDM  = 2.  # Dark Mater spin

#------------------------------------------------------------------------------------------------------#
#          We call the solver, and save the arrays containing the full evolution of the PBH,           #
#    SM and DR comoving energy densities, together with the evolution of the PBH mass and spin         #
#                              as function of the log_10 @ scale factor.                               #
#                  We compute for both monochromatic and mass distribution scenario                    #
#------------------------------------------------------------------------------------------------------#

#+++++++++++++++++++++++++++++#
#        Monochromatic        #
#+++++++++++++++++++++++++++++#

start = time.time()

Oh2m = FBEqs_Sol(Mi, asi, bi, mDM, sDM)

xm, tm, MBHm, astm, Radm, PBHm, TUnm, NDBEm, Tevm  = Oh2m.Solt()

end = time.time()

#------------------------------------------------------------#
#                                                            #
#                     Determining Oh^2                     #
#                                                            #
#------------------------------------------------------------#

nphi = (2.*zeta(3)/np.pi**2)*TUnm[0]**3             # Initial photon number density

rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3

T0 = 2.34865e-13  # Temperature today in GeV

Oh2m  = NDBEm[-1] * nphi * 10.**(-3.*xm[-1]) * 10.**mDM * (bh.gstarS(T0)/bh.gstarS(TUnm[-1]))*(T0/TUnm[-1])**3*(1/rc)

print("Oh^2 = {0:.6E}".format(Oh2m))

#===========================#
#           Plots           #
#===========================#

fig, ax = plt.subplots(2, 2, figsize=(15.,10.))

wm = -1. + (1/3.)*(3*10**(-3*xm)*PBHm + 4*10**(-4*xm)*Radm)/(10**(-3*xm)*PBHm + 10**(-4*xm)*Radm)

ax[0,0].plot(xm, wm, c='#009C3B', label='Mono')
ax[0,0].set_ylim(0., 0.35)
ax[0,0].set_title(r"$m_\chi=10^{10}$ GeV, $M_{\rm PBH}^{\rm mid}=10^3$ g")
ax[0,0].set_xlabel(r"$\log(a)$")
ax[0,0].set_ylabel(r"$\omega$")
ax[0,0].legend(loc="lower left", fontsize = "small")

ax[0,1].plot(xm, 10**-xm*Radm, label='R-Mono', lw = 2, color = '#5D9CF3')
ax[0,1].plot(xm, PBHm, label='PBH-Mono', lw = 2, color = '#1e1f26')
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel(r"$log(a)$")
ax[0,1].set_ylabel(r"$\rho_{i} a^4$")
ax[0,1].legend(loc="lower left", fontsize = "small")

ax[1,0].plot(xm, TUnm, label=r"Mono", color = '#66023C',)
ax[1,0].set_ylabel(r"$T_{p}$ [GeV]")
ax[1,0].set_xlabel(r"$log(a)$")
ax[1,0].set_yscale('log')
ax[1,0].legend(loc="lower left", fontsize = "small")

ax[1,1].plot(xm, NDBEm, label=r"Mono", color = '#66023C')
ax[1,1].set_ylabel(r"$n_\chi/n_\gamma^{\rm in}$")
ax[1,1].set_xlabel(r"$log(a)$")

ax[1,1].set_yscale('log')
ax[1,1].legend(loc="lower left", fontsize = "small")

plt.show()