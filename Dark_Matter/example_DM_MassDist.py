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
from odeintw import odeintw
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

import BHProp as bh

from SolFBEqs_Mono import FBEqs_Sol                   # Monochromatic Scenario
from SolFBEqs_MassDist import FBEqs_Sol as FBSol_MD   # Mass Distributions

import time

#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

Mi   = 5.  # Peak mass in g at formation  --> Taken here as a parameter
asi  = 0.  # PBH initial rotation a_star factor
bi   = -3. # Initial PBH fraction 
mDM  = 1.  # Log10 @ Dark Matter mass
sDM  = 0.  # Dark Mater spin

##########################################
#  Distribution types:
#    0 -> Lognormal, requires 1 parameter, sigma
#    1 -> Power law, requires 2 parameters, [sigma, alpha]
#    2 -> Critical collapse, doesn't require any parameters
#    3 -> Metric preheating, doesn't require any parameters
#########################################

typ = 0
sig = 1.0 
alpha  = 2.75

if typ == 1:
    pars_MD = [sig, alpha]
else:
    pars_MD = sig

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

print(f"\n Monochromatic Time {end - start} s\n") #

#+++++++++++++++++++++++++++++#
#       Mass Distribution     #
#+++++++++++++++++++++++++++++#

start = time.time()

Oh2 = FBSol_MD(Mi, bi, typ, pars_MD, mDM, sDM)

x, t, Rad, PBH, TUn, NDBE, Tev = Oh2.Solt()

end = time.time()

print(f"Mass Distribution Time {end - start} s\n")

#------------------------------------------------------------#
#                                                            #
#                     Determining Oh^2                     #
#                                                            #
#------------------------------------------------------------#

nphi = (2.*zeta(3)/np.pi**2)*TUnm[0]**3             # Initial photon number density

rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3

T0 = 2.34865e-13  # Temperature today in GeV

Oh2   = NDBE[-1] * nphi * 10.**(-3.*x[-1]) * 10.**mDM * (bh.gstarS(T0)/bh.gstarS(TUn[-1]))*(T0/TUn[-1])**3*(1/rc)
Oh2m  = NDBEm[-1] * nphi * 10.**(-3.*xm[-1]) * 10.**mDM * (bh.gstarS(T0)/bh.gstarS(TUnm[-1]))*(T0/TUnm[-1])**3*(1/rc)

print(Oh2, Oh2m)
print("Oh^2/Oh^2_mono = {}, Diff = {} %".format(Oh2/Oh2m, 100.*(Oh2-Oh2m)/Oh2m))

#===========================#
#           Plots           #
#===========================#

fig, ax = plt.subplots(2, 2, figsize=(15.,10.))

w = -1. + (1/3.)*(3*10**(-3*x)*PBH + 4*10**(-4*x)*Rad)/(10**(-3*x)*PBH + 10**(-4*x)*Rad)
wm = -1. + (1/3.)*(3*10**(-3*xm)*PBHm + 4*10**(-4*xm)*Radm)/(10**(-3*xm)*PBHm + 10**(-4*xm)*Radm)

ax[0,0].plot(x, w, c='#009C3B', label='Full')
ax[0,0].plot(xm, wm, c='#009C3B', ls='--', label='Mono')
ax[0,0].set_ylim(0., 0.35)
ax[0,0].set_title(r"$m_\chi=10^{10}$ GeV, $M_{\rm PBH}^{\rm mid}=10^3$ g")
ax[0,0].set_xlabel(r"$\log(a)$")
ax[0,0].set_ylabel(r"$\omega$")
ax[0,0].legend(loc="lower left", fontsize = "small")

ax[0,1].plot(x, 10**-x*Rad, label='R', lw = 2, color = '#5D9CF3')
ax[0,1].plot(x, PBH, label='PBH', lw = 2, color = '#1e1f26')
ax[0,1].plot(xm, 10**-xm*Radm, label='R-Mono', lw = 2, ls='--', color = '#5D9CF3')
ax[0,1].plot(xm, PBHm, label='PBH-Mono', lw = 2, color = '#1e1f26', ls='--')
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel(r"$log(a)$")
ax[0,1].set_ylabel(r"$\rho_{i} a^4$")
ax[0,1].legend(loc="lower left", fontsize = "small")

ax[1,0].plot(x, TUn, label=r"$\sigma=0.5$", color = '#66023C')
ax[1,0].plot(xm, TUnm, label=r"Mono", color = '#66023C', linestyle='--')
ax[1,0].set_ylabel(r"$T_{p}$ [GeV]")
ax[1,0].set_xlabel(r"$log(a)$")
ax[1,0].set_yscale('log')
ax[1,0].legend(loc="lower left", fontsize = "small")

ax[1,1].plot(x, NDBE, label=r"$\sigma=0.5$", color = '#66023C')
ax[1,1].plot(xm, NDBEm, label=r"Mono", color = '#66023C', linestyle='--')
ax[1,1].set_ylabel(r"$n_\chi/n_\gamma^{\rm in}$")
ax[1,1].set_xlabel(r"$log(a)$")

ax[1,1].set_yscale('log')
ax[1,1].legend(loc="lower left", fontsize = "small")

plt.show()


