###################################################################################################
#                                                                                                 #
#                               Primordial Black Hole Evaporation                                 #
#                         Dark Matter Production from Hawking Radiation                           #
#                             Considering Mass and Spin Distributions                             #
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

import BHProp as bh

from SolFBEqs_Mono import FBEqs_Sol                         # Monochromatic Scenario
from SolFBEqs_SpinMassDist import FBEqs_Sol as FBSol_SMD    # Mass and Spin Distributions

import time

#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

Mc  = 5.  # Peak mass in g at formation  --> Taken here as a parameter
asc = 0.  # PBH initial rotation a_star factor
bi  = -7. # Initial PBH fraction

mDM = 1.  # Log10@Dark Matter mass, in GeV
sDM = 0.  # Dark Matter spin

##########################################
#  Distribution types:
#  Mass:
#    0 -> Lognormal, requires 1 parameter, sigma
#    1 -> Power law, requires 2 parameters, [sigma, alpha]
#    2 -> Critical collapse, doesn't require any parameters
#    3 -> Metric preheating, doesn't require any parameters
#  Spin:
#    0 -> Gaussian
#    1 -> Merger spin distribution
#########################################

typ_MD  = 0   # Type of mass distribution
sig_M   = 0.5 # Width of the distribution
alpha   = 1.0 # Alpha parameter -> used for Power law scenario

if typ_MD == 1:
    pars_MD = [sig_M, alpha]
else:
    pars_MD = sig_M

typ_SD  = 0   # Type of spin distribution
sig_a   = 0.1 # Width of the spin distribution
pars_SD = sig_a

typ  = [typ_MD, typ_SD]
pars = [pars_MD, pars_SD]

#------------------------------------------------------------------------------------------------------#
#          We call the solver, and save the arrays containing the full evolution of the PBH,           #
#    SM and DR comoving energy densities, together with the evolution of the PBH mass and spin         #
#                              as function of the log_10 @ scale factor.                               #
#               We compute for both monochromatic and mass & spin distribution scenario                #
#------------------------------------------------------------------------------------------------------#

#+++++++++++++++++++++++++++++#
#        Monochromatic        #
#+++++++++++++++++++++++++++++#

start = time.time()

Oh2m = FBEqs_Sol(Mc, asc, bi, mDM, sDM)

xm, tm, MBHm, astm, Radm, PBHm, TUnm, NDBEm, Tev  = Oh2m.Solt()

end = time.time()

nphim = (2.*zeta(3)/np.pi**2)*TUnm[0]**3         # Initial photon number density -> Monochromatic case

print(f"Monochromatic Time {end - start} s\n")

#+++++++++++++++++++++++++++++#
#       Mass Distribution     #
#+++++++++++++++++++++++++++++#

Oh2 = FBSol_SMD(Mc, asc, bi, typ, pars, mDM, sDM)

x, t, Rad, PBH, TUn, NDBE  = Oh2.Solt()

nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3           # Initial photon number density

#------------------------------------------------------------#
#                                                            #
#                     Conversion to Oh^2                     #
#                                                            #
#------------------------------------------------------------#

rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3

T0 = 2.34865e-13  # Temperature today in GeV

Oh2  = NDBE[-1] * nphi * 10.**(-3.*x[-1]) * 10.**mDM * (bh.gstarS(T0)/bh.gstarS(TUn[-1]))*(T0/TUn[-1])**3*(1/rc)

Oh2m = NDBEm[-1] * nphim * 10.**(-3.*xm[-1]) * 10.**mDM * (bh.gstarS(T0)/bh.gstarS(TUnm[-1]))*(T0/TUnm[-1])**3*(1/rc)

print(Oh2,Oh2m)
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

#plt.savefig("./Comp_Mono_Distr.pdf")
plt.show()


