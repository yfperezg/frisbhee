###################################################################################################
#                                                                                                 #
#                               Primordial Black Hole Evaporation                                 #
#                            Determination of DNeff for Dark Radiation                            #
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

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, real, imag
from numpy import absolute, angle, array, savetxt

import BHProp as bh

from DNeff_Mono import FBEqs_Sol                    # Monochromatic Scenario
from DNeff_MassDist import FBEqs_Sol as FBSol_MD    # Mass Distributions

import time

#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

Mi   = 7.  # Peak mass in g at formation  --> Taken here as a parameter
asi  = 0.  # PBH initial rotation a_star factor
bi   = -3. # Initial PBH fraction
sDR  = 0.0 # Spin of Dark Radiation

Dic_sDR = {0.:'scl', 0.5:'fer', 1.:'vec', 2.:'gra'}

################################################################
#  Distribution types:
#    0 -> Lognormal, requires 1 parameter, sigma
#    1 -> Power law, requires 2 parameters, [sigma, alpha]
#    2 -> Critical collapse, doesn't require any parameters
#    3 -> Metric preheating, doesn't require any parameters
################################################################

typ = 0
sig = 1.0 
alpha = 2.5

if typ == 1:
    pars_MD = [sig, alpha]
else:
    pars_MD = sig

Dis_types = {0:"LN", 1:"PL", 2:"CC", 3:"MP"}

#------------------------------------------------------------------------------------------------------#
#          We call the solver, and save the arrays containing the full evolution of the PBH,           #
#    SM and DR comoving energy densities, together with the evolution of the PBH mass and spin         #
#                              as function of the log_10 @ scale factor.                               #
#                  We compute for both monochromatic and mass distribution scenario                    #
#------------------------------------------------------------------------------------------------------#

#+++++++++++++++++++++++++++++#
#        Monochromatic        #
#+++++++++++++++++++++++++++++#

SolDR_m = FBEqs_Sol(Mi, asi, bi, sDR)

xm, tm, MBHm, astm, Radm, PBHm, TUnm, DRadm  = SolDR_m.Solt()

#+++++++++++++++++++++++++++++#
#       Mass Distribution     #
#+++++++++++++++++++++++++++++#

SolDR = FBSol_MD(Mi, bi, typ, pars_MD, sDR)

x, t, Rad, PBH, TUn, DRad  = SolDR.Solt()

fig, ax = plt.subplots(2, 2, figsize=(15.,10.))

w = -1. + (1/3.)*(3*10**(-3*x)*PBH + 4*10**(-4*x)*Rad)/(10**(-3*x)*PBH + 10**(-4*x)*Rad)
wm = -1. + (1/3.)*(3*10**(-3*xm)*PBHm + 4*10**(-4*xm)*Radm)/(10**(-3*xm)*PBHm + 10**(-4*xm)*Radm)

# Determining DNeff...

#Monochromatic
rDR_rRad_m = (DRadm[-1]/(Radm[-1]))

EV_EQ_m = (bh.gstar(TUnm[-1])/bh.gstar(0.75e-9)) * (bh.gstarS(0.75e-9)/bh.gstarS(TUnm[-1]))**(4./3.)

DNEff_m = ((8./7.)*(4./11.)**(-4./3.) + 3.045) * rDR_rRad_m * EV_EQ_m

#Mass distribution
rDR_rRad = (DRad[-1]/(Rad[-1]))

EV_EQ = (bh.gstar(TUn[-1])/bh.gstar(0.75e-9)) * (bh.gstarS(0.75e-9)/bh.gstarS(TUn[-1]))**(4./3.)

DNEff = ((8./7.)*(4./11.)**(-4./3.) + 3.045) * rDR_rRad * EV_EQ


print(10**Mi, DNEff, DNEff_m)
    
print("DNeff/DNeff_mono = {}, Diff = {} %".format(DNEff/DNEff_m, 100.*(DNEff-DNEff_m)/DNEff_m))

#===========================#
#           Plots           #
#===========================#

ax[0,0].plot(x, w, c='#009C3B', label='Full')
ax[0,0].plot(xm, wm, c='#009C3B', ls='--', label='Mono')
ax[0,0].set_ylim(0., 0.35)
ax[0,0].set_title(r"spin DR = {0}, M_c = 10^{1}".format(sDR, Mi))
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

ax[1,0].plot(x, TUn, label=r"Extended", color = '#66023C')
ax[1,0].plot(xm, TUnm, label=r"Mono", color = '#66023C', linestyle='--')
ax[1,0].set_ylabel(r"$T_{p}$ [GeV]")
ax[1,0].set_xlabel(r"$log(a)$")
ax[1,0].set_yscale('log')
ax[1,0].legend(loc="lower left", fontsize = "small")

ax[1,1].plot(x, DRad/Rad, label=r"Extended", color = '#66023C')
ax[1,1].plot(xm, DRadm/Radm, label=r"Mono", color = '#66023C', linestyle='--')
ax[1,1].set_ylim(1.e-4, 1.e1) 
ax[1,1].set_ylabel(r"$n_\chi/n_\gamma^{\rm in}$")
ax[1,1].set_xlabel(r"$log(a)$")
ax[1,1].set_yscale('log')
ax[1,1].legend(loc="lower left", fontsize = "small")

plt.savefig("./Example_solution_DNeff_Mass_Dist.pdf")
plt.show()
