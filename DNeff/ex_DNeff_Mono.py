###################################################################################################
#                                                                                                 #
#                               Primordial Black Hole Evaporation                                 #
#                            Determination of DNeff for Dark Radiation                            #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                                   Based on: arXiv:2207.09462                                    #
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
import time
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, real, imag
from numpy import absolute, angle, array, savetxt

import BHProp as bh # Schwarzschild and Kerr BHs library

from DNeff_Mono import FBEqs_Sol # Main Solver

#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

Mi  = 5.    # Log10@ Initial BH mass in g
asi = 0.99  # Initial a* value, a* = 0. -> Schwarzschild, a* > 0. -> Kerr.
bi  = -3.   # Log10@beta^\prime
sDR = 2.    # Spin of Dark Radiation

Dic_sDR = {0.:'scl', 0.5:'fer', 1.:'vec', 2.:'gra'}

#------------------------------------------------------------------------------------------------------#
#          We call the solver, and save the arrays containing the full evolution of the PBH,           #
#    SM and DR comoving energy densities, together with the evolution of the PBH mass and spin         #
#                              as function of the log_10 @ scale factor.                               #
#------------------------------------------------------------------------------------------------------#

Oh2m = FBEqs_Sol(Mi, asi, bi, sDR)

a, t, MBH, ast, Rad, PBH, TUn, DRad  = Oh2m.Solt()

# Saving the arrays...

solTab = array([a, t, ast, MBH, Rad, PBH, TUn, DRad])
savetxt("./Data/DNeff/SolFBEqs_mono_"+Dic_sDR[sDR]+"_a*="+str(asi)+".txt",solTab.T)

# Determining DNeff...

rDR_rRad = DRad[-1]/Rad[-1] # DR to SM energy densities ratio
EV_EQ = (bh.gstar(TUn[-1])/bh.gstar(0.75e-9)) * (bh.gstarS(0.75e-9)/bh.gstarS(TUn[-1]))**(4./3.) # Conversion factor to matter-radiation
DNeff = ((8./7.)*(4./11.)**(-4./3.) + 3.045) * rDR_rRad * EV_EQ # Determining DNEff

print('M_PBH = 10^{} g, a_*={}: DNeff = {}'.format(Mi, asi, DNeff))

#===========================#
#           Plots           #
#===========================#

# Non-comoving Energy densities

rrad = Rad * 10**(-4*a)
rdrd = DRad* 10**(-4*a)
rpbh = PBH * 10**(-3*a)
rtot = rrad + rpbh

# Plot

title_1 = '$M_{\\rm PBH}^{\\rm in}=10$^'+'{} g, '.format(Mi) + '$a_\star=${}'.format(asi)
title_2 = '$\\beta^\prime=10$^'+'{}'.format(bi)

fig, ax = plt.subplots(2, 2, figsize=(12.,7.5))

ax[0,0].plot(t/t[-1], MBH/10.**Mi, label='PBH Mass', color=(0.39, 0.0, 0.8))
ax[0,0].plot(t/t[-1], ast/asi, label='PBH $a_\star$', dashes=[6, 2], color=(0.0, 0.7, 0.44))
ax[0,0].set_title(title_1)
ax[0,0].set_xlabel(r"$\xi = t/\tau$")
ax[0,0].set_ylabel(r"$f(t)$")
ax[0,0].legend(loc="lower left", fontsize = "small")

ax[0,1].plot(a, 10**(3.*a)*rrad, label='SM Radiation', lw = 1.5, color=(0.2, 0.6, 1.))
ax[0,1].plot(a, 10**(3.*a)*rpbh, label='PBH', lw = 1.5, color='k')
ax[0,1].plot(a, 10**(3.*a)*rdrd, label='Dark Radiation', color=(0.96, 0.22, 0.))
ax[0,1].set_title(title_2)
ax[0,1].set_ylim(1.e45, 1.e54) 
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel(r"$log(a)$")
ax[0,1].set_ylabel(r"$\rho_{i} a^3$")
ax[0,1].legend(loc="lower left", fontsize = "small")

ax[1,0].plot(a, TUn, color = '#66023C')
ax[1,0].set_ylabel(r"$T_{\rm plasma}$ [GeV]")
ax[1,0].set_xlabel(r"$log(a)$")
ax[1,0].set_yscale('log')

ax[1,1].plot(a, DRad/Rad)
ax[1,1].set_ylim(1.e-4, 1.e1) 
ax[1,1].set_ylabel(r"$\rho_{\rm GRAV}/\rho_{\rm SM}$")
ax[1,1].set_xlabel(r"$log(a)$")
ax[1,1].set_yscale('log')

plt.savefig("./Example_solution_DNeff.pdf")
plt.show()
