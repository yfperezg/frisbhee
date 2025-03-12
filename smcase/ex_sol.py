###################################################################################################
#                                                                                                 #
#                               Primordial Black Hole Evaporation                                 #
#                                                                                                 #
#                                Author: Yuber F. Perez-Gonzalez                                  #
#                                   Based on: arXiv:2207.09462                                    #
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

from src import bhprop as bh # Schwarzschild and Kerr BHs library

from smcase.soleqs import FBEqs_Sol # Main Solver

#----------------------------------------#
#           Main Parameters              #
#----------------------------------------#

Mi  =  2.0 # Log10@ Initial BH mass in g
asi =  0.9999 # Initial a* value, a* = 0. -> Schwarzschild, a* > 0. -> Kerr.
bi  = -5.0 # Log10@beta^\prime

SBHi  = 2.*pi*bh.GN*(10.**Mi/bh.GeV_in_g)**2*(1. + sqrt(1. - asi**2)) # Initial Bekenstein-Hawking entropy  -- Dimensionless

#------------------------------------------------------------------------------------------------------#
#          We call the solver, and save the arrays containing the full evolution of the PBH,           #
#    SM and DR comoving energy densities, together with the evolution of the PBH mass and spin         #
#                              as function of the log_10 @ scale factor.                               #
#------------------------------------------------------------------------------------------------------#

Oh2m = FBEqs_Sol(Mi, asi, bi, False)

a, t, MBH, ast, SBH, SRD, Rad, PBH, TUn, Teql, Tdec, Tev, x_Page, t_Page  = Oh2m.Solt()

print("{0:.6E}".format(TUn[0]), Teql, Tdec, Tev)

# Saving the arrays...
solTab = array([a, t, ast, MBH, SBH, SRD, Rad, PBH, TUn])
#savetxt("./Data/SolFBEqs_mono_"+Dic_sDR[sDR]+"_a*="+str(asi)+".txt",solTab.T)

#===========================#
#           Plots           #
#===========================#

# Non-comoving Energy densities

rrad = Rad * 10**(-4*a)
rpbh = PBH * 10**(-3*a)
rtot = rrad + rpbh

# Plot

title_1 = '$M_{\\rm PBH}^{\\rm in}=10$^'+'{} g, '.format(Mi) + '$a_\\star=${}'.format(asi)
title_2 = '$\\beta^\\prime=10$^'+'{}'.format(bi)

fig, ax = plt.subplots(2, 2, figsize=(12.,7.5))

ax[0,0].plot(t/t[-1], MBH/10.**Mi, label='PBH Mass', color=(0.39, 0.0, 0.8))
#ax[0,0].plot(t/t[-1], ast/asi, label='PBH $a_\star$', dashes=[6, 2], color=(0.0, 0.7, 0.44))
ax[0,0].set_title(title_1)
ax[0,0].set_xlabel(r"$\xi = t/\tau$")
ax[0,0].set_ylabel(r"$f(t)$")
ax[0,0].legend(loc="lower left", fontsize = "small")
ax[0,0].axvline(x=t_Page/t[-1], ls='-.', color='g')

ax[0,1].plot(a, 10**(3.*a)*rrad, label='SM Radiation', lw = 1.5, color=(0.2, 0.6, 1.))
ax[0,1].plot(a, 10**(3.*a)*rpbh, label='PBH', lw = 1.5, color='k')
#ax[0,1].plot(t/t[-1], 10**(3.*a)*rrad, label='SM Radiation', lw = 1.5, color=(0.2, 0.6, 1.))
#ax[0,1].plot(t/t[-1], 10**(3.*a)*rpbh, label='PBH', lw = 1.5, color='k')
ax[0,1].set_title(title_2)
#ax[0,1].set_ylim(1.e45, 1.e54) 
ax[0,1].set_yscale('log')
ax[0,1].set_xlabel(r"$log(a)$")
ax[0,1].set_ylabel(r"$\rho_{i} a^3$")
ax[0,1].legend(loc="lower left", fontsize = "small")
ax[0,1].axvline(x=x_Page, ls='-.', color='g')
#ax[0,1].axvline(x=t_Page/t[-1], ls='-.', color='g')

ax[1,0].plot(a, TUn, color = '#66023C')
ax[1,0].set_ylabel(r"$T_{\rm plasma}$ [GeV]")
ax[1,0].set_xlabel(r"$log(a)$")
ax[1,0].set_yscale('log')
ax[1,0].axvline(x=x_Page, ls='-.', color='g')

ax[1,1].plot(t/t[-1], SBH/SBHi)
ax[1,1].plot(t/t[-1], SRD/SBHi)
ax[1,1].set_ylim(0, 1.0) 
ax[1,1].set_ylabel(r"$S/S_{in}$")
ax[1,1].set_xlabel(r"$\xi = t/\tau$")
#ax[1,1].set_yscale('log')
ax[1,1].axvline(x=t_Page/t[-1], ls='-.', color='g')

plt.savefig(path+"/plots/Example_solution.pdf")
plt.show()