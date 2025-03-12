##################################################################################
#                                                                                #
#                  Primordial Black Hole + Freeze-out Dark Matter.               #
#                                                                                #
##################################################################################
import sys
import numpy as np
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

from dm.planck_relics import FBEqs_Sol

import time

#from Omega_h2_DM_X_v2 import FrInPBH as FrIn2

Mi   = 5.  # Horizon mass in g at formation  --> Taken here as a parameter
asi  = 0.  # PBH initial rotation a_star factor
bi   = -5. # Initial PBH fraction 
mDM  = 0.

Oh2m = FBEqs_Sol(Mi, asi, bi)

xm, tm, MBHm, astm, Radm, PBHm, TUnm, NPRm, Tev  = Oh2m.Solt()

end = time.time()

fig, ax = plt.subplots(2, 2, figsize=(15.,10.))

ax[0,0].plot(xm, Radm/(Radm + 10**xm*(PBHm)), c='#44327E', label='R-Mono')
ax[0,0].plot(xm, 10**xm*(PBHm)/(Radm + 10**xm*(PBHm)), c='#009C3B', label='PBH-Mono')
ax[0,0].set_yscale('log')
#ax[0,0].set_xlim(0,0.85*a[-1])
ax[0,0].set_ylim(1.e-5, 1.e1)
ax[0,0].set_title(r"$m_\chi=10^{10}$ GeV, $M_{\rm PBH}^{\rm mid}=10^3$ g")
ax[0,0].set_xlabel(r"$\log(a)$")
ax[0,0].set_ylabel(r"$\frac{\rho_{i}}{\rho_{tot}}$")
ax[0,0].legend(loc="lower left", fontsize = "small")

ax[0,1].plot(xm, 10**-xm*Radm, label='R-Mono', lw = 2, color = '#5D9CF3')
ax[0,1].plot(xm, PBHm, label='PBH-Mono', lw = 2, color = '#1e1f26')
#ax[0,1].set_xlim(0,0.85*a[-1])
ax[0,1].set_yscale('log')
#ax[0,1].set_ylim(1.e40, 1.e62) 
ax[0,1].set_xlabel(r"$log(a)$")
ax[0,1].set_ylabel(r"$\rho_{i} a^4$")
ax[0,1].legend(loc="lower left", fontsize = "small")

ax[1,0].plot(xm, TUnm, label=r"Mono", color = '#66023C')
#ax[1,0].set_xlim(0,0.85*a[-1])
#ax[1,0].set_ylim(1.e0,5.*TUn[0])
#ax[1,0].axvline(x=aflog10, alpha=0.5, color = '#4E2A84', linestyle='--')
ax[1,0].set_ylabel(r"$T_{p}$ [GeV]")
ax[1,0].set_xlabel(r"$log(a)$")
ax[1,0].set_yscale('log')
ax[1,0].legend(loc="lower left", fontsize = "small")

ax[1,1].plot(xm, NPRm, label=r"Mono", color = '#66023C')
#ax[1,1].set_xlim(0,0.85*a[-1])
#ax[1,1].set_ylim(1.e-9, 1.e1) 
ax[1,1].set_ylabel(r"$n_\chi/n_\gamma^{\rm in}$")
ax[1,1].set_xlabel(r"$log(a)$")
#ax[1,1].set_xscale('log')
ax[1,1].set_yscale('log')
ax[1,1].legend(loc="lower left", fontsize = "small")

#plt.savefig("./Comp_Mono_Distr.pdf")
plt.show()

#------------------------------------------------------------#
#                                                            #
#                     conversion to Oh^2                     #
#                                                            #
#------------------------------------------------------------#

nphi = (2.*zeta(3)/np.pi**2)*TUnm[0]**3             # Initial photon number density

rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3

T0 = 2.34865e-13  # Temperature today in GeV

Oh2m = NPRm[-1] * 10.**(-3.*xm[-1]) * bh.mPL * (bh.gstarS(T0)/bh.gstarS(Tev))*(T0/Tev)**3*(1/rc)

print("Oh^2 = {0:.6E}".format(Oh2m))
