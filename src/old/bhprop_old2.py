###################################################################################################
#                                                                                                 #
#                               Schwarzschild and Kerr BHs Library                                #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                                   Based on: arXiv:2107.00013                                    #
#                                 Last update: December 6th, 2023                                 #
#                                                                                                 #
###################################################################################################

import numpy as np
from mpmath import polylog
from scipy import interpolate
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn, spherical_jn, jv
from scipy.interpolate import RectBivariateSpline


from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, real

from collections import OrderedDict
olderr = np.seterr(all='ignore')

# Particle masses, in GeV

mW   = 80.379
mZ   = 91.1876
mH   = 125.18
me   = 0.5109989461e-3
mmu  = 105.6583745e-3
mtau = 1.77686
mu   = 2.2e-3
md   = 4.6e-3
ms   = 95e-3
mc   = 1.275
mb   = 4.18
mt   = 173.1
mg   = 0.6      # Ficticious gluon mass ---> indicates the QCD phase transition, following PRD41(1990)3052

# Degrees of freedom of the SM ---> Before the EW phase transition

gW  = 4.        # W
gZ  = 2.        # Z
gH  = 4.        # Higgs
gp  = 2.        # photon
gg  = 2.        # graviton
ggl = 16.       # gluons
gl  = 2.*2.     # leptons
gq  = 2.*2.*3.  # quarks
gnu = 2.        # LH neutrino

gf = 3.*gnu + 3.*gl + 6.*gq   # Total number of SM fermion dofs 
gs = gH                       # Total number of SM scalar dofs
gv = gW + gZ + gp + gg + ggl  # Total number of SM vector dofs

# Constants

c     = 299792.458       # in km/s
gamma = sqrt(3.)**-3.    # Collapse factor
GCF   = 6.70883e-39      # Gravitational constant in GeV^-2
mPL   = GCF**-0.5        # Planck mass in GeV
v     = 174              # Higgs vev
csp   = 0.35443          # sphaleron conversion factor
GF    = 1.1663787e-5     # Fermi constant in GeV^-2

# Conversion factors

GeV_in_g     = 1.782661907e-24  # 1 GeV in g
Mpc_in_cm    = 3.085677581e24   # 1 Mpc in cm

cm_in_invkeV = 5.067730938543699e7       # 1 cm in keV^-1
year_in_s    = 3.168808781402895e-8      # 1 year in s
GeV_in_invs  = cm_in_invkeV * c * 1.e11  # 1 GeV in s^-1

MPL   = mPL * GeV_in_g        # Planck mass in g
kappa = mPL**4 * GeV_in_g**3  # Evaporation constant in g^3 * GeV -- from PRD41(1990)3052
mPL_red = mPL/sqrt(8.*pi)  # Reduced Planck mass

# BH Temperature in GeV

def TBH(M, astar):

    M_GeV = M/GeV_in_g
    
    return (1./(4.*pi*GCF*M_GeV))*(sqrt(abs(1. - astar**2))/(1. + sqrt(abs(1. - astar**2)))) # M in g

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                  Momentum Integrated Rate for Kerr BHs                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#


def Gamma_S(M, ast, m):# Scalar, in GeV

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)
    
    hs = 10.**(0.39273676881556124 - 0.07212262928269993*ast + 0.12449061251994815*ast**2
               + (0.00009524790725091*ast**2)/(-1.025 + ast)**2 -  0.6630039387105334*ast**3 + 0.20597619699493652*ast**4)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [0.908948, -0.717238, 4.53781, -10.7304, 7.11179, -0.000286806]
        b0, b1, b2, b3, b4, b5 = [7.71534, -1.25411, 3.64632, -18.5727, 10.439, -0.000485377]
        c0, c1, c2, c3, c4, c5 = [-0.402682, 0.156468, -2.03774, 7.73825, -6.39103, 0.000585775]
        
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        z = GM * m

        In = hs * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
        
    else:
        
        In = hs
    
    return  (27/(1024. * pi**4 * GM)) * In


def Gamma_F(M, ast, m):# Fermion

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    hf = 10.**(-0.04783695964578665 + 0.013984310871408692*ast + 0.5688253273581945*ast**2
               + (0.00003280653440514944*ast**2)/(-1.025 + ast)**2 - 0.036327327226993424*ast**3 - 0.18526329851489926*ast**4)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [1.02698, 0.0915114, -0.723386, 1.48036, -1.38637, 0.000193827]
        b0, b1, b2, b3, b4, b5 = [8.66596, -0.845019, 1.08049, -8.92803, 2.77038, -0.00131193]
        c0, c1, c2, c3, c4, c5 = [-0.46751, 0.137131, -0.504895, 0.781955, 0.223372, -0.000357428]

        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)
        
        z = GM * m

        In = hf * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
                        
    else:
        
        In = hf
    
    return  2. * (27/(1024. * pi**4 * GM)) * In


def Gamma_V(M, ast, m):# Vector

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    hv = 10.**(-0.5599312867164357 - 0.12712336386255416*ast + 3.446923230985025*ast**2
               + (0.00009543107588469785*ast**2)/(-1.025 + ast)**2 - 3.8597576626367047*ast**3 + 1.903809455654925*ast**4)
    
    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [1.13063, 0.10242, -0.665276, 1.5559, -1.30436, -0.0000798625]
        b0, b1, b2, b3, b4, b5 = [9.1147, -0.450361, -3.4622, 4.33463, -6.48433, -0.000418639]
        c0, c1, c2, c3, c4, c5 = [-0.522355, -0.17723, 1.15501, -2.50918, 1.95021, 0.0000871877]

        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        z = GM * m

        In = hv * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
                                
    else:
        
        In = hv
    
    return  3. * (27/(1024. * pi**4 * GM)) * In

def Gamma_G(M, ast, m):# Spin 2

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    hg = 10.**(-1.6914380919125194 + 0.10713534704840354*ast + 8.602630801168678*ast**2
               + (0.00030945691177488064*ast**2)/(-1.025 + ast)**2 - 11.26044905052906*ast**3 + 5.96247972620344*ast**4)

    if m > 0.:
        
        if ast <= 1.e-5:

            B, C, nu = [22.325, -21.2326, 0.12076]

            z = m/TKBH

        else:

            a0, a1, a2, a3, a4, a5 = [1.28037, 0.0711855, -0.239972, 0.762718, -0.673144, -0.0000505832]
            b0, b1, b2, b3, b4, b5 = [9.1527, -0.441805, -7.91835, 13.9276, -12.4764, -0.000891415]
            c0, c1, c2, c3, c4, c5 = [-0.643453, -0.0804094, 0.326238, -0.918118, 0.777633, 0.00005796]

            B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
            C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
            nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

            z = GM * m

        In = hg * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
                                        
    else:
        
        In = hg
    
    return  5. * (27/(1024. * pi**4 * GM)) * In

def Gamma_GO(M, ast, m):# Geometric optics limit

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    zBH = m/TKBH

    In = - zBH * polylog(2, -exp(-zBH)) - polylog(3, -exp(-zBH))# DM emission rate including greybody factors
    
    return  2. * (27/(1024. * pi**4 * GM)) * In

def Gamma_DM(M, ast, mdm, s):

    if s == 0.:
        f = Gamma_S(M, ast, mdm)
    elif s == 0.5:
        f = Gamma_F(M, ast, mdm) # Assuming Majorana DM
    elif s == 1.:
        f = Gamma_V(M, ast, mdm)
    elif s == 2.:
        f = Gamma_G(M, ast, mdm)
        
    return f

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                         Total f functions ---> related to the mass rate, dM/dt                                      #
#                                                   Counting SM dofs + Dark Radiation                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------#
#                     f = - M^2 dM/dt fitted functions                      #
#-------------------------------------------------------------------#

def fs(astar): return (-4.118059636065065 - 0.304070437116496*astar + 1.7414539129754338*astar**2
                       + (0.00016247868687708833*astar**2)/(-1.025 + astar)**2 - 4.038134093574355*astar**3 + 2.96644749765774*astar**4)

def ff(astar): return (-4.380432126443562 - 0.21848948569771906*astar + 2.216260125642757*astar**2
                       + (0.00016500097094511814*astar**2)/(-1.025 + astar)**2 - 3.0278311164742657*astar**3 + 1.8172716129518027*astar**4)
    
def fv(astar): return (-4.768125053334267 - 0.20380890746978889*astar + 4.369105751017979*astar**2
                       + (0.000180938577270079*astar**2)/(-1.025 + astar)**2 - 5.359060591581632*astar**3 + 2.8820245266622875*astar**4)

def fg(astar): return (-5.719681302881681 + 0.1067525411095276*astar + 9.361839615626291*astar**2
                       + (0.0003719947970764218*astar**2)/(-1.025 + astar)**2 - 12.46911299460744*astar**3 + 6.757216601387031*astar**4)

#--------------------------------------------------------------------------------#
#              Our interpolated forms including the particle's mass              #
#--------------------------------------------------------------------------------#

# Scalar

def phi_s(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f0 = 10.**fs(ast)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [0.858267, 1.13329, -6.88816, 11.2483, -5.56238, 0.000101146]
        b0, b1, b2, b3, b4, b5 = [7.06988, 2.40603, -21.8821, 25.0015, -11.0752, -0.00149611]
        c0, c1, c2, c3, c4, c5 = [-0.256082, -2.10605, 13.1112, -23.5922, 12.7525, -0.000240998]

        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)
        
        z = GM * m

        In = f0 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))   
        
    else:
        
        In = f0

    return In

# Fermion

def phi_f(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f12  = 10.**ff(ast)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [1.06034, -0.54818, 3.11063, -6.9571, 4.16081, 0.0000501825]
        b0, b1, b2, b3, b4, b5 = [8.28637, 1.81691, -17.0186, 17.3372, -9.32249, -0.00102115]
        c0, c1, c2, c3, c4, c5 = [-0.46356, 1.07903, -6.42023, 14.0977, -8.79169, 0.0000631419]

        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        z = GM * m

        In = f12 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = f12

    return In

# Vector

def phi_v(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f1 = 10.**fv(ast)

    if m > 0.:
        
        if ast <= 1.e-5:

            B, C, nu = [14.0361, -10.7138, 0.307206]

            z = m/TKBH

        else:

            a0, a1, a2, a3, a4, a5 = [1.14914, 0.0233038, -0.267976, 0.496932, -0.732845, 0.0000736326]
            b0, b1, b2, b3, b4, b5 = [9.02047, -1.60749, 2.4884, -14.6396, 7.17955, -0.0010815]
            c0, c1, c2, c3, c4, c5 = [-0.517646, -0.0423408, 0.45894, -0.895684, 1.11853, -0.000231453]

            B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
            C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
            nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

            z = GM * m

        In = f1 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = f1
    
    return In

# Tensor - spin2

def phi_g(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f2 = 10.**fg(ast)

    if m > 0.:
        
        if ast <= 1.e-5:

            B, C, nu = [21.50941, -20.5135, 0.173423]

            z = m/TKBH

        else:

            a0, a1, a2, a3, a4, a5 = [1.30252, 0.0504643, -0.301552, 0.914737, -0.876773, -0.0000311124]
            b0, b1, b2, b3, b4, b5 = [9.26442, -1.53376, -4.16324, 5.04574, -7.1057, -0.000616182]
            c0, c1, c2, c3, c4, c5 = [-0.656544, -0.0596208, 0.416083, -1.15561, 1.05567, 0.0000348137]
            
            B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
            C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
            nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

            z = GM * m

        In = f2 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = f2
    
    return In

#------------------------------------------#
#              SM Contribution             #
#------------------------------------------#

def fSM(M, ast):
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * phi_v(M, ast, 0.)  # Photon
    fgl = ggl * phi_v(M, ast, 0.6) # Gluon
    fW  =  gW * phi_v(M, ast, mW)  # W
    fZ  =  gZ * phi_v(M, ast, mZ)  # Z
    fH  =  gH * phi_s(M, ast, mH)  # Higgs

    fnu = 3. * gnu * phi_f(M, ast, 0.) # Active neutrinos
    
    fl  = gl * (phi_f(M, ast, me) + phi_f(M, ast, mmu) + phi_f(M, ast, mtau))  # Charged leptons
    
    fq  = gq * (phi_f(M, ast, mu) + phi_f(M, ast, md) + phi_f(M, ast, ms) +
                phi_f(M, ast, mc) + phi_f(M, ast, mb) + phi_f(M, ast, mt))    # Quarks

    
    return fgr + fp + fgl + fW + fZ + fH + fnu + fl + fq

# RH neutrino contribution

def fRH(M, ast, mrh): return gnu * phi_f(M, ast, mrh)

# DM contribution

def fDM(M, ast, mdm, s):
    
    if s == 0.:
        f = phi_s(M, ast, mdm)
    elif s == 0.5:
        f = 2.*phi_f(M, ast, mdm) # Assuming Majorana DM
    elif s == 1.:
        f = 3.*phi_v(M, ast, mdm)
    elif s == 2.:
        f = 5.*phi_g(M, ast, mdm)
        
    return f

def fX(M, ast, mX):   return 3. * phi_v(M, ast, mX)

# Dark Radiation

def fDR(M, ast, s): # Depending on the particle's spin

    if s == 0.:
        f = phi_s(M, ast, 0.)
    elif s == 0.5:
        f = 2.*phi_f(M, ast, 0.)
    elif s == 1.:
        f = 2.*phi_v(M, ast, 0.)
    elif s == 2.:
        f = 2.*phi_g(M, ast, 0.)
        
    return f

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                Total g functions ---> related to the angular momentum rate, da_*/dt                                 #
#                                                   Counting SM dofs + Dark Radiation                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

def gs(astar): return (-4.045229894094535 - 0.16884071313609533*astar + 1.7004406922685915*astar**2
                       + (0.00012407642385735108*astar**2)/(-1.025 + astar)**2 - 2.203764068638818*astar**3 + 1.4236859073851342*astar**4)
    
def gf(astar): return (-3.5048907255963506 - 0.1707592461742352*astar + 1.1081436707194507*astar**2
                       + (0.00013090814453809555*astar**2)/(-1.025 + astar)**2 - 2.077329775142311*astar**3 + 1.5064422940204771*astar**4)

def gv(astar): return (-3.6137778725792122 - 0.18534083027260329*astar + 2.053163757714352*astar**2
                       + (0.00014943669027045418*astar**2)/(-1.025 + astar)**2 - 2.5244879958300217*astar**3 + 1.6384065784010777*astar**4)

def gG(astar): return (-4.262556170489987 - 0.321092080102583*astar + 6.571363507950664*astar**2
                       + (0.00033100809271126064*astar**2)/(-1.025 + astar)**2 - 8.05439652085206*astar**3 + 4.544847430292653*astar**4)

#--------------------------------------------------------------------------------#
#              Our interpolated forms including the particle's mass              #
#--------------------------------------------------------------------------------#

# Scalar

def gam_s(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    g0 = 10.**gs(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.15021, -0.0960927, 0.288357, -1.07176, 0.445158, 0.000135345]
        b0, b1, b2, b3, b4, b5 = [8.1449, 0.464533, -10.3936, 5.47071, -2.18433, -0.0013135]
        c0, c1, c2, c3, c4, c5 = [-0.549769, 0.321081, -1.60387, 3.74354, -2.05203, -0.000217226]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = g0 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = g0
    
    return In

# Fermion

def gam_f(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    g12 = 10.**gf(ast)

    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.00612, -0.37506, 1.99906, -5.13281, 3.36953, 6.72191e-6]
        b0, b1, b2, b3, b4, b5 = [7.54615, 1.92532, -18.744, 22.1178, -11.7847, -0.000856291]
        c0, c1, c2, c3, c4, c5 = [-0.428076, 0.622237, -3.636, 9.10877, -6.23936, 0.000121852]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = g12 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))        
                
    else:
        
        In = g12

    return In

# Vector

def gam_v(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    g1 = 10.**gv(ast)

    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.13229, 0.0350832, -0.413466, 0.621237, -0.699533, 0.0000492522]
        b0, b1, b2, b3, b4, b5 = [8.76985, -0.732767, -1.61877, -8.23582, 4.28217, -0.000958016]
        c0, c1, c2, c3, c4, c5 = [-0.519134, -0.0487171, 0.554668, -0.895856, 0.985914, -0.000184225]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = g1 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
                        
    else:
        
        In = g1

    return In

# Tensor - spin2

def gam_g(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    g2 = 10.**gG(ast)

    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.29532, 0.0688836, -0.530873, 1.28982, -1.03602, -0.0000330189]
        b0, b1, b2, b3, b4, b5 = [9.12665, -0.226946, -9.69857, 13.4095, -11.0022, -0.000625013]
        c0, c1, c2, c3, c4, c5 = [-0.657139, -0.0828605, 0.66564, -1.5549, 1.2225, 0.0000372539]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = g2 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
                                
    else:
        In = g2
        #print(In)
        
    return In

#------------------------------------------#
#              SM Contribution             #
#------------------------------------------#

def gSM(M, ast):

    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.                      # Graviton
    fp  =  gp * gam_v(M, ast, 0.)  # Photon
    fgl = ggl * gam_v(M, ast, 0.6) # Gluon
    fW  =  gW * gam_v(M, ast, mW)  # W
    fZ  =  gZ * gam_v(M, ast, mZ)  # Z
    fH  =  gH * gam_s(M, ast, mH)  # Higgs

    fnu = 3. * gnu * gam_f(M, ast, 0.)                           # Active neutrinos
    
    fl  = gl * (gam_f(M, ast, me) + gam_f(M, ast, mmu) + gam_f(M, ast, mtau))  # Charged leptons
    
    fq  = gq * (gam_f(M, ast, mu) + gam_f(M, ast, md) + gam_f(M, ast, ms) +
                gam_f(M, ast, mc) + gam_f(M, ast, mb) + gam_f(M, ast, mt))    # Quarks

    return fgr + fp + fgl + fW + fZ + fH + fnu + fl + fq

def gRH(M, ast, mRH): return gnu * gam_f(M, ast, mRH)

# Dark Matter contribution

def gDM(M, ast, mdm, s):

    if s == 0.:
        f = gam_s(M, ast, mdm)
    elif s == 0.5:
        f = 2.*gam_f(M, ast, mdm) # Assuming Majorana DM
    elif s == 1.:
        f = 3.*gam_v(M, ast, mdm)
    elif s == 2.:
        f = 5.*gam_g(M, ast, mdm)
        
    return f

# Mediator contribution

def gX(M, ast, mX): return 3. * gam_v(M, ast, mX)

# Dark Radiation

def gDR(M, ast, s):

    if s == 0.:
        f = gam_s(M, ast, 0.)
    elif s == 0.5:
        f = 2.*gam_f(M, ast, 0.)
    elif s == 1.:
        f = 2.*gam_v(M, ast, 0.)
    elif s == 2.:
        f = 2.*gam_g(M, ast, 0.)
        
    return f

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                 Total zeta functions ---> related to the entropy rate, dSrad/dt                                     #
#                                                   Counting SM dofs + Dark Radiation                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

def ss(astar): return (-2.4580065972167535 - 0.06075517807138173*astar + 0.18143760561156974*astar**2
                       + (0.00008880759636540837*astar**2)/(-1.025 + astar)**2 - 0.6268401123928045*astar**3 + 0.2837999358639147*astar**4)
    
def sf(astar): return (-2.772837253362536 - 0.01390992880804958*astar + 0.4884878815202036*astar**2
                       + (0.00004571797475591007*astar**2)/(-1.025 + astar)**2 - 0.2132532132038227*astar**3 + 0.09051282274568348*astar**4)

def sv(astar): return (-3.1936361744593986 - 0.13994963995986145*astar + 2.8007955076642284*astar**2
                       + (0.00009910225955325424*astar**2)/(-1.025 + astar)**2 - 3.127450574608424*astar**3 + 1.61686432496589*astar**4)

def sg(astar): return (-4.188659246819801 + 0.05128915862639619*astar + 7.358438540503012*astar**2
                       + (0.00021737455037611103*astar**2)/(-1.025 + astar)**2 - 9.358360385409766*astar**3 + 4.794844169149003*astar**4)

#--------------------------------------------------------------------------------#
#              Our interpolated forms including the particle's mass              #
#--------------------------------------------------------------------------------#

# Scalar

def zet_s(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    z0 = 10.**ss(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [0.862639, -0.299957, 2.03926, -5.87951, 4.30497, -0.000188365]
        b0, b1, b2, b3, b4, b5 = [6.88522, -0.294533, -2.75879, -4.66752, 2.47259, -0.00048632]
        c0, c1, c2, c3, c4, c5 = [-0.378201, -0.389578, 1.53331, 0.31072, -1.85282, 0.000361938]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = z0 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = z0

    return In
    

# Fermion

def zet_f(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    z12 = 10.**sf(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [0.99758, -0.00576664, -0.210207, 0.222121, -0.472107, 0.000171936]
        b0, b1, b2, b3, b4, b5 = [7.93571, 0.147245, -5.77358, 1.87751, -1.77874, -0.00124554]
        c0, c1, c2, c3, c4, c5 = [-0.455288, 0.369066, -1.85025, 3.77071, -1.82242, -0.000266735]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = z12 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = z12

    return In

# Vector

def zet_v(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    z1 = 10.**sv(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.10223, 0.0719912, -0.600077, 1.24585, -1.07814, -0.0000373172]
        b0, b1, b2, b3, b4, b5 = [8.50988, -0.665385, -3.27291, 0.264623, -2.10992, -0.000491667]
        c0, c1, c2, c3, c4, c5 = [-0.50167, -0.143651, 1.07013, -2.16797, 1.72439, -5.71712e-6]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = z1 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = z1
    
    return In

# Tensor - spin2

def zet_g(M, ast, m):

    GM = GCF * (M/GeV_in_g) # in GeV^-1

    z2 = 10.**sg(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [ 1.25678, 0.0942764, -0.548557, 1.26598, -1.02362, -0.0000456537]
        b0, b1, b2, b3, b4, b5 = [ 8.69785, -0.611793, -8.11752, 10.8511, -9.2552, -0.000559518]
        c0, c1, c2, c3, c4, c5 = [-0.622981, -0.115483, 0.709105, -1.57263, 1.22883, 0.0000458349]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = z2 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = z2

    return In

#------------------------------------------#
#              SM Contribution             #
#------------------------------------------#

def zSM(M, ast):
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * zet_v(M, ast, 0.)  # Photon
    fgl = ggl * zet_v(M, ast, 0.6) # Gluon
    fW  =  gW * zet_v(M, ast, mW)  # W
    fZ  =  gZ * zet_v(M, ast, mZ)  # Z
    fH  =  gH * zet_s(M, ast, mH)  # Higgs

    fnu = 3. * gnu * zet_f(M, ast, 0.) # Active neutrinos
    
    fl  = gl * (zet_f(M, ast, me) + zet_f(M, ast, mmu) + zet_f(M, ast, mtau))  # Charged leptons
    
    fq  = gq * (zet_f(M, ast, mu) + zet_f(M, ast, md) + zet_f(M, ast, ms) +
                zet_f(M, ast, mc) + zet_f(M, ast, mb) + zet_f(M, ast, mt))    # Quarks

    
    return fgr + fp + fgl + fW + fZ + fH + fnu + fl + fq

# RH neutrino contribution

def zRH(M, ast, mrh): return gnu * zet_f(M, ast, mrh)

# DM contribution

def zDM(M, ast, mdm, s):
    
    if s == 0.:
        f = zet_s(M, ast, mdm)
    elif s == 0.5:
        f = 2.*zet_f(M, ast, mdm) # Assuming Majorana DM
    elif s == 1.:
        f = 3.*zet_v(M, ast, mdm)
    elif s == 2.:
        f = 5.*zet_g(M, ast, mdm)
        
    return f

def zX(M, ast, mX):   return 3. * zet_v(M, ast, mX)

# Dark Radiation

def zDR(M, ast, s): # Depending on the particle's spin

    if s == 0.:
        f = zet_s(M, ast, 0.)
    elif s == 0.5:
        f = 2.*zet_f(M, ast, 0.)
    elif s == 1.:
        f = 2.*zet_v(M, ast, 0.)
    elif s == 2.:
        f = 2.*zet_g(M, ast, 0.)
        
    return f

#---------------------------------------------------------------------------------------------------------------------------------------#
#                                                                 PBHs lifetime                                                         #
#---------------------------------------------------------------------------------------------------------------------------------------#

def ItauSM(tl, v): # Standard Model + Gravitons
    
    M   = v[0]
    ast = v[1]

    FSM = fSM(M, ast) + gg * phi_g(M, ast, 0.)  # 
    GSM = gSM(M, ast) + gg * gam_g(M, ast, 0.)  #

    dMdtl   = - log(10.) * 10.**tl * kappa * FSM * M**-2
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GSM - 2.*FSM)

    return [dMdtl, dastdtl]

def ItauFO(tl, v, mDM, sDM): # Freeze Out case
    
    M   = v[0]
    ast = v[1]

    FSM = fSM(M, ast)
    FDM = fDM(M, ast, mDM, sDM) # DM evaporation contribution
    FT  = FSM + FDM             # Total Evaporation contribution

    GSM = gSM(M, ast)
    GDM = gDM(M, ast, mDM, sDM) # DM evaporation contribution
    GT  = GSM + GDM             # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * kappa * FT * M**-2
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GT - 2.*FT)

    return [dMdtl, dastdtl]

def ItauDR(tl, v, s): # Dark Radiation Case
    
    M   = v[0]
    ast = v[1]

    FSM = fSM(M, ast)
    FDR = fDR(M, ast, s) # DM evaporation contribution
    FT  = FSM + FDR      # Total Evaporation contribution

    GSM = gSM(M, ast)
    GDR = gDR(M, ast, s) # DM evaporation contribution
    GT  = GSM + GDR      # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * kappa * FT * M**-2
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GT - 2.*FT)

    return [dMdtl, dastdtl]

def ItauFI(tl, v, mDM, sDM, mX): # Freeze In case (Including mediator)
    
    M   = v[0]
    ast = v[1]

    FSM = fSM(M, ast)
    FDM = fDM(M, ast, mDM, sDM) # DM evaporation contribution
    FX  = fX(M, ast, mX)        # Mediator contribution
    FT  = FSM + FDM + FX        # Total Evaporation contribution

    GSM = gSM(M, ast)
    GDM = gDM(M, ast, mDM, sDM) # DM evaporation contribution
    GX  = gX(M, ast, mX)        # Mediator contribution
    GT  = GSM + GDM + GX        # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * kappa * FT * M**-2
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GT - 2.*FT)

    return [dMdtl, dastdtl]

def ItauRH(tl, v, M1, M2, M3): # Including 3 RH neutrinos
    
    M   = v[0]
    ast = v[1]

    FSM  = fSM(M, ast)
    FRH1 = fRH(M, ast, M1)           # 1 RH neutrino evaporation contribution
    FRH2 = fRH(M, ast, M2)           # 2 RH neutrino evaporation contribution
    FRH3 = fRH(M, ast, M3)           # 3 RH neutrino evaporation contribution
    FT   = FSM + FRH1 + FRH2 + FRH3  # Total Evaporation contribution

    GSM  = gSM(M, ast)
    GRH1 = gRH(M, ast, M1)           # 1 RH neutrino evaporation contribution
    GRH2 = gRH(M, ast, M2)           # 2 RH neutrino evaporation contribution
    GRH3 = gRH(M, ast, M3)           # 3 RH neutrino evaporation contribution
    GT   = GSM + GRH1 + GRH2 + GRH3  # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * kappa * FT * M**-2
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GT - 2.*FT)

    return [dMdtl, dastdtl]

def Itau_NSC(tl, v, Nscls): # Dark Radiation Case
    
    M   = v[0]
    ast = v[1]

    FSM = fSM(M, ast) + gg * phi_g(M, ast, 0.)  # Including gravitons
    FNS = Nscls * phi_s(M, ast, 0.) # N scalars evaporation contribution
    FT  = FSM + FNS                 # Total Evaporation contribution

    GSM = gSM(M, ast) + gg * gam_g(M, ast, 0.)  # Including gravitons
    GNS = Nscls * gam_s(M, ast, 0.) # N scalars evaporation contribution
    GT  = GSM + GNS                 # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * kappa * FT * M**-2
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GT - 2.*FT)

    return [dMdtl, dastdtl]


# Determining the scale fator where PBHs evaporate

def afin(aexp, rPBHi, rRadi, t, ail):

    a = [10.**(aexp[0])]

    ain = 10.**ail # Initial scale factor
    
    A = -ain * rPBHi * sqrt(GCF * (ain * rPBHi + rRadi))
    B = a[0] * rPBHi * sqrt(GCF * (a[0] * rPBHi + rRadi))
    C = 2. * rRadi * (sqrt(GCF*(ain * rPBHi + rRadi)) - sqrt(GCF*(a[0]*rPBHi + rRadi)))
    D = GCF * sqrt(6.*pi) * rPBHi**2
    
    return [A + B + C - D*t]

#-------------------------------------------------------------------------------------------------#
#                                   g*(T) and g*S(T) interpolation                                #
#-------------------------------------------------------------------------------------------------#
import os

path, filename = os.path.split(os.path.realpath(__file__))
data_dir = os.path.dirname(path+"/Data/")

datg  = os.path.join(data_dir, "gstar.dat")
datgS = os.path.join(data_dir, "gstarS.dat")

gTab  = np.loadtxt(datg)
gSTab = np.loadtxt(datgS)

Ttab = gTab[:,0]
gtab = gTab[:,1]
tck  = interpolate.splrep(Ttab, gtab, s=0)

def gstar(T): return interpolate.splev(T, tck, der=0)

def dgstardT(T): return interpolate.splev(T, tck, der = 1)

TStab = gSTab[:,0]
gstab = gSTab[:,1]
tckS  = interpolate.splrep(TStab, gstab, s=0)

def gstarS(T): return interpolate.splev(T, tckS, der = 0)

def dgstarSdT(T): return interpolate.splev(T, tckS, der = 1)


#------------------------------------------------------------------------------------------------#
#                      Unintegrated evaporation rate per momentum per time                       #
#------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------------------------------------------------------------------------#
#
#                                            Greybody Factors for Different Spin fields                                               #
#
#-------------------------------------------------------------------------------------------------------------------------------------#
at = os.path.join(data_dir, "Absxsec/sigma_K_a.dat")
Et = os.path.join(data_dir, "Absxsec/sigma_K_E.dat")

atab = np.loadtxt(at, delimiter="\t")
Etab = np.loadtxt(Et, delimiter="\t")

#-------------------------------#
#            Scalars            #
#-------------------------------#

ss_dir = os.path.join(data_dir, "Absxsec/sigma_s_0.dat")
s0Tab = np.loadtxt(ss_dir)
sig_0 = interpolate.interp1d(s0Tab[:,0], s0Tab[:,1], kind='linear', bounds_error=False, fill_value=(0., 27.))

def G0(x):

    if 0. <= x and x <= 1.:
        sigma = sig_0(x)

    else:
        sigma = 27.*(1. - spherical_jn(0,2.*np.sqrt(27.)*np.pi*x))

    return sigma*x**2

#---------------------------------------#
#           Fermion                     #
#---------------------------------------#

sf_dir = os.path.join(data_dir, "Absxsec/sigma_s_0.5.dat")
sfTab = np.loadtxt(sf_dir)
sig_f = interpolate.interp1d(sfTab[:,0], sfTab[:,1], kind='linear', bounds_error=False, fill_value=(0., 27.))

def G05(x):

    if 0. <= x and x <= 1:
        sigma = sig_f(x)

    else:
        sigma = 27.*(1. - np.sqrt(1/(4.*np.sqrt(27.)*x))*jv(2.5,2.*np.sqrt(27.)*np.pi*x))

    return sigma*x**2


#-------------------------------#
#            Vectors            #
#-------------------------------#

sv_dir = os.path.join(data_dir, "Absxsec/sigma_s_1.dat")
s1Tab = np.loadtxt(sv_dir)
sig_1 = interpolate.interp1d(s1Tab[:,0], s1Tab[:,1], kind='linear', bounds_error=False, fill_value=(0., 27.))

def G1(x):

    if 0. <= x and x <= 1.:
        sigma = sig_1(x)

    else:
        sigma = 27.*(1. - spherical_jn(0, 2.*np.sqrt(27.)*np.pi*x))

    return sigma*x**2


def G1_v2(x):

    if 0.0142 <= x and x <= 1.:
        sigma = sig_1(x)

    if 1.0 < x :
        sigma = 27.*(1. - spherical_jn(0, 2.*np.sqrt(27.)*np.pi*x))

    if x < 0.0142:
        sigma = ((64.0*np.pi)/(3.0))  * x**2 / np.pi

    return sigma*x**2



#-------------------------------#
#           Graviton            #
#-------------------------------#

sg_dir = os.path.join(data_dir, "grav_absxsec.dat")
s2Tab  = np.loadtxt(sg_dir)
sig_2 = interpolate.interp1d(s2Tab[:,0], s2Tab[:,1], kind='linear', bounds_error=False, fill_value=(0., 27.))


def G2(x): # p in GeV, MBH in g

    if x <= 2.e-1:
        sigma = 0.145442 * (x/0.202878) ** 3
    elif x > 2.e-1 and x <= 4.:
        sigma =  abs(sig_2(x))
    elif x > 4.:
        sigma =  27.*(1. - spherical_jn(0, 2.*pi*sqrt(27)*x))
    else:
        sigma =  0.

    return sigma*x**2

    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#             Tables --  Kerr BHs                 #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

ss_K_dir = os.path.join(data_dir, "Absxsec/sigma_K_s_0.dat")
sv_K_dir = os.path.join(data_dir, "Absxsec/sigma_K_s_0.5.dat")
sf_K_dir = os.path.join(data_dir, "Absxsec/sigma_K_s_1.dat")
sg_K_dir = os.path.join(data_dir, "Absxsec/sigma_K_s_2.dat")

Kstab = np.loadtxt(ss_K_dir, delimiter="\t")
Kftab = np.loadtxt(sf_K_dir, delimiter="\t")
Kvtab = np.loadtxt(sv_K_dir, delimiter="\t")
Kgtab = np.loadtxt(sg_K_dir, delimiter="\t")

sig_Ks = interpolate.interp2d(Etab, atab, abs(Kstab), kind='linear',
                              bounds_error=False, fill_value=0.)#RectBivariateSpline(atab, Etab, Kstab)
sig_Kf = interpolate.interp2d(Etab, atab, abs(Kftab), kind='linear',
                              bounds_error=False, fill_value=0.)#RectBivariateSpline(atab, Etab, Kftab)
sig_Kv = interpolate.interp2d(Etab, atab, abs(Kvtab), kind='linear',
                              bounds_error=False, fill_value=0.)#RectBivariateSpline(atab, Etab, Kvtab)
    
#-------------------------------#
#            Scalars            #
#-------------------------------#


def d2Ns_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GCF * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)

    if (ast == 0.):
        return 0.159154943*(G0(x)/(np.exp(p/TBHK) - 1.))
    else:
        return sig_Ks(x, ast)

#---------------------------------------#
#           Massless Fermion            #
#---------------------------------------#


def d2Nf_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GCF * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)
    
    if (ast == 0.):
        return 0.159154943*(G05(x)/(np.exp(p/TBHK) + 1.))
    else:
        return sig_Kf(x, ast)

#---------------------------------------#
#            Massive Fermion            #
#---------------------------------------#  

x_dir = os.path.join(data_dir, "Absxsec/sigma_m_0.5_x.dat")
m_dir = os.path.join(data_dir, "Absxsec/sigma_m_0.5_m.dat")   
G05m_dir = os.path.join(data_dir, "Absxsec/sigma_m_0.5.dat")

G05mtab = np.loadtxt(G05m_dir, delimiter="\t")

mtab  = np.loadtxt(m_dir, delimiter="\t")
Emtab = np.loadtxt(x_dir, delimiter="\t")
G05m    = interpolate.interp2d(Etab, atab, abs(Kstab), kind='linear',
                               bounds_error=False, fill_value=0.)

def d2Nmf_dpdt(p, MBH, ast, m): # p in GeV, MBH in g
    fact = (MBH/GeV_in_g)
    x    = GCF * fact * p # Dimensionless momentum ->  x = G*MBH*p
    mu   = GCF * fact * m # Dimensionless mass     -> mu = G*MBH*m
    TBHK = TBH(MBH, ast)

    if(ast == 0.):
        return 0.159154943*(G05m(x, mu)/(np.exp(np.sqrt(p*p + m*m)/TBHK) + 1.))
    else:
        return sig_Kf(x, ast)

#-------------------------------#
#            Vectors            #
#-------------------------------#

def d2Nv_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GCF * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)
    
    if (ast == 0.):
        return 0.159154943*(G1(x)/(np.exp(p/TBHK) - 1.))
    else:
        return sig_Kv(x, ast)

#-------------------------------#
#            Vectors            #
#-------------------------------#

def d2Nmv_dpdt(p, MBH, ast, m): # p in GeV, MBH in g

    x    = GCF * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)
    
    if (ast == 0.):
        return 0.159154943*(G1(x)/(np.exp(np.sqrt(p**2 + m**2)/TBHK) - 1.))
    else:
        return sig_Kv(x, ast)
    
#-------------------------------#
#           Graviton            #
#-------------------------------#

sig_Kg = interpolate.interp2d(Etab, atab, abs(Kgtab), kind='linear',
                              bounds_error=False, fill_value=0.) #RectBivariateSpline(atab, Etab, Kgtab)

def d2Ng_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GCF * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)

    if (ast == 0.):
        return 0.159154943*(G2(x)/(np.exp(p/TBHK) - 1.))
    #elif x < 1.e-10:
    #    return 0.159154943*(G2(x)/(np.exp(p/TBHK) - 1.))
    else:
        return sig_Kg(x, ast)[0]
