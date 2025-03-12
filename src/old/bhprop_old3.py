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


from numpy import sqrt, log, exp, log10, pi, tanh, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, real

from collections import OrderedDict
olderr = np.seterr(all='ignore')

# Constants

c     = 299792.458       # in km/s
gamma = sqrt(3.)**-3.    # Collapse factor
GN    = 6.70883e-39      # Gravitational constant in GeV^-2
mPL   = 1./sqrt(GN)      # Planck mass in GeV
v     = 174              # Higgs vev
csp   = 0.35443          # sphaleron conversion factor
GF    = 1.1663787e-5     # Fermi constant in GeV^-2
LQCD  = 0.2              # Lambda QCD in GeV
TEW   = 159.5            # Electroweak phase transition temperature in GeV

# Conversion factors

GeV_in_g     = 1.782661907e-24  # 1 GeV in g
Mpc_in_cm    = 3.085677581e24   # 1 Mpc in cm

cm_in_invkeV = 5.067730938543699e7       # 1 cm in keV^-1
year_in_s    = 3.168808781402895e-8      # 1 year in s
GeV_in_invs  = cm_in_invkeV * c * 1.e11  # 1 GeV in s^-1

MPL   = mPL * GeV_in_g        # Planck mass in g
kappa = mPL**4 * GeV_in_g**3  # Evaporation constant in g^3 * GeV -- from PRD41(1990)3052
mPL_red = 1./sqrt(8.*pi*GN)   # Reduced mass Planck

# Particle masses, in GeV

mW   = 80.379
mZ   = 91.1876
mH   = 125.18
me   = 0.5109989461e-3
mmu  = 105.6583745e-3
mtau = 1.77686
mu   = 336e-3#2.2e-3#
md   = 340e-3#4.6e-3#
ms   = 486e-3#95e-3#
mc   = 1.275
mb   = 4.18
mt   = 173.1
mg   = 0.200        # Ficticious gluon mass ---> indicates the QCD phase transition, following PRD41(1990)3052

m_pi0 = 0.1349768   # Neutral pion mass
m_pic = 0.13957039  # Charged pion mass
m_Kc  = 493.677     # Charged Kaon
m_K0  = 497.611     # Neutral Kaon

# Particles' lifetime, in s

tau_W   = 2.202/GeV_in_invs
tau_Z   = 2.4955/GeV_in_invs
tau_H   = 3.7e-3/GeV_in_invs
tau_mu  = 2.1969811e-6
tau_tau = 2.903e-13 
tau_t   = 1.42/GeV_in_invs

tau_pic = 2.60e-8 # charged pions in s
tau_pi0 = 8.5e-17 # pi 0

tau_Kc  = 1.2380e-8

# Neutrino parameters, from NuFit6, assuming NO and taking the results with SK

Dm31 = 2.513e-3 # Atmospheric quadratic neutrino mass difference, in eV^2
Dm21 = 7.490e-5 # Solar quadratic neutrino mass difference, in eV^2

# Degrees of freedom of the SM 

# ---> Above the EW phase transition
gW_aEW  = 2.*2.     # W
gZ_aEW  = 2.        # Z
gH_aEW  = 4.        # Higgs

# ---> Below the EW phase transition
gW  = 2.*3.     # W
gZ  = 3.        # Z
gH  = 1.        # Higgs
gp  = 2.        # photon
gg  = 2.        # graviton

ggl = 8.*2.     # gluons
gl  = 2.*2.     # leptons
gq  = 2.*2.*3   # quarks
gnu = 2.        # LH neutrino

gf = 3.*gnu + 3.*gl + 6.*gq   # Total number of SM fermion dofs 
gs = gH                       # Total number of SM scalar dofs
gv = gW + gZ + gp + gg + ggl  # Total number of SM vector dofs

g_pi0 = 1 # Neutral pion
g_pic = 2 # Charged pion

f_QCD = 1 # Parameter to constrain how fast we change between quarks to hadrons degrees-of-freedom

#---------------------------------------------------------------------------------------------------------------------#
#                                                  BH Temperature in GeV                                              #
#---------------------------------------------------------------------------------------------------------------------#

def TBH(M, astar):

    M_GeV = M/GeV_in_g
    
    return (1./(4.*pi*GN*M_GeV))*(sqrt(abs(1. - astar**2))/(1. + sqrt(abs(1. - astar**2)))) # M in g

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                  Momentum Integrated Rate for Kerr BHs                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#


def Gamma_S(M, ast, m):# Scalar, in GeV

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)
    
    hs = 10.**(0.3891202551434314 - 0.027358152833834106*ast - 0.022376552767462348*ast**2 
               + (0.00009835559447136233*ast**2)/(-1.025 + ast)**2 - 0.4821918307149832*ast**3 + 0.12926706388940493*ast**4)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [0.940377, -0.689675, 4.44686, -10.9065, 7.37227, -0.000318996]
        b0, b1, b2, b3, b4, b5 = [7.72257, -0.748929, -0.338703, -11.9697, 6.82825, -0.000384524]
        c0, c1, c2, c3, c4, c5 = [-0.399598, 0.0133639, -1.3773, 7.0668, -6.27621, 0.000629345]
        
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        z = GM * m

        In = hs * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
        
    else:
        
        In = hs
    
    return  (27/(1024. * pi**4 * GM)) * In


def Gamma_F(M, ast, m):# Fermion

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    hf = 10.**(-0.04716796508441029 + 0.00036421843653813835*ast + 0.6381107254113885*ast**2 
               + (0.0000234617959684773*ast**2)/(-1.025 + ast)**2 - 0.16089673280158395*ast**3 - 0.11259864747002547*ast**4)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [1.08606, 0.0536342, -0.518663, 1.03296, -1.17882, 0.000221855]
        b0, b1, b2, b3, b4, b5 = [9.09361, -1.19198, 2.88914, -15.438, 7.37464, -0.00161798]
        c0, c1, c2, c3, c4, c5 = [-0.486654, 0.277357, -1.26955, 2.25462, -0.527783, -0.000420154]

        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)
        
        z = GM * m

        In = hf * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
                        
    else:
        
        In = hf
    
    return  (27/(1024. * pi**4 * GM)) * In


def Gamma_V(M, ast, m):# Vector

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)
    
    hv = 10.**(-0.5636366140508233 - 0.08021106448222838*ast + 3.295380399704949*ast**2 
               + (0.00009510434823780674*ast**2)/(-1.025 + ast)**2 - 3.684397070950851*ast**3 + 1.838594066120566*ast**4)
    
    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [1.21236, 0.0930657, -0.606395, 1.53686, -1.42133, -0.0000789555]
        b0, b1, b2, b3, b4, b5 = [10.0163, -1.71734, 3.48126, -9.53202, 0.552603, -0.000291326]
        c0, c1, c2, c3, c4, c5 = [-0.54568, -0.212927, 1.37124, -3.02044, 2.41605, 0.0000852966]

        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        z = GM * m

        In = hv * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
                                
    else:
        
        In = hv
    
    return  (27/(1024. * pi**4 * GM)) * In

def Gamma_G(M, ast, m):# Spin 2

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    hg = 10.**(-1.6874671321658943 + 0.02894868683289116*ast + 8.98016696161364*ast**2 
               + (0.0003004670661671682*ast**2)/(-1.025 + ast)**2 - 11.901307189829502*ast**3 + 6.314005790871132*ast**4)

    if m > 0.:
        
        if ast <= 1.e-5:

            B, C, nu = [22.325, -21.2326, 0.12076]

            z = m/TKBH

        else:

            a0, a1, a2, a3, a4, a5 = [1.4105, 0.105735, -0.355946, 1.23979, -1.11272, -0.000061655]
            b0, b1, b2, b3, b4, b5 = [11.0087, -1.96276, -1.01536, 4.59208, -11.019, -0.000842485]
            c0, c1, c2, c3, c4, c5 = [-0.696409, -0.119079, 0.485082, -1.44864, 1.24649, 0.0000742369]

            B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
            C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
            nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

            z = GM * m

        In = hg * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu)) # DM emission rate including greybody factors
                                        
    else:
        
        In = hg
    
    return  (27/(1024. * pi**4 * GM)) * In

def Gamma_GO(M, ast, m):# Geometric optics limit

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    zBH = m/TKBH

    In = - zBH * polylog(2, -exp(-zBH)) - polylog(3, -exp(-zBH))# DM emission rate including greybody factors
    
    return  2. * (27/(1024. * pi**4 * GM)) * In

def Gamma_DM(M, ast, mdm, s):

    if s == 0.:
        f = Gamma_S(M, ast, mdm)
    elif s == 0.5:
        f = 2. * Gamma_F(M, ast, mdm) # Assuming Majorana DM
    elif s == 1.:
        f = 3. * Gamma_V(M, ast, mdm)
    elif s == 2.:
        f = 5. * Gamma_G(M, ast, mdm)
        
    return f

def Gamma_DR(M, ast, mdm, s):

    if s == 0.:
        f = Gamma_S(M, ast, mdm)
    elif s == 0.5:
        f = 2.*Gamma_F(M, ast, mdm) # Assuming Majorana DM
    elif s == 1.:
        f = 2.*Gamma_V(M, ast, mdm)
    elif s == 2.:
        f = 2.*Gamma_G(M, ast, mdm)
        
    return f

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                         Total f functions ---> related to the mass rate, dM/dt                                      #
#                                                   Counting SM dofs + Dark Radiation                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------#
#                     f = - M^2 dM/dt fitted functions                      #
#-------------------------------------------------------------------#

def fs(astar): return (-4.128749655067042 - 0.16759901907033878*astar + 1.290173196518525*astar**2 
                       + (0.00016365496466816672*astar**2)/(-1.025 + astar)**2 - 3.4942374252463737*astar**3 + 2.74961998847241*astar**4)

def ff(astar): return (-4.388102600029355 - 0.1128003671589209*astar + 1.8337492370718769*astar**2 
                       + (0.00016817382335538385*astar**2)/(-1.025 + astar)**2 - 2.5125751027117*astar**3 + 1.5825520279914687*astar**4)
    
def fv(astar): return (-4.774460038909468 - 0.12611258859190277*astar + 4.127986304376674*astar**2 
                       + (0.0001797800275056758*astar**2)/(-1.025 + astar)**2 - 5.095527421651263*astar**3 + 2.7922511175912317*astar**4)

def fg(astar): return (-5.716053544318367 + 0.028216961187875027*astar + 9.761134714039448*astar**2 
                       + (0.00036146596930611107*astar**2)/(-1.025 + astar)**2 - 13.168596070839765*astar**3 + 7.148963046631307*astar**4)

#--------------------------------------------------------------------------------#
#              Our interpolated forms including the particle's mass              #
#--------------------------------------------------------------------------------#

# Scalar

def phi_s(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f0 = 10.**fs(ast)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [0.87796, 1.36825, -8.43348, 14.1271, -7.16721, 0.000160475]
        b0, b1, b2, b3, b4, b5 = [7.01903, 2.00973, -21.2264, 23.5781, -10.1526, -0.00198272]
        c0, c1, c2, c3, c4, c5 = [-0.229276, -2.44717, 15.4855, -28.3087, 15.4948, -0.000336993]

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

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f12  = 10.**ff(ast)

    if m > 0.:
        
        a0, a1, a2, a3, a4, a5 = [1.11298, -0.63483, 3.61466, -8.26333, 5.05179, 0.0000357659]
        b0, b1, b2, b3, b4, b5 = [8.52617, 3.04334, -26.1905, 32.0687, -16.8691, -0.00110689]
        c0, c1, c2, c3, c4, c5 = [-0.461234, 1.27322, -7.67927, 17.2333, -10.9567, 0.00012094]

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

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f1 = 10.**fv(ast)

    if m > 0.:
        
        if ast <= 1.e-5:

            B, C, nu = [14.0361, -10.7138, 0.307206]

            z = m/TKBH

        else:

            a0, a1, a2, a3, a4, a5 = [1.22506, -0.00982, -0.0614102, -0.0279036, -0.477626, 0.000106361]
            b0, b1, b2, b3, b4, b5 = [9.76473, -2.11804, 4.91754, -24.4636, 14.4807, -0.00147742]
            c0, c1, c2, c3, c4, c5 = [-0.52625, 0.0171385, 0.13962, -0.185154, 0.836461, -0.000343343]

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

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f2 = 10.**fg(ast)

    if m > 0.:
        
        if ast <= 1.e-5:

            B, C, nu = [21.50941, -20.5135, 0.173423]

            z = m/TKBH

        else:

            a0, a1, a2, a3, a4, a5 = [1.43093, 0.0747246, -0.357396, 1.16272, -1.19794, -0.0000321492]
            b0, b1, b2, b3, b4, b5 = [11.0979, -3.39601, 5.25366, -14.2897, 2.25185, -0.000433874]
            c0, c1, c2, c3, c4, c5 = [-0.700339, -0.0978541, 0.582046, -1.59121, 1.49295, 0.000043548]
            
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

    T = TBH(M, ast)
    
    ''' 
    Contribution from each particle --> We do not include the Graviton contribution here.
    For BH temperatures larger than the EW phase transition, we consider massless gauge bosons with 2 dofs and 4 scalar dofs
    for below, we consider massive gauge bosons and 1 scalar dof
    '''

    fgr =  0.
    fp  =  gp * phi_v(M, ast, 0.)  # Photon
    fgl = ggl * phi_v(M, ast, 0.6) # Gluon

    # Electroweak dofs

    fGB_aEW = gW_aEW * phi_v(M, ast, 0.)  + gZ_aEW * phi_v(M, ast, 0.)  + gH_aEW * phi_s(M, ast, 0.)  # above EW    

    fGB_bEW = gW * phi_v(M, ast, mW)  + gZ * phi_v(M, ast, mZ)  + gH * phi_s(M, ast, mH)  # below EW

    fGB = 0.5*(fGB_aEW + fGB_bEW + (fGB_aEW - fGB_bEW)*tanh((T-TEW)/10.))

    fnu = 3. * gnu * phi_f(M, ast, 0.) # Active neutrinos
    
    fl  = gl * (phi_f(M, ast, me) + phi_f(M, ast, mmu) + phi_f(M, ast, mtau))  # Charged leptons

    # QCD dofs


    fq_aLQCD  = gq * (phi_f(M, ast, mu) + phi_f(M, ast, md) + phi_f(M, ast, ms) +
                phi_f(M, ast, mc) + phi_f(M, ast, mb) + phi_f(M, ast, mt))    # Quarks

    fq_bLQCD  = g_pi0 * phi_s(M, ast, m_pi0) + g_pic * phi_f(M, ast, m_pic)

    fq = 0.5*(fq_aLQCD + fq_bLQCD + (fq_aLQCD - fq_bLQCD)*tanh((T-LQCD)/f_QCD))
    
    return fgr + fp + fnu + fgl + fGB + fl + fq

#--------------------------------------------------------------#
#              SM Contribution + massive neutrinos             #
#--------------------------------------------------------------#

def fSM_nu(M, ast, m0):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * phi_v(M, ast, 0.)  # Photon
    fgl = ggl * phi_v(M, ast, 0.6) # Gluon

    # Electrowak dofs

    fGB_aEW = gW_aEW * phi_v(M, ast, 0.)  + gZ_aEW * phi_v(M, ast, 0.)  + gH_aEW * phi_s(M, ast, 0.)  # above EW    

    fGB_bEW = gW * phi_v(M, ast, mW)  + gZ * phi_v(M, ast, mZ)  + gH * phi_s(M, ast, mH)  # below EW

    fGB = 0.5*(fGB_aEW + fGB_bEW + (fGB_aEW - fGB_bEW)*tanh((T-TEW)/10.))

    # Neutrino masses, assuming Normal Ordering, in GeV

    m1 = m0*1.e-9 
    m2 = sqrt(Dm21 + m0*m0)*1.e-9 
    m3 = sqrt(Dm31 + m0*m0)*1.e-9 

    fnu = gnu * (phi_f(M, ast, m1) + phi_f(M, ast, m2) + phi_f(M, ast, m3)) # Active Majorana neutrinos
    
    fl  = gl * (phi_f(M, ast, me) + phi_f(M, ast, mmu) + phi_f(M, ast, mtau))  # Charged leptons

    # QCD dofs

    fq_aLQCD  = gq * (phi_f(M, ast, mu) + phi_f(M, ast, md) + phi_f(M, ast, ms) +
                phi_f(M, ast, mc) + phi_f(M, ast, mb) + phi_f(M, ast, mt))    # Quarks

    fq_bLQCD  = g_pi0 * phi_s(M, ast, m_pi0) + g_pi0 * phi_s(M, ast, m_pi0)

    fq = 0.5*(fq_aLQCD + fq_bLQCD + (fq_aLQCD - fq_bLQCD)*tanh((T-LQCD)/f_QCD))

    
    return fgr + fp + fnu + fgl + fGB + fl + fq

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

# SUSY contribution, assuming all partners with the same mass Lambda

def fSUSY(M, ast, Lambda):
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fnt =  4 * 2 * phi_f(M, ast, Lambda) # Neutralinos,  4 -> particle number, 2 -> Majorana
    fcg =  2 * 4 * phi_f(M, ast, Lambda) # Charginos,    2 -> particle number, 4 -> Dirac
    fgl =  8 * 2 * phi_f(M, ast, Lambda) # Gluinos,      8 -> particle number, 2 -> Majorana

    fH  =  4 * phi_s(M, ast, Lambda)     # Higgs bosons, not including h0 -> SM one
    
    fsl =  9 * phi_s(M, ast, Lambda)     # Sleptons,     9 -> particle number
    fsq = 12 * 3 * phi_s(M, ast, Lambda) # Squarks,     12 -> particle number, 3 -> color

    ftot = fnt + fcg + fgl + fH + fsl + fsq
    
    return ftot

# Dark Sector with N copies of the Standard Model

def fSM_DS(M, ast, L_DS):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fp  =  gp * phi_v(M, ast, L_DS)  # Photon
    fgl = ggl * phi_v(M, ast, L_DS) # Gluon

    # Electroweak dofs
    fGB = gW * phi_v(M, ast, L_DS)  + gZ * phi_v(M, ast, L_DS)  + gH * phi_s(M, ast, L_DS)

    # Neutrino masses, assuming Normal Ordering, in GeV

    fnu = gnu * (phi_f(M, ast, L_DS) + phi_f(M, ast, L_DS) + phi_f(M, ast, L_DS)) # Active Majorana neutrinos
    
    fl  = gl * (phi_f(M, ast, L_DS) + phi_f(M, ast, L_DS) + phi_f(M, ast, L_DS))  # Charged leptons

    # QCD dofs

    fq  = gq * (phi_f(M, ast, L_DS) + phi_f(M, ast, md) + phi_f(M, ast, L_DS) +
                phi_f(M, ast, L_DS) + phi_f(M, ast, L_DS) + phi_f(M, ast, L_DS))    # Quarks
    
    return fp + fnu + fgl + fGB + fl + fq

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                Total g functions ---> related to the angular momentum rate, da_*/dt                                 #
#                                                   Counting SM dofs + Dark Radiation                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

def gs(astar): return (-4.051551680929044 - 0.0906840762040224*astar + 1.4494270120860604*astar**2 
                       + (0.00012404227920010614*astar**2)/(-1.025 + astar)**2 - 1.9117571961975526*astar**3 + 1.3126804298107988*astar**4)
    
def gf(astar): return (-3.5115113216682095 - 0.0809224871511442*astar 
                       + 0.7847225497595978*astar**2 + (0.00013240801697621499*astar**2)/(-1.025 + astar)**2 - 1.6431128012131193*astar**3 + 1.3093862882449498*astar**4)

def gv(astar): return (-3.620619113318074 - 0.10052289144242625*astar + 1.7817866500602548*astar**2 
                       + (0.0001490487102809037*astar**2)/(-1.025 + astar)**2 - 2.212048942765569*astar**3 + 1.5222029958417689*astar**4)

def gG(astar): return (-4.27332250841818 - 0.19349041845338877*astar + 6.191951514721466*astar**2 
                       + (0.0003260121463999355*astar**2)/(-1.025 + astar)**2 - 7.669134056579806*astar**3 + 4.432340856016421*astar**4)

#--------------------------------------------------------------------------------#
#              Our interpolated forms including the particle's mass              #
#--------------------------------------------------------------------------------#

# Scalar

def gam_s(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    g0 = 10.**gs(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.23912, -0.145684, 0.46855, -1.70819, 0.874656, 0.000170998]
        b0, b1, b2, b3, b4, b5 = [8.89446, 1.01101, -17.457, 13.3638, -4.51256, -0.00179104]
        c0, c1, c2, c3, c4, c5 = [-0.584079, 0.489567, -2.49244, 5.81403, -3.35954, -0.000260367]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = g0 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = g0
    
    return In

# Fermion

def gam_f(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    g12 = 10.**gf(ast)

    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.05092, -0.404388, 2.14148, -5.74908, 3.88458, -0.0000102967]
        b0, b1, b2, b3, b4, b5 = [7.65199, 2.75823, -25.7826, 34.0066, -18.1279, -0.0008927]
        c0, c1, c2, c3, c4, c5 = [-0.422124, 0.628113, -3.72642, 9.9281, -7.0558, 0.000168765]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = g12 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))        
                
    else:
        
        In = g12

    return In

# Vector

def gam_v(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    g1 = 10.**gv(ast)

    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.21193, 0.00812571, -0.279915, 0.210145, -0.488696, 0.0000735645]
        b0, b1, b2, b3, b4, b5 = [9.53409, -1.00818, -1.25682, -14.2174, 9.4546, -0.00124971]
        c0, c1, c2, c3, c4, c5 = [-0.53927, -0.00212104, 0.326782, -0.32044, 0.740241, -0.000269688]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = g1 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
                        
    else:
        
        In = g1

    return In

# Tensor - spin2

def gam_g(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    g2 = 10.**gG(ast)

    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.43243, 0.0234646, -0.309702, 1.11494, -1.14007, -0.0000375206]
        b0, b1, b2, b3, b4, b5 = [11.1005, -3.13161, 3.24023, -10.6557, 0.49634, -0.000495192]
        c0, c1, c2, c3, c4, c5 = [-0.714614, -0.0460537, 0.495306, -1.45766, 1.382, 0.0000487401]
    
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

    T = TBH(M, ast)

    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.                      # Graviton
    fp  =  gp * gam_v(M, ast, 0.)  # Photon
    fgl = ggl * gam_v(M, ast, 0.6) # Gluon


    # Electrowak dofs

    fGB_aEW = gW_aEW * gam_v(M, ast, 0.)  + gZ_aEW * gam_v(M, ast, 0.)  + gH_aEW * gam_s(M, ast, 0.)  # below EW    

    fGB_bEW = gW * gam_v(M, ast, mW)  + gZ * gam_v(M, ast, mZ)  + gH * gam_s(M, ast, mH)  # below EW

    fGB = 0.5*(fGB_aEW + fGB_bEW + (fGB_aEW - fGB_bEW)*tanh((T-TEW)/10.))


    fnu = 3. * gnu * gam_f(M, ast, 0.)                           # Active neutrinos
    
    fl  = gl * (gam_f(M, ast, me) + gam_f(M, ast, mmu) + gam_f(M, ast, mtau))  # Charged leptons

    fq_aLQCD  = gq * (gam_f(M, ast, mu) + gam_f(M, ast, md) + gam_f(M, ast, ms) +
                      gam_f(M, ast, mc) + gam_f(M, ast, mb) + gam_f(M, ast, mt))    # Quarks

    fq_bLQCD  = g_pi0 * gam_s(M, ast, m_pi0) + g_pi0 * gam_s(M, ast, m_pi0)

    fq = 0.5*(fq_aLQCD + fq_bLQCD + (fq_aLQCD - fq_bLQCD)*tanh((T-LQCD)/f_QCD))

    return fgr + fp + fnu + fgl + fGB + fl + fq

def gSM_test(M, ast):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * gam_v(M, ast, 0.)  # Photon
    fnu = 3. * gnu * gam_f(M, ast, 0.) # Active neutrinos

    if T >= mW:  fW =  gW * gam_v(M, ast, 0.) 
    else: fW = 0

    if T >= mZ:  fZ =  gZ * gam_v(M, ast, 0.) 
    else: fZ = 0.

    if T >= mH:  fH =  gH * gam_s(M, ast, 0.)  # Higgs
    else: fH = 0.

    if T >= 0.6: fgl = ggl * gam_v(M, ast, 0.) # Gluon
    else: fgl = 0.

    if T >= mtau: ftau = gl*gam_f(M, ast, 0.)
    else: ftau = 0.

    if T >= mmu:  fmu = gl*gam_f(M, ast, 0.)
    else: fmu = 0.

    if T >= me:   fe = gl*gam_f(M, ast, 0.)
    else: fe = 0.

    fl = fe + fmu + ftau

    if T >= LQCD:

        if T >= mt: ft = gq * gam_f(M, ast, 0.)
        else: ft = 0.

        if T >= mb: fb = gq * gam_f(M, ast, 0.)
        else: fb = 0.

        if T >= mc: fc = gq * gam_f(M, ast, 0.)
        else: fc = 0.

        if T >= ms: fs = gq * gam_f(M, ast, 0.)
        else: fs = 0.

        if T >= md: fd = gq * gam_f(M, ast, 0.)
        else: fd = 0.

        if T >= mu: fu = gq * gam_f(M, ast, 0.)
        else: fu = 0.

        fq = fu + fd + fs + fc + fb + ft

        
    else: # Below Lambda_QCD we only include pions


        if T >= m_pi0: fpi0 = g_pi0 * gam_f(M, ast, 0.)
        else: fpi0 = 0.

        if T >= m_pic: fpic = g_pic * gam_f(M, ast, 0.)
        else: fpic = 0.

        fq = fpi0 + fpic 
    
    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq

#--------------------------------------------------------------#
#              SM Contribution + massive neutrinos             #
#--------------------------------------------------------------#

def gSM_nu(M, ast, m0):

    T = TBH(M, ast)

    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.                      # Graviton
    fp  =  gp * gam_v(M, ast, 0.)  # Photon
    fgl = ggl * gam_v(M, ast, 0.6) # Gluon

    # Electrowak dofs

    fGB_aEW = gW_aEW * gam_v(M, ast, 0.)  + gZ_aEW * gam_v(M, ast, 0.)  + gH_aEW * gam_s(M, ast, 0.)  # below EW    

    fGB_bEW = gW * gam_v(M, ast, mW)  + gZ * gam_v(M, ast, mZ)  + gH * gam_s(M, ast, mH)  # below EW

    fGB = 0.5*(fGB_aEW + fGB_bEW + (fGB_aEW - fGB_bEW)*tanh((T-TEW)/10.))

    # Neutrino masses, assuming Normal Ordering, in GeV

    m1 = m0*1.e-9 
    m2 = sqrt(Dm21 + m0*m0)*1.e-9 
    m3 = sqrt(Dm31 + m0*m0)*1.e-9 

    fnu = gnu * (gam_f(M, ast, m1) + gam_f(M, ast, m2) + gam_f(M, ast, m3)) # Active Majorana neutrinos
    
    fl  = gl * (gam_f(M, ast, me) + gam_f(M, ast, mmu) + gam_f(M, ast, mtau))  # Charged leptons

    # QCD dofs

    fq_aLQCD  = gq * (gam_f(M, ast, mu) + gam_f(M, ast, md) + gam_f(M, ast, ms) +
                      gam_f(M, ast, mc) + gam_f(M, ast, mb) + gam_f(M, ast, mt))    # Quarks

    fq_bLQCD  = g_pi0 * gam_s(M, ast, m_pi0) + g_pi0 * gam_s(M, ast, m_pi0)

    fq = 0.5*(fq_aLQCD + fq_bLQCD + (fq_aLQCD - fq_bLQCD)*tanh((T-LQCD)/f_QCD))

    return fgr + fp + fnu + fgl + fGB + fl + fq


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

# SUSY contribution, assuming all partners with the same mass Lambda

def gSUSY(M, ast, Lambda):
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    gnt =  4 * 2 * gam_f(M, ast, Lambda) # Neutralinos,  4 -> particle number, 2 -> Majorana
    gcg =  2 * 4 * gam_f(M, ast, Lambda) # Charginos,    2 -> particle number, 4 -> Dirac
    ggl =  8 * 2 * gam_f(M, ast, Lambda) # Gluinos,      8 -> particle number, 2 -> Majorana

    gH  =  4 * gam_s(M, ast, Lambda)     # Higgs bosons, not including h0 -> SM one
    
    gsl =  9 * gam_s(M, ast, Lambda)     # Sleptons,     9 -> particle number
    gsq = 12 * 3 * gam_s(M, ast, Lambda) # Squarks,     12 -> particle number, 3 -> color

    gtot = gnt + gcg + ggl + gH + gsl + gsq

    return gtot

# Dark Sector with N copies of the Standard Model

def gSM_DS(M, ast, L_DS):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fp  =  gp * gam_v(M, ast, L_DS)  # Photon
    fgl = ggl * gam_v(M, ast, L_DS) # Gluon

    # Electroweak dofs
    fGB = gW * gam_v(M, ast, L_DS)  + gZ * gam_v(M, ast, L_DS)  + gH * gam_s(M, ast, L_DS)

    # Neutrino masses, assuming Normal Ordering, in GeV

    fnu = gnu * (gam_f(M, ast, L_DS) + gam_f(M, ast, L_DS) + gam_f(M, ast, L_DS)) # Active Majorana neutrinos
    
    fl  = gl * (gam_f(M, ast, L_DS) + gam_f(M, ast, L_DS) + gam_f(M, ast, L_DS))  # Charged leptons

    # QCD dofs

    fq  = gq * (gam_f(M, ast, L_DS) + gam_f(M, ast, md) + gam_f(M, ast, L_DS) +
                gam_f(M, ast, L_DS) + gam_f(M, ast, L_DS) + gam_f(M, ast, L_DS))    # Quarks
    
    return fp + fnu + fgl + fGB + fl + fq

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                 Total zeta functions ---> related to the entropy rate, dSrad/dt                                     #
#                                                   Counting SM dofs + Dark Radiation                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

def ss(astar): return (-2.460323213604857 - 0.03349413973761712*astar + 0.09903041578417396*astar**2 
                       + (0.00009013284038414188*astar**2)/(-1.025 + astar)**2 - 0.5365159920225852*astar**3 + 0.2508482330305566*astar**4)
    
def sf(astar): return (-2.773268849885142 - 0.011169206384581439*astar + 0.4930645901558424*astar**2 
                       + (0.00003924116394805256*astar**2)/(-1.025 + astar)**2 - 0.2416498538441225*astar**3 + 0.11458654058513627*astar**4)

def sv(astar): return (-3.1980258315861803 - 0.08405827596103263*astar + 2.618581209011898*astar**2 
                       + (0.00009887488882198228*astar**2)/(-1.025 + astar)**2 - 2.9135919491836413*astar**3 + 1.5354812403053106*astar**4)

def sg(astar): return (-4.186144418797312 + 0.0005940192025746495*astar + 7.606082144442883*astar**2 
                       + (0.00021124526331598126*astar**2)/(-1.025 + astar)**2 - 9.78104587310765*astar**3 + 5.02725611162615*astar**4)

#--------------------------------------------------------------------------------#
#              Our interpolated forms including the particle's mass              #
#--------------------------------------------------------------------------------#

# Scalar

def zet_s(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    z0 = 10.**ss(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [0.888622, -0.232117, 1.68035, -5.49261, 4.22413, -0.000200455]
        b0, b1, b2, b3, b4, b5 = [6.80584, -0.0656114, -5.16384, -0.361395, -0.048858, -0.000465038]
        c0, c1, c2, c3, c4, c5 = [-0.372997, -0.581049, 2.56896, -1.2217, -1.18751, 0.000369008]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = z0 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = z0

    return In
    

# Fermion

def zet_f(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    z12 = 10.**sf(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.05, -0.0450657, -0.0175852, -0.259348, -0.197823, 0.000192183]
        b0, b1, b2, b3, b4, b5 = [8.16879, 0.313733, -7.60497, 2.68588, -1.27935, -0.00149678]
        c0, c1, c2, c3, c4, c5 = [-0.471581, 0.531858, -2.75174, 5.62418, -2.89043, -0.000298896]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = z12 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = z12

    return In

# Vector

def zet_v(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    z1 = 10.**sv(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [1.17668, 0.05448, -0.526749, 1.08955, -1.06011, -0.0000314984]
        b0, b1, b2, b3, b4, b5 = [9.15761, -1.42107, -0.0724796, -8.38185, 3.31895, -0.000512756]
        c0, c1, c2, c3, c4, c5 = [-0.521575, -0.161093, 1.2139, -2.44029, 2.00427, -0.0000369006]
    
        B  = 10.**(a0 + a1*ast + a2*ast**2 + a3*ast**3 + a4*ast**4 + (a5*ast**2)/(ast-1.025)**2)
        C  = b0 + b1*ast + b2*ast**2 + b3*ast**3 + b4*ast**4 + (b5*ast**2)/(ast-1.025)**2
        nu = 10.**(c0 + c1*ast + c2*ast**2 + c3*ast**3 + c4*ast**4 + (c5*ast**2)/(ast-1.025)**2)

        In = z1 * (1. - (1. + exp(-B * log10(abs(z)) - C))**(-nu))
        
    else:
        
        In = z1
    
    return In

# Tensor - spin2

def zet_g(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1

    z2 = 10.**sg(ast)
    
    if m > 0.:
              
        z = GM * m # Dimensionless parameter -- gravitational coupling GMm

        a0, a1, a2, a3, a4, a5 = [ 1.38097, 0.0297939, -0.241619, 0.946005, -1.04207, -0.0000500249]
        b0, b1, b2, b3, b4, b5 = [10.2653, -3.60948, 4.87505, -12.7666, 2.2968, -0.000473508]
        c0, c1, c2, c3, c4, c5 = [-0.672912, -0.0552404, 0.440382, -1.32336, 1.30022, 0.0000499002]
    
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

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * zet_v(M, ast, 0.)  # Photon
    fgl = ggl * zet_v(M, ast, 0.6) # Gluon
    
    # Electrowak dofs

    fGB_aEW = gW_aEW * zet_v(M, ast, mW)  + gZ_aEW * zet_v(M, ast, 0.)  + gH_aEW * zet_s(M, ast, 0.)  # below EW    

    fGB_bEW = gW * zet_v(M, ast, mW)  + gZ * zet_v(M, ast, mZ)  + gH * zet_s(M, ast, mH)  # below EW

    fGB = 0.5*(fGB_aEW + fGB_bEW + (fGB_aEW - fGB_bEW)*tanh((T-TEW)/10.))

    fnu = 3. * gnu * zet_f(M, ast, 0.) # Active neutrinos
    
    fl  = gl * (zet_f(M, ast, me) + zet_f(M, ast, mmu) + zet_f(M, ast, mtau))  # Charged leptons

    # QCD dofs

    fq_aLQCD  = gq * (zet_f(M, ast, mu) + zet_f(M, ast, md) + zet_f(M, ast, ms) +
                      zet_f(M, ast, mc) + zet_f(M, ast, mb) + zet_f(M, ast, mt))    # Quarks

    fq_bLQCD  = g_pi0 * zet_s(M, ast, m_pi0) + g_pi0 * zet_s(M, ast, m_pi0)

    fq = 0.5*(fq_aLQCD + fq_bLQCD + (fq_aLQCD - fq_bLQCD)*tanh((T-LQCD)/f_QCD))

    
    return fgr + fp + fnu + fgl + fGB + fl + fq

#--------------------------------------------------------------#
#              SM Contribution + massive neutrinos             #
#--------------------------------------------------------------#

def zSM_nu(M, ast, m0):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * zet_v(M, ast, 0.)  # Photon
    fgl = ggl * zet_v(M, ast, 0.6) # Gluon
    
    # Electrowak dofs

    fGB_aEW = gW_aEW * zet_v(M, ast, mW)  + gZ_aEW * zet_v(M, ast, 0.)  + gH_aEW * zet_s(M, ast, 0.)  # below EW    

    fGB_bEW = gW * zet_v(M, ast, mW)  + gZ * zet_v(M, ast, mZ)  + gH * zet_s(M, ast, mH)  # below EW

    fGB = 0.5*(fGB_aEW + fGB_bEW + (fGB_aEW - fGB_bEW)*tanh((T-TEW)/10.))   

    # Neutrino masses, assuming Normal Ordering, in GeV

    m1 = m0*1.e-9 
    m2 = sqrt(Dm21 + m0*m0)*1.e-9 
    m3 = sqrt(Dm31 + m0*m0)*1.e-9 

    fnu = gnu * (zet_f(M, ast, m1) + zet_f(M, ast, m2) + zet_f(M, ast, m3)) # Active Majorana neutrinos
    
    fl  = gl * (zet_f(M, ast, me) + zet_f(M, ast, mmu) + zet_f(M, ast, mtau))  # Charged leptons

    fq_aLQCD  = gq * (zet_f(M, ast, mu) + zet_f(M, ast, md) + zet_f(M, ast, ms) +
                      zet_f(M, ast, mc) + zet_f(M, ast, mb) + zet_f(M, ast, mt))    # Quarks

    fq_bLQCD  = g_pi0 * zet_s(M, ast, m_pi0) + g_pi0 * zet_s(M, ast, m_pi0)

    fq = 0.5*(fq_aLQCD + fq_bLQCD + (fq_aLQCD - fq_bLQCD)*tanh((T-LQCD)/f_QCD))

    return fgr + fp + fnu + fgl + fGB + fl + fq



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

# SUSY contribution, assuming all partners with the same mass Lambda

def zSUSY(M, ast, Lambda):
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    znt =  4 * 2 * zet_f(M, ast, Lambda) # Neutralinos,  4 -> particle number, 2 -> Majorana
    zcg =  2 * 4 * zet_f(M, ast, Lambda) # Charginos,    2 -> particle number, 4 -> Dirac
    zgl =  8 * 2 * zet_f(M, ast, Lambda) # Gluinos,      8 -> particle number, 2 -> Majorana

    zH  =  4 * zet_s(M, ast, Lambda)     # Higgs bosons, not including h0 -> SM one
    
    zsl =  9 * zet_s(M, ast, Lambda)     # Sleptons,     9 -> particle number
    zsq = 12 * 3 * zet_s(M, ast, Lambda) # Squarks,     12 -> particle number, 3 -> color

    ztot = znt + zcg + zgl + zH + zsl + zsq

    return ztot

# Dark Sector with N copies of the Standard Model

def zSM_DS(M, ast, L_DS):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fp  =  gp * zet_v(M, ast, L_DS)  # Photon
    fgl = ggl * zet_v(M, ast, L_DS) # Gluon

    # Electroweak dofs
    fGB = gW * zet_v(M, ast, L_DS)  + gZ * zet_v(M, ast, L_DS)  + gH * zet_s(M, ast, L_DS)

    # Neutrino masses, assuming Normal Ordering, in GeV

    fnu = gnu * (zet_f(M, ast, L_DS) + zet_f(M, ast, L_DS) + zet_f(M, ast, L_DS)) # Active Majorana neutrinos
    
    fl  = gl * (zet_f(M, ast, L_DS) + zet_f(M, ast, L_DS) + zet_f(M, ast, L_DS))  # Charged leptons

    # QCD dofs

    fq  = gq * (zet_f(M, ast, L_DS) + zet_f(M, ast, md) + zet_f(M, ast, L_DS) +
                zet_f(M, ast, L_DS) + zet_f(M, ast, L_DS) + zet_f(M, ast, L_DS))    # Quarks
    
    return fp + fnu + fgl + fGB + fl + fq

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
    
    M   = v[0]/GeV_in_g          # PBH mass in GeV
    ast = v[1]

    FSM = fSM(M, ast)
    FDM = fDM(M, ast, mDM, sDM) # DM evaporation contribution
    FX  = fX(M, ast, mX)        # Mediator contribution
    FT  = FSM + FDM + FX        # Total Evaporation contribution

    GSM = gSM(M, ast)
    GDM = gDM(M, ast, mDM, sDM) # DM evaporation contribution
    GX  = gX(M, ast, mX)        # Mediator contribution
    GT  = GSM + GDM + GX        # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * FT/(GN**2 * M**2)
    dastdtl = - log(10.) * 10.**tl * ast * (GT - 2.*FT)/(GN**2 * M**3)

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
    
    M   = v[0]  # BH mass in GeV
    ast = v[1]

    M_GeV = M/GeV_in_g

    FSM = fSM(M, ast) + gg * phi_g(M, ast, 0.)  # Including gravitons
    FNS = Nscls * phi_s(M, ast, 0.) # N scalars evaporation contribution
    FT  = FSM + FNS                 # Total Evaporation contribution

    GSM = gSM(M, ast) + gg * gam_g(M, ast, 0.)  # Including gravitons
    GNS = Nscls * gam_s(M, ast, 0.) # N scalars evaporation contribution
    GT  = GSM + GNS                 # Total Evaporation contribution

    dMdtl   = - FT/(GN**2 * M_GeV**2)
    dastdtl = - ast * (GT - 2.*FT)/(GN**2 * M_GeV**3)

    Jac = log(10.) * 10.**tl

    return Jac*np.array([GeV_in_g * dMdtl, dastdtl])

def Itau_NSC_entropy(tl, v, Nscls): # Dark Radiation Case
    
    M   = v[0]  # BH mass in GeV
    ast = v[1]
    SBH = v[2]
    SRD = v[3]

    M_GeV = M/GeV_in_g

    FSM = fSM(M, ast) + gg * phi_g(M, ast, 0.)  # Including gravitons
    FNS = Nscls * phi_s(M, ast, 0.) # N scalars evaporation contribution
    FT  = FSM + FNS                 # Total Evaporation contribution

    GSM = gSM(M, ast) + gg * gam_g(M, ast, 0.)  # Including gravitons
    GNS = Nscls * gam_s(M, ast, 0.) # N scalars evaporation contribution
    GT  = GSM + GNS                 # Total Evaporation contribution

    ZSM = zSM(M, ast) + 2.0 * zet_g(M, ast, 0.) # SM + graviton contribution
    ZNS = Nscls * zet_s(M, ast, 0.) # N scalars evaporation contribution
    ZT  = ZSM + ZNS                 # Total Evaporation contribution

    dMdtl   = - FT/(GN**2 * M_GeV**2)
    dastdtl = - ast * (GT - 2.*FT)/(GN**2 * M_GeV**3)

    dSBHdtl   = - 2. * pi * (2.*FT + (2.*FT - ast**2 * GT)/sqrt(1. - ast**2))/(GN * M_GeV)
    dSRaddtl  =   ZT/(GN * M_GeV)

    Jac = log(10.) * 10.**tl

    return Jac*np.array([GeV_in_g * dMdtl, dastdtl, dSBHdtl, dSRaddtl])

def Itau_NSC_DR(tl, v, s, Nscls): # Dark Radiation Case
    
    M   = v[0]
    ast = v[1]

    FSM = fSM(M, ast)               # Including gravitons
    FDR = fDR(M, ast, s)            # DM evaporation contribution
    FNS = Nscls * phi_s(M, ast, 0.) # N scalars evaporation contribution
    FT  = FSM + FDR + FNS           # Total Evaporation contribution

    GSM = gSM(M, ast)               # Including gravitons
    GDR = gDR(M, ast, s)            # DM evaporation contribution
    GNS = Nscls * gam_s(M, ast, 0.) # N scalars evaporation contribution
    GT  = GSM + GDR + GNS            # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * kappa * FT * M**-2
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GT - 2.*FT)

    return [dMdtl, dastdtl]


def ItauDR_MB(tl, v, s, k): # Dark Radiation Case
    
    M   = v[0]
    ast = v[1]

    M_GeV = (M/GeV_in_g) # PBH mass in GeV

    S_BH = 2.*pi*GN*M_GeV**2*(1. + sqrt(1. - ast**2)) # PBH entropy

    FSM = fSM(M, ast)
    FDR = fDR(M, ast, s) # DM evaporation contribution
    FT  = FSM + FDR      # Total Evaporation contribution

    GSM = gSM(M, ast)
    GDR = gDR(M, ast, s) # DM evaporation contribution
    GT  = GSM + GDR      # Total Evaporation contribution

    dMdtl   = - log(10.) * 10.**tl * kappa * FT * M**-2/S_BH**k
    dastdtl = - log(10.) * 10.**tl * ast * kappa * M**-3 * (GT - 2.*FT)/S_BH**k

    return [dMdtl, dastdtl]


# Determining the scale fator where PBHs evaporate

def afin(aexp, rPBHi, rRadi, t, ail):

    a = [10.**(aexp[0])]

    ain = 10.**ail # Initial scale factor
    
    A = -ain * rPBHi * sqrt(GN * (ain * rPBHi + rRadi))
    B = a[0] * rPBHi * sqrt(GN * (a[0] * rPBHi + rRadi))
    C = 2. * rRadi * (sqrt(GN*(ain * rPBHi + rRadi)) - sqrt(GN*(a[0]*rPBHi + rRadi)))
    D = GN * sqrt(6.*pi) * rPBHi**2
    
    return [A + B + C - D*t]

#-------------------------------------------------------------------------------------------------#
#                                   g*(T) and g*S(T) interpolation                                #
#-------------------------------------------------------------------------------------------------#
import os
path, filename = os.path.split(os.path.realpath(__file__))

datg  = os.path.join(path, "data/gstar.dat")
datgS = os.path.join(path, "data/gstarS.dat")

gTab  = np.loadtxt(datg)
gSTab = np.loadtxt(datgS)

Ttab = gTab[:,0]
gtab = gTab[:,1]
tck  = interpolate.splrep(Ttab, gtab, s=0)

def gstar(T): return interpolate.splev(T, tck, der=0)

def dgstardT(T): 
    return interpolate.splev(T, tck, der = 1)

TStab = gSTab[:,0]
gstab = gSTab[:,1]
tckS  = interpolate.splrep(TStab, gstab, s=0)

def gstarS(T): return interpolate.splev(T, tckS, der = 0)

def dgstarSdT(T): return interpolate.splev(T, tckS, der = 1)


#-------------------------------------------------------------------------------------------------------------------------------------#
#
#                                             Hawking Spectra for Different Spin fields                                               #
#
#-------------------------------------------------------------------------------------------------------------------------------------#

    
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#
#             Tables --  Kerr BHs                 #
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~#

# at = os.path.join(path, "data/absxsec/sigma_K_a.dat")
# Et = os.path.join(path, "data/absxsec/sigma_K_E.dat")

at = os.path.join(path, "data/absxsec/a_array.txt")
Et = os.path.join(path, "data/absxsec/w_array.txt")

atab = np.loadtxt(at, delimiter="\t")
Etab = np.loadtxt(Et, delimiter="\t")

ss_K_dir = os.path.join(path, "data/absxsec/d2Ns_dEdt.txt")
sf_K_dir = os.path.join(path, "data/absxsec/d2Nf_dEdt.txt")
sv_K_dir = os.path.join(path, "data/absxsec/d2Nv_dEdt.txt")
sg_K_dir = os.path.join(path, "data/absxsec/d2Ng_dEdt.txt")

Kstab = np.loadtxt(ss_K_dir, delimiter=" ")
Kftab = np.loadtxt(sf_K_dir, delimiter=" ")
Kvtab = np.loadtxt(sv_K_dir, delimiter=" ")
Kgtab = np.loadtxt(sg_K_dir, delimiter=" ")

# sig_Ks = interpolate.interp2d(Etab, atab, abs(Kstab), kind='linear', bounds_error=False, fill_value=0.)#RectBivariateSpline(atab, Etab, Kstab)
# sig_Kf = interpolate.interp2d(Etab, atab, abs(Kftab), kind='linear', bounds_error=False, fill_value=0.)#RectBivariateSpline(atab, Etab, Kftab)
# sig_Kv = interpolate.interp2d(Etab, atab, abs(Kvtab), kind='linear', bounds_error=False, fill_value=0.)#RectBivariateSpline(atab, Etab, Kvtab)
# sig_Kg = interpolate.interp2d(Etab, atab, abs(Kgtab), kind='linear', bounds_error=False, fill_value=0.) #RectBivariateSpline(atab, Etab, Kgtab)

sig_Ks = interpolate.RegularGridInterpolator((Etab, atab), abs(Kstab.T), bounds_error=False, fill_value = None)#RectBivariateSpline(atab, Etab, Kstab)
sig_Kf = interpolate.RegularGridInterpolator((Etab, atab), abs(Kftab.T), bounds_error=False, fill_value = None)#RectBivariateSpline(atab, Etab, Kftab)
sig_Kv = interpolate.RegularGridInterpolator((Etab, atab), abs(Kvtab.T), bounds_error=False, fill_value = None)#RectBivariateSpline(atab, Etab, Kvtab)
sig_Kg = interpolate.RegularGridInterpolator((Etab, atab), abs(Kgtab.T), bounds_error=False, fill_value=0.) #RectBivariateSpline(atab, Etab, Kgtab)
    
#-------------------------------#
#            Scalars            #
#-------------------------------#


def d2Ns_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)

    return sig_Ks(x, ast)

#---------------------------------------#
#           Massless Fermion            #
#---------------------------------------#


def d2Nf_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)
 
    return sig_Kf(x, ast)

#---------------------------------------#
#            Massive Fermion            #
#---------------------------------------#  

x_dir = os.path.join(path, "data/absxsec/sigma_m_0.5_x.dat")
m_dir = os.path.join(path, "data/absxsec/sigma_m_0.5_m.dat")   
G05m_dir = os.path.join(path, "data/absxsec/sigma_m_0.5_s.dat")

G05mtab = np.loadtxt(G05m_dir, delimiter="\t")

mtab  = np.loadtxt(m_dir, delimiter="\t")
Emtab = np.loadtxt(x_dir, delimiter="\t")
#G05m  = interpolate.interp2d(Emtab, mtab, abs(G05mtab), kind='linear', bounds_error=False, fill_value=0.)
G05m  = interpolate.RegularGridInterpolator((Emtab, mtab), abs(G05mtab.T), bounds_error=False, fill_value=0.)

def d2Nmf_dpdt(p, MBH, ast, m): # p in GeV, MBH in g, m in GeV
    fact = (MBH/GeV_in_g)
    x    = GN * fact * p # Dimensionless momentum ->  x = G*MBH*p
    mu   = GN * fact * m # Dimensionless mass     -> mu = G*MBH*m
    GM   = GN * MBH/GeV_in_g
    TBHK = TBH(MBH, ast)

    if(m > 0. and ast == 0.):
        return 0.159154943*(G05m(x, mu)[0]*(GM*p)**2/(np.exp(np.sqrt(p*p + m*m)/TBHK) + 1.))
    else:
        return sig_Kf(x, ast)

#-------------------------------#
#            Vectors            #
#-------------------------------#

def d2Nv_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)

    return sig_Kv(x, ast)

#-------------------------------#
#            Vectors            #
#-------------------------------#

def d2Nmv_dpdt(p, MBH, ast, m): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)
 
    return sig_Kv(x, ast)

#-------------------------------#
#           Graviton            #
#-------------------------------#

def d2Ng_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)

    return sig_Kg(x, ast)[0]
