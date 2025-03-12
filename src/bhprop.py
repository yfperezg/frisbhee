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

import os
path, filename = os.path.split(os.path.realpath(__file__))


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
LQCD  = 0.322            # Lambda QCD in GeV
TEW   = 159.5            # Electroweak phase transition temperature in GeV

# Conversion factors

GeV_in_g     = 1.782661907e-24  # 1 GeV in g
Mpc_in_cm    = 3.085677581e24   # 1 Mpc in cm

cm_in_invkeV = 5.067730938543699e7       # 1 cm in keV^-1
year_in_s    = 3.168808781402895e-8      # 1 year in s
GeV_in_invs  = cm_in_invkeV * c * 1.e11  # 1 GeV in s^-1

MPL   = mPL * GeV_in_g        # Planck mass in g
kappa = mPL**4 * GeV_in_g**3  # Evaporation constant in g^3 * GeV -- from PRD41(1990)3052
mPL_red = 1./sqrt(8.*pi*GN)   # Reduced mass Planck in GeV

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

#-------------------------------------------------------------------------------------------------------------#
#             Interpolating Tables with energy-integrated Hawking spectrum, f, g and s functions              #
#-------------------------------------------------------------------------------------------------------------#

ast_tab = np.loadtxt(os.path.join(path, "data/absxsec/ast_tab.txt")) # Array containing a* values
z_tab   = np.loadtxt(os.path.join(path, "data/absxsec/z_tab.txt"))   # Array containing z = GMm, m= particle's mass, values

# Scalars

psi_scl_tab   = np.loadtxt(os.path.join(path, "data/absxsec/psi_scl_z.txt"))
phi_scl_tab   = np.loadtxt(os.path.join(path, "data/absxsec/phi_scl_z.txt"))
gamma_scl_tab = np.loadtxt(os.path.join(path, "data/absxsec/gam_scl_z.txt"))
zeta_scl_tab  = np.loadtxt(os.path.join(path, "data/absxsec/zeta_scl_z.txt"))


psi_scl_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), psi_scl_tab, bounds_error=False, fill_value = None)
phi_scl_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), phi_scl_tab, bounds_error=False, fill_value = None)
gamma_scl_int = interpolate.RegularGridInterpolator((ast_tab, z_tab), gamma_scl_tab, bounds_error=False, fill_value = None)
zeta_scl_int  = interpolate.RegularGridInterpolator((ast_tab, z_tab), zeta_scl_tab, bounds_error=False, fill_value = None)

# Fermions

psi_fer_tab   = np.loadtxt(os.path.join(path, "data/absxsec/psi_fer_z.txt"))
phi_fer_tab   = np.loadtxt(os.path.join(path, "data/absxsec/phi_fer_z.txt"))
gamma_fer_tab = np.loadtxt(os.path.join(path, "data/absxsec/gam_fer_z.txt"))
zeta_fer_tab  = np.loadtxt(os.path.join(path, "data/absxsec/zeta_fer_z.txt"))


psi_fer_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), psi_fer_tab, bounds_error=False, fill_value = None)
phi_fer_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), phi_fer_tab, bounds_error=False, fill_value = None)
gamma_fer_int = interpolate.RegularGridInterpolator((ast_tab, z_tab), gamma_fer_tab, bounds_error=False, fill_value = None)
zeta_fer_int  = interpolate.RegularGridInterpolator((ast_tab, z_tab), zeta_fer_tab, bounds_error=False, fill_value = None)

# Vectors

psi_vec_tab   = np.loadtxt(os.path.join(path, "data/absxsec/psi_vec_z.txt"))
phi_vec_tab   = np.loadtxt(os.path.join(path, "data/absxsec/phi_vec_z.txt"))
gamma_vec_tab = np.loadtxt(os.path.join(path, "data/absxsec/gam_vec_z.txt"))
zeta_vec_tab  = np.loadtxt(os.path.join(path, "data/absxsec/zeta_vec_z.txt"))


phi_vec_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), phi_vec_tab, bounds_error=False, fill_value = None)
psi_vec_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), psi_vec_tab, bounds_error=False, fill_value = None)
gamma_vec_int = interpolate.RegularGridInterpolator((ast_tab, z_tab), gamma_vec_tab, bounds_error=False, fill_value = None)
zeta_vec_int  = interpolate.RegularGridInterpolator((ast_tab, z_tab), zeta_vec_tab, bounds_error=False, fill_value = None)

# Spin-2

psi_gra_tab   = np.loadtxt(os.path.join(path, "data/absxsec/psi_gra_z.txt"))
phi_gra_tab   = np.loadtxt(os.path.join(path, "data/absxsec/phi_gra_z.txt"))
gamma_gra_tab = np.loadtxt(os.path.join(path, "data/absxsec/gam_gra_z.txt"))
zeta_gra_tab  = np.loadtxt(os.path.join(path, "data/absxsec/zeta_gra_z.txt"))


psi_gra_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), psi_gra_tab, bounds_error=False, fill_value = None)
phi_gra_int   = interpolate.RegularGridInterpolator((ast_tab, z_tab), phi_gra_tab, bounds_error=False, fill_value = None)
gamma_gra_int = interpolate.RegularGridInterpolator((ast_tab, z_tab), gamma_gra_tab, bounds_error=False, fill_value = None)
zeta_gra_int  = interpolate.RegularGridInterpolator((ast_tab, z_tab), zeta_gra_tab, bounds_error=False, fill_value = None)

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                  Momentum Integrated Rate for Kerr BHs                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#

#-------------------------------------------------------------------#
#                    Integrated Hawking Spectrum                    #
#-------------------------------------------------------------------#

def hs(astar): return (0.3891202551434314 - 0.02735815283383375*astar - 0.022376552767463448*astar**2 
                     + (0.00009835559447136233*astar**2)/(-1.025 + astar)**2 - 0.48219183071498173*astar**3 + 0.12926706388940423*astar**4)

def hf(astar): return (-0.04716796508441029 + 0.0003642184365387838*astar + 0.6381107254113888*astar**2 
                     + (0.000023461795968477088*astar**2)/(-1.025 + astar)**2 - 0.16089673280158695*astar**3 - 0.11259864747002321*astar**4)
    
def hv(astar): return (-0.5636366140508233 - 0.0802110644822287*astar + 3.2953803997049373*astar**2 
                     + (0.00009510434823780855*astar**2)/(-1.025 + astar)**2 - 3.6843970709508245*astar**3 + 1.8385940661205495*astar**4)

def hg(astar): return (-1.6874671321658943 + 0.028948686832896774*astar + 8.980166961613598*astar**2 
                     + (0.00030046706616716914*astar**2)/(-1.025 + astar)**2 - 11.901307189829424*astar**3 + 6.314005790871085*astar**4)



def Gamma_S(M, ast, m):# Scalar, in GeV

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)
    
    Gs = 10.**hs(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  psi_scl_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = Gs

    else: In = 0.
    
    return  (27/(1024. * pi**4 * GM)) * In


def Gamma_F(M, ast, m):# Fermion

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    G12  = 10.**hf(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  phi_fer_int([ast, log10(z)])[0]  
        
    elif log10(z) < -3.0:
        
        In = G12

    else: In = 0.
    
    return  (27/(1024. * pi**4 * GM)) * In


def Gamma_V(M, ast, m):# Vector

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)
    
    G1 = 10.**hv(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  phi_vec_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = G1

    else: In = 0.
    
    return  (27/(1024. * pi**4 * GM)) * In

def Gamma_G(M, ast, m):# Spin 2

    GM = GN * (M/GeV_in_g) # in GeV^-1

    TKBH = TBH(M, ast)

    G2 = 10.**fg(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  phi_gra_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = G2

    else: In = 0.
    
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
#                     f = - M^2 dM/dt fitted functions              #
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

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  phi_scl_int([ast, log10(z)])[0] 
        
    elif log10(z) < -3.0:
        
        In = f0

    else: In = 0.

    return In

# Fermion

def phi_f(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f12  = 10.**ff(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  phi_fer_int([ast, log10(z)])[0] 
        
    elif  log10(z) < -3.0:
        
        In = f12

    else: In = 0.

    return In

# Vector

def phi_v(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f1 = 10.**fv(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  phi_vec_int([ast, log10(z)])[0] 
        
    elif  log10(z) < -3.0:
        
        In = f1

    else: In = 0.
    
    return In

# Tensor - spin2

def phi_g(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f2 = 10.**fg(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  phi_gra_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f2

    else: In = 0.
    
    return In

#------------------------------------------#
#              SM Contribution             #
#------------------------------------------#

def fSM(M, ast):

    T = TBH(M, ast)
    
    ''' 
    Contribution from each particle --> We do not include the Graviton contribution here.
    For BH temperatures larger than the EW phase transition, we consider massless gauge bosons with 2 dofs and 4 scalar dofs
    for below, we consider massive gauge bosons and 1 scalar dof.
    Similarly, for the QCD phase transition, we consider all quarks dofs when BH temperature is above Lambda_QCD = 322 MeV. 
    Below that temperature, we only consider the contribution of charged and neutral pions.
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

    fq_bLQCD  = g_pi0 * phi_s(M, ast, m_pi0) + g_pic * phi_f(M, ast, m_pic)   # Neutral and charged pions

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

# Dark Sector with N copies of the Standard Model + graviton with the same mass L_DS

def fSM_DS(M, ast, L_DS):

    T = TBH(M, ast)

    # Electroweak + QCD gauge bosons dofs
    fv = (gW_aEW + gZ_aEW + ggl + gp) * phi_v(M, ast, L_DS) 
    
    # Higgs 
    fs = gH_aEW * phi_s(M, ast, L_DS)

    # Fermions
    ff = gf * phi_f(M, ast, L_DS) 

    # Graviton
    fg = gg * phi_g(M, ast, L_DS)
    
    return fv + fs + ff + fg

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                Total g functions ---> related to the angular momentum rate, da_*/dt                                 #
#                                                   Counting SM dofs + Dark Radiation                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

def gs_f(astar): return (-4.051551680929044 - 0.0906840762040224*astar + 1.4494270120860604*astar**2 
                       + (0.00012404227920010614*astar**2)/(-1.025 + astar)**2 - 1.9117571961975526*astar**3 + 1.3126804298107988*astar**4)
    
def gf_f(astar): return (-3.5115113216682095 - 0.0809224871511442*astar 
                       + 0.7847225497595978*astar**2 + (0.00013240801697621499*astar**2)/(-1.025 + astar)**2 - 1.6431128012131193*astar**3 + 1.3093862882449498*astar**4)

def gv_f(astar): return (-3.620619113318074 - 0.10052289144242625*astar + 1.7817866500602548*astar**2 
                       + (0.0001490487102809037*astar**2)/(-1.025 + astar)**2 - 2.212048942765569*astar**3 + 1.5222029958417689*astar**4)

def gG_f(astar): return (-4.27332250841818 - 0.19349041845338877*astar + 6.191951514721466*astar**2 
                       + (0.0003260121463999355*astar**2)/(-1.025 + astar)**2 - 7.669134056579806*astar**3 + 4.432340856016421*astar**4)

#--------------------------------------------------------------------------------#
#              Our interpolated forms including the particle's mass              #
#--------------------------------------------------------------------------------#

# Scalar

def gam_s(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f0 = 10.**gs_f(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  gamma_scl_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f0

    else: In = 0.

    return In

# Fermion

def gam_f(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f12  = 10.**gf_f(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  gamma_fer_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f12

    else: In = 0.

    return In

# Vector

def gam_v(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f1 = 10.**gv_f(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  gamma_vec_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f1

    else: In = 0.
    
    return In

# Tensor - spin2

def gam_g(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f2 = 10.**gG_f(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  gamma_gra_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f2

    else: In = 0.
    
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

# Dark Sector with N copies of the Standard Model + graviton with the same mass L_DS

def gSM_DS(M, ast, L_DS):

    T = TBH(M, ast)

    # Electroweak + QCD gauge bosons dofs
    fv = (gW_aEW + gZ_aEW + ggl + gp) * gam_v(M, ast, L_DS) 
    
    # Higgs 
    fs = gH_aEW * gam_s(M, ast, L_DS)

    # Fermions
    ff = gf * gam_f(M, ast, L_DS) 

    # Graviton

    fg = gg * gam_g(M, ast, L_DS)
    
    return fv + fs + ff + fg

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
    
    TKBH = TBH(M, ast)

    f0 = 10.**ss(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  zeta_scl_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f0

    else: In = 0.

    return In

# Fermion

def zet_f(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f12  = 10.**sf(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  zeta_fer_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f12
    
    else:  In = 0.

    return In

# Vector

def zet_v(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f1 = 10.**sv(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  zeta_vec_int([ast, log10(z)])[0]  
        
    elif  log10(z) < -3.0:
        
        In = f1

    else:  In = 0.
    
    return In

# Tensor - spin2

def zet_g(M, ast, m):

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
    TKBH = TBH(M, ast)

    f2 = 10.**sg(ast)

    z = GM * m 

    if -3.0 < log10(z) < log10(2.5):
        
        In =  zeta_gra_int([ast, log10(z)])[0] 
        
    elif  log10(z) < -3.0:
        
        In = f2

    else:  In = 0.
    
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

# Dark Sector with N copies of the Standard Model + graviton with the same mass L_DS

def zSM_DS(M, ast, L_DS):

    T = TBH(M, ast)

    # Electroweak + QCD gauge bosons  dofs
    fv = (gW_aEW + gZ_aEW  + ggl + gp) * zet_v(M, ast, L_DS) 
    
    # Higgs 
    fs = gH_aEW * zet_s(M, ast, L_DS)

    # Fermions
    ff = gf * zet_f(M, ast, L_DS) 

    # Graviton

    fg = gg * zet_g(M, ast, L_DS)
    
    return fv + fs + ff + fg

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
sig_Kg = interpolate.RegularGridInterpolator((Etab, atab), abs(Kgtab.T), bounds_error=False, fill_value = 0.) #RectBivariateSpline(atab, Etab, Kgtab)
    
#-------------------------------#
#            Scalars            #
#-------------------------------#


def d2Ns_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)

    return sig_Ks([x, ast])[0]

#---------------------------------------#
#           Massless Fermion            #
#---------------------------------------#


def d2Nf_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)
 
    return sig_Kf([x, ast])[0]

#---------------------------------------#
#            Massive Fermion            #
#---------------------------------------#  


dat_sigf  = os.path.join(path, "data/absxsec/sigma_s_0.5.dat")
sigf_0_Tab  = np.loadtxt(dat_sigf)

etab   = sigf_0_Tab[:,0]
sf0tab = sigf_0_Tab[:,1]
sf0_int = interpolate.splrep(etab, sf0tab, s=0)

def sigma_f_0(x): return interpolate.splev(x, sf0_int, der=0) # Absorption cross section for massless fermions

def sigma_f_LE(x,mu):
    '''
        Low energy limit of absorption cross section, taken from PRD71(2005)124020
    '''

    u = sqrt(x*x/(x*x + mu*mu)) # velocity

    return (4*pi*pi*(1+u*u)*mu)/(u*u*sqrt(1-u*u)*(1 - exp(-2*pi*mu*(1+u*u)/(u*sqrt(1-u*u))))) # 

def sigma_f_HE(x,mu):
    '''
        High energy limit of absorption cross section, taken from PRD18(1978)1798, PRD71(25005)124020
    '''
    u = sqrt(x*x/(x*x + mu*mu)) # velocity

    return (pi/(2*u**4)) * (8*u**4 + 20*u*u - 1 + sqrt((1+8*u*u)**3)) 


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

    if mu > 0. and ast == 0.:

        Gfm = 0.

        if mu < 0.01:

            if 0 <= x <= 1: Gfm = pi*sigma_f_0(x)
            
            else: Gfm = 27*pi*(1. - sqrt(1./(sqrt(27)*pi))*jv(2.5, 2.*sqrt(27.)*pi*x))

        elif 0.01 <= mu <= 1.0:

            if x < 1.e-15: Gfm = sigma_f_LE(x,mu)
            
            elif 1.e-15 <= x <= 1.9: Gfm = pi*G05m([x, mu])[0]

            else: Gfm = sigma_f_HE(x,mu) * (1. - sqrt(x/(2*sqrt(27)*(2*x*x + mu*mu)))*jv(2.5-0.35*mu+2.55*mu*mu, sqrt(27.)*pi*mu*(mu/x + 2*x/mu)))

        else: Gfm = sigma_f_HE(x,mu)

        return (Gfm*(GM*p)**2/(np.exp(np.sqrt(p*p + m*m)/TBHK) + 1.))/(2*pi*pi)
    
    else:
        return sig_Kf([x, ast])[0]

    # if(m > 0. and ast == 0.):
    #     return 0.159154943*(G05m([x, mu])[0]*(GM*p)**2/(np.exp(np.sqrt(p*p + m*m)/TBHK) + 1.))
    # else:
    #     return sig_Kf([x, ast])[0]

#-------------------------------#
#            Vectors            #
#-------------------------------#

def d2Nv_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)
 
    return sig_Kv([x, ast])[0]

#-------------------------------#
#           Graviton            #
#-------------------------------#

def d2Ng_dpdt(p, MBH, ast): # p in GeV, MBH in g

    x    = GN * (MBH/GeV_in_g) * p
    TBHK = TBH(MBH, ast)

    return sig_Kg([x, ast])[0]
