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
mu   = 336e-3#2.2e-3
md   = 340e-3#4.6e-3
ms   = 486e-3#95e-3
mc   = 1.275
mb   = 4.18
mt   = 173.1
mg   = 0.6   # Ficticious gluon mass ---> indicates the QCD phase transition, following PRD41(1990)3052

m_pi0 = 0.1349768   # Neutral pion mass
m_pic = 0.13957039  # Charged pion mass

# Neutrino parameters

Dm31 = 2.511e-3 # Atmospheric quadratic neutrino mass difference, in eV^2
Dm21 = 7.410e-5 # Solar quadratic neutrino mass difference, in eV^2

# Degrees of freedom of the SM ---> Before the EW phase transition

gW  = 2.*2.     # W
gZ  = 2.        # Z
gH  = 4.        # Higgs
gp  = 2.        # photon
gg  = 2.        # graviton
ggl = 8.*2.       # gluons
gl  = 2.*2.     # leptons
gq  = 2.*2.*3  # quarks
gnu = 2.        # LH neutrino

gf = 3.*gnu + 3.*gl + 6.*gq   # Total number of SM fermion dofs 
gs = gH                       # Total number of SM scalar dofs
gv = gW + gZ + gp + gg + ggl  # Total number of SM vector dofs

g_pi0 = 1 # Neutral pion
g_pic = 2 # Charged pion

# Constants

c     = 299792.458       # in km/s
gamma = sqrt(3.)**-3.    # Collapse factor
GN    = 6.70883e-39      # Gravitational constant in GeV^-2
mPL   = 1./sqrt(GN)      # Planck mass in GeV
v     = 174              # Higgs vev
csp   = 0.35443          # sphaleron conversion factor
GF    = 1.1663787e-5     # Fermi constant in GeV^-2
LQCD  = 0.2              # Lambda QCD in GeV

# Conversion factors

GeV_in_g     = 1.782661907e-24  # 1 GeV in g
Mpc_in_cm    = 3.085677581e24   # 1 Mpc in cm

cm_in_invkeV = 5.067730938543699e7       # 1 cm in keV^-1
year_in_s    = 3.168808781402895e-8      # 1 year in s
GeV_in_invs  = cm_in_invkeV * c * 1.e11  # 1 GeV in s^-1

MPL   = mPL * GeV_in_g        # Planck mass in g
kappa = mPL**4 * GeV_in_g**3  # Evaporation constant in g^3 * GeV -- from PRD41(1990)3052
mPL_red = 1./sqrt(8.*pi*GN)   # Reduced mass Planck

# BH Temperature in GeV

def TBH(M, astar):

    M_GeV = M/GeV_in_g
    
    return (1./(4.*pi*GN*M_GeV))*(sqrt(abs(1. - astar**2))/(1. + sqrt(abs(1. - astar**2)))) # M in g

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                  Momentum Integrated Rate for Kerr BHs                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#


def Gamma_S(M, ast, m):# Scalar, in GeV

    GM = GN * (M/GeV_in_g) # in GeV^-1

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

    GM = GN * (M/GeV_in_g) # in GeV^-1

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
    
    return  (27/(1024. * pi**4 * GM)) * In


def Gamma_V(M, ast, m):# Vector

    GM = GN * (M/GeV_in_g) # in GeV^-1

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
    
    return  (27/(1024. * pi**4 * GM)) * In

def Gamma_G(M, ast, m):# Spin 2

    GM = GN * (M/GeV_in_g) # in GeV^-1

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

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
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

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
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

    GM = GN * (M/GeV_in_g) # in GeV^-1
    
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

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * phi_v(M, ast, 0.)  # Photon
    fgl = ggl * phi_v(M, ast, 0.6) # Gluon
    fW  =  gW * phi_v(M, ast, mW)  # W
    fZ  =  gZ * phi_v(M, ast, mZ)  # Z
    fH  =  gH * phi_s(M, ast, mH)  # Higgs

    fnu = 3. * gnu * phi_f(M, ast, 0.) # Active neutrinos
    
    fl  = gl * (phi_f(M, ast, me) + phi_f(M, ast, mmu) + phi_f(M, ast, mtau))  # Charged leptons

    if T >= LQCD:
    
        fq  = gq * (phi_f(M, ast, mu) + phi_f(M, ast, md) + phi_f(M, ast, ms) +
                    phi_f(M, ast, mc) + phi_f(M, ast, mb) + phi_f(M, ast, mt))    # Quarks
        
    else: # Below Lambda_QCD we only include pions

        fq  = g_pi0 * phi_s(M, ast, m_pi0) + g_pi0 * phi_s(M, ast, m_pi0)
    
    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq

def fSM_test(M, ast):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * phi_v(M, ast, 0.)  # Photon
    fnu = 3. * gnu * phi_f(M, ast, 0.) # Active neutrinos

    if T >= mW:  fW =  gW * phi_v(M, ast, 0.) 
    else: fW = 0

    if T >= mZ:  fZ =  gZ * phi_v(M, ast, 0.) 
    else: fZ = 0.

    if T >= mH:  fH =  gH * phi_s(M, ast, 0.)  # Higgs
    else: fH = 0.

    if T >= 0.6: fgl = ggl * phi_v(M, ast, 0.) # Gluon
    else: fgl = 0.

    if T >= mtau: ftau = gl*phi_f(M, ast, 0.)
    else: ftau = 0.

    if T >= mmu:  fmu = gl*phi_f(M, ast, 0.)
    else: fmu = 0.

    if T >= me:   fe = gl*phi_f(M, ast, 0.)
    else: fe = 0.

    fl = fe + fmu + ftau

    if T >= LQCD:

        if T >= mt: ft = gq * phi_f(M, ast, 0.)
        else: ft = 0.

        if T >= mb: fb = gq * phi_f(M, ast, 0.)
        else: fb = 0.

        if T >= mc: fc = gq * phi_f(M, ast, 0.)
        else: fc = 0.

        if T >= ms: fs = gq * phi_f(M, ast, 0.)
        else: fs = 0.

        if T >= md: fd = gq * phi_f(M, ast, 0.)
        else: fd = 0.

        if T >= mu: fu = gq * phi_f(M, ast, 0.)
        else: fu = 0.

        fq = fu + fd + fs + fc + fb + ft

        
    else: # Below Lambda_QCD we only include pions


        if T >= m_pi0: fpi0 = g_pi0 * phi_f(M, ast, 0.)
        else: fpi0 = 0.

        if T >= m_pic: fpic = g_pic * phi_f(M, ast, 0.)
        else: fpic = 0.

        fq = fpi0 + fpic 
    
    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq

#--------------------------------------------------------------#
#              SM Contribution + massive neutrinos             #
#--------------------------------------------------------------#

def fSM_nu(M, ast, m0):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * phi_v(M, ast, 0.)  # Photon
    fgl = ggl * phi_v(M, ast, 0.6) # Gluon
    fW  =  gW * phi_v(M, ast, mW)  # W
    fZ  =  gZ * phi_v(M, ast, mZ)  # Z
    fH  =  gH * phi_s(M, ast, mH)  # Higgs

    # Neutrino masses, assuming Normal Ordering, in GeV

    m1 = m0*1.e-9 
    m2 = sqrt(Dm21 + m0*m0)*1.e-9 
    m3 = sqrt(Dm31 + m0*m0)*1.e-9 

    fnu = gnu * (phi_f(M, ast, m1) + phi_f(M, ast, m2) + phi_f(M, ast, m3)) # Active Majorana neutrinos
    
    fl  = gl * (phi_f(M, ast, me) + phi_f(M, ast, mmu) + phi_f(M, ast, mtau))  # Charged leptons
    
    if T >= LQCD:
    
        fq  = gq * (phi_f(M, ast, mu) + phi_f(M, ast, md) + phi_f(M, ast, ms) +
                    phi_f(M, ast, mc) + phi_f(M, ast, mb) + phi_f(M, ast, mt))    # Quarks
        
    else: # Below Lambda_QCD we only include pions

        fq  = g_pi0 * phi_s(M, ast, m_pi0) + g_pi0 * phi_s(M, ast, m_pi0)

    
    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq

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

    # Electroweak + QCD gauge bosons + Higgs dofs
    fv = (gW + gZ + gH + ggl + gp) * phi_v(M, ast, L_DS) 
    
    # Higgs 
    fs = gH * phi_s(M, ast, L_DS)

    # Fermions
    ff = (gnu + gl + gq) * phi_f(M, ast, L_DS) 

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

    g0 = 10.**gs_f(ast)
    
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

    GM = GN * (M/GeV_in_g) # in GeV^-1

    g12 = 10.**gf_f(ast)

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

    GM = GN * (M/GeV_in_g) # in GeV^-1

    g1 = 10.**gv_f(ast)

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

    GM = GN * (M/GeV_in_g) # in GeV^-1

    g2 = 10.**gG_f(ast)

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

    T = TBH(M, ast)

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
    
    if T >= LQCD:

        fq  = gq * (gam_f(M, ast, mu) + gam_f(M, ast, md) + gam_f(M, ast, ms) +
                    gam_f(M, ast, mc) + gam_f(M, ast, mb) + gam_f(M, ast, mt))    # Quarks
        
    else: # Below Lambda_QCD we only include pions

        fq = g_pi0 * gam_s(M, ast, m_pi0) + g_pi0 * gam_s(M, ast, m_pi0)

    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq

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
    fW  =  gW * gam_v(M, ast, mW)  # W
    fZ  =  gZ * gam_v(M, ast, mZ)  # Z
    fH  =  gH * gam_s(M, ast, mH)  # Higgs

    # Neutrino masses, assuming Normal Ordering, in GeV

    m1 = m0*1.e-9 
    m2 = sqrt(Dm21 + m0*m0)*1.e-9 
    m3 = sqrt(Dm31 + m0*m0)*1.e-9 

    fnu = gnu * (gam_f(M, ast, m1) + gam_f(M, ast, m2) + gam_f(M, ast, m3)) # Active Majorana neutrinos
    
    fl  = gl * (gam_f(M, ast, me) + gam_f(M, ast, mmu) + gam_f(M, ast, mtau))  # Charged leptons
    
    if T >= LQCD:
    
        fq  = gq * (gam_f(M, ast, mu) + gam_f(M, ast, md) + gam_f(M, ast, ms) +
                    gam_f(M, ast, mc) + gam_f(M, ast, mb) + gam_f(M, ast, mt))    # Quarks
        
    else: # Below Lambda_QCD we only include pions

        fq = g_pi0 * gam_s(M, ast, m_pi0) + g_pi0 * gam_s(M, ast, m_pi0)

    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq


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

    # Electroweak + QCD gauge bosons + Higgs dofs
    fv = (gW + gZ + gH + ggl + gp) * gam_v(M, ast, L_DS) 
    
    # Higgs 
    fs = gH * gam_s(M, ast, L_DS)

    # Fermions
    ff = (gnu + gl + gq) * gam_f(M, ast, L_DS) 

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

    GM = GN * (M/GeV_in_g) # in GeV^-1

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

    GM = GN * (M/GeV_in_g) # in GeV^-1

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

    GM = GN * (M/GeV_in_g) # in GeV^-1

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

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * zet_v(M, ast, 0.)  # Photon
    fgl = ggl * zet_v(M, ast, 0.6) # Gluon
    fW  =  gW * zet_v(M, ast, mW)  # W
    fZ  =  gZ * zet_v(M, ast, mZ)  # Z
    fH  =  gH * zet_s(M, ast, mH)  # Higgs

    fnu = 3. * gnu * zet_f(M, ast, 0.) # Active neutrinos
    
    fl  = gl * (zet_f(M, ast, me) + zet_f(M, ast, mmu) + zet_f(M, ast, mtau))  # Charged leptons
    
    if T >= LQCD:
    
        fq  = gq * (zet_f(M, ast, mu) + zet_f(M, ast, md) + zet_f(M, ast, ms) +
                    zet_f(M, ast, mc) + zet_f(M, ast, mb) + zet_f(M, ast, mt))    # Quarks
        
    else: # Below Lambda_QCD we only include pions

        fq = g_pi0 * zet_s(M, ast, m_pi0) + g_pi0 * zet_s(M, ast, m_pi0)

    
    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq

#--------------------------------------------------------------#
#              SM Contribution + massive neutrinos             #
#--------------------------------------------------------------#

def zSM_nu(M, ast, m0):

    T = TBH(M, ast)
    
    # Contribution from each particle --> We do not include the Graviton contribution here

    fgr =  0.
    fp  =  gp * zet_v(M, ast, 0.)  # Photon
    fgl = ggl * zet_v(M, ast, 0.6) # Gluon
    fW  =  gW * zet_v(M, ast, mW)  # W
    fZ  =  gZ * zet_v(M, ast, mZ)  # Z
    fH  =  gH * zet_s(M, ast, mH)  # Higgs

    # Neutrino masses, assuming Normal Ordering, in GeV

    m1 = m0*1.e-9 
    m2 = sqrt(Dm21 + m0*m0)*1.e-9 
    m3 = sqrt(Dm31 + m0*m0)*1.e-9 

    fnu = gnu * (zet_f(M, ast, m1) + zet_f(M, ast, m2) + zet_f(M, ast, m3)) # Active Majorana neutrinos
    
    fl  = gl * (zet_f(M, ast, me) + zet_f(M, ast, mmu) + zet_f(M, ast, mtau))  # Charged leptons
    
    if T >= LQCD:
    
        fq  = gq * (zet_f(M, ast, mu) + zet_f(M, ast, md) + zet_f(M, ast, ms) +
                    zet_f(M, ast, mc) + zet_f(M, ast, mb) + zet_f(M, ast, mt))    # Quarks
        
    else: # Below Lambda_QCD we only include pions

        fq = g_pi0 * zet_s(M, ast, m_pi0) + g_pi0 * zet_s(M, ast, m_pi0)

    
    return fgr + fp + fnu + fgl + fW + fZ + fH + fl + fq



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

    # Electroweak + QCD gauge bosons + Higgs dofs
    fv = (gW + gZ + gH + ggl + gp) * zet_v(M, ast, L_DS) 
    
    # Higgs 
    fs = gH * zet_s(M, ast, L_DS)

    # Fermions
    ff = (gnu + gl + gq) * zet_f(M, ast, L_DS) 

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

    dMdtl   = - log(10.) * 10.**tl * FT/(GN**2 * M_GeV**2)
    dastdtl = - log(10.) * 10.**tl * ast * (GT - 2.*FT)/(GN**2 * M_GeV**3)

    return [GeV_in_g * dMdtl, dastdtl]

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

    S_BH = 4.*pi*GN*M_GeV**2 # PBH entropy

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