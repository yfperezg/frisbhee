###################################################################################################
#                                                                                                 #
#                       Primordial Black Hole + Dark Radiation Generation.                        #
#                                     Only DM from evaporation                                    #
#                      Considering Mass and spin Distributions f_BH(M, a*)                        #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                                    Based on: arXiv:2212.XXXXX                                   #
#                                                                                                 #
###################################################################################################

import numpy as np
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, dblquad, quad_vec, ode, solve_ivp, fixed_quad
from scipy.optimize import root
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline
from numpy.polynomial import Polynomial
from pathos.multiprocessing import ProcessingPool as Pool
from termcolor import colored
from tqdm import tqdm
import time

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, array

import BHProp as bh #Schwarzschild and Kerr BHs library

from Integrator import Simp2D # Our 2D integrator

import warnings
warnings.filterwarnings('ignore')

# --------------------------------------------------- Main Parameters ---------------------------------------------------- #
#
#          - 'Mi'   : Primordial BH initial peak Mass in grams                                                             #
#
#          - 'ai'   : Primordial BH initial peak angular momentum a*                                                       # 
#
#          - 'bi'   : Primordial BH initial fraction beta^prime                                                            #
#
#          - 'typ'  : Type of Mass & Spin distribution                                                                     #
#
#          - 'pars' : Parameters for the distribution                                                                      #
#
#          - 'mDM'  : Log10@ Dark Matter Mass                                                                              #
#
#          - 'sDM'  : Dark Matter Spin                                                                                     #
#
#------------------------------------------------------------------------------------------------------------------------- #

#-------------------------------------   Credits  ------------------------------------#
#
#      If using this code, please cite:                                               #
#
#      - arXiv:2107.00013,  arXiv:2107.00016, arXiv:2207.09462. arXiv:2212.XXXXX      #
#
#-------------------------------------------------------------------------------------#

#-------------------------------------------------#
#     Normalized Log-Normal Mass Distribution     #
#-------------------------------------------------#

def fBH_M(M, Mc, typ, pars):# M, Mc in grams

    if typ == 0: # Log-normal

        sig = pars

        Mc *= exp(sig**2)
    
        f = (1/(sqrt(2.*pi)*sig*M))*exp(-0.5*log(M/Mc)**2/sig**2)

    elif typ == 1: # Power-Law

        sig, alpha = pars
    
        Mf = Mc * 10**sig
        
        if alpha!=1:
            C  = (1.-alpha)/(Mf**(1.-alpha) - Mc**(1.-alpha))
        else:
            C = 1./np.log(Mf/Mc)
            
        if M>=Mc and M<=Mf:
            f = C*M**(-alpha)
        else:
            f = 0.

    elif typ == 2: # Critical Collapse
    
        f = (1./0.350877) * (M**1.85/Mc**2.85) * exp(-(M/Mc)**2.85)

    elif typ == 3: # Metric Preheating
        
        Mmin = Mc*10**(-5)
        
        Mmax = Mc*10**2
        
        if M>=Mmin and M<=Mmax:
            
            f = (0.648634433548238*np.exp(-(M/Mc)**1.2 - (1.6306469705871256*1.e-15*Mc**4)/M**4)*(M/Mc)**0.57)/M
        else:
            f = 0 

    return f

#------------------------------------#
#     Gaussian Spin Distribution     #
#------------------------------------#

def fBH_a(ast, astc, typ, pars):

    if typ == 0: # Gaussian
        
        sig = pars
        
        f_spin = (1/(sqrt(2.*pi)*sig))*exp(-0.5*(ast - astc)*(ast - astc)/(sig*sig))

    elif typ == 1: # Spin distribution from 1703.06869
    
        if ast < 0.4 or ast > 0.93:
            f_spin = 0.
        else:
            p = Polynomial([6.819874438769596e6,-1.3250915137798008e8,1.1253448102561085e9,-5.40302380879242e9,1.5655630692764347e10,
                           -2.6082116498402466e10,1.7147720219376637e10,1.6077104908527758e10, -2.87802820869501e10,
                           -1.2345548106792042e10,3.489883665861251e10,1.958516352643353e10,-3.690130173608378e10,
                           -3.701421846868658e10,2.856022877782092e10,5.8993706890593414e10,-4.485720546779111e9,-7.516833160228568e10,
                           -3.2376668012703716e10,7.996148940057748e10,7.127467221058759e10,-8.985027718706477e10,
                           -9.508870119860562e10,1.7675143388536224e11,-9.404366690370149e10,1.7634214053762215e10])
            f_spin = p(ast)
    
    return f_spin

#------------------------------------#
#     Mass and Spin Distributions    #
#------------------------------------#
    
def fBH_M_a(M, ast, Mc, astc, typ, pars):

    pars_MD, pars_SD = pars 

    typ_MD, typ_SD   = typ
    
    f_M = fBH_M(M, Mc, typ_MD, pars_MD)
    
    f_a = fBH_a(ast, astc, typ_SD, pars_SD)
    
    return f_M*f_a

def Int_rPBH(ast, mu, Mc, astc, typ, pars):# M in grams
    
    M = 10.**mu
    
    return M * fBH_M_a(M, ast, Mc, astc, typ, pars) * M * log(10.)

#---------------------------------------------------------------------------#
#   dM/dt, da*/dt including full grebody factors, including a DM fermion    #
#---------------------------------------------------------------------------#

def eps(M, ast, mDM, sDM):

    FSM = bh.fSM(M, ast)           # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                # Total Energy contribution
    
    return FT

def gam(M, ast, mDM, sDM): 
    
    GSM = bh.gSM(M, ast)           # SM contribution
    GDM = bh.gDM(M, ast, mDM, sDM) # DM contribution
    GT  = GSM + GDM                # Total Energy contribution
    
    return GT # SM contribution


def dMdt(M, ast, mDM, sDM):

    FSM = bh.fSM(M, ast)           # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                # Total Energy contribution
    
    return -bh.kappa * FT/(M*M)


def dastdt(M, ast, mDM, sDM):

    FSM = bh.fSM(M, ast)           # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                # Total Energy contribution
    
    GSM = bh.gSM(M, ast)           # SM contribution
    GDM = bh.gDM(M, ast, mDM, sDM) # DM contribution
    GT  = GSM + GDM                # Total Energy contribution
    
    return - ast * bh.kappa * (GT - 2.*FT)/(M*M*M)

#---------------------------------------------------------------------#
#     Solving the PBH evolution from initial mass to Planck mass      #
#---------------------------------------------------------------------#

def PBH_time_ev(Mi, asi, mDM, sDM):
    
    tBE    = []
    MBHBE  = []
    astBE  = []
    
    taut = -80.
    
    def PlanckMass(t, v, Mi):

        eps = 0.01

        if (eps*Mi > bh.MPL): Mst = eps*Mi
        else: Mst = bh.MPL
    
        return v[0] - Mst 

    while Mi >= 2.* bh.MPL:

        MPL = lambda t, x:PlanckMass(t, x, Mi)
        MPL.terminal  = True
        MPL.direction = -1.
            
        tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM, sDM), t_span = [-80., 40.], y0 = [Mi, asi], 
                            events=MPL, rtol=1.e-10, atol=1.e-15)

        tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV
    
        tBE    = append(tBE,    log10(10.**tau_sol.t[:] + 10.**taut))
        MBHBE  = append(MBHBE,  tau_sol.y[0,:])
        astBE  = append(astBE,  tau_sol.y[1,:])
    
        Mi   = tau_sol.y[0,-1]  
        asi  = tau_sol.y[1,-1]    
        taut = log10(10.**tau_sol.t[-1] + 10.**taut)
        
    return [tBE, MBHBE, astBE, taut]

#----------------------#
#     PBH lifetime     #
#----------------------#

def tau(Mi, asi, mDM, sDM):
    
    taut = -80.
    
    def PlanckMass(t, v, Mi):

        eps = 0.01

        if (eps*Mi > bh.MPL): Mst = eps*Mi
        else: Mst = bh.MPL
    
        return v[0] - Mst 
    
    MPL = lambda t, x:PlanckMass(t, x, Mi)
    MPL.terminal  = True
    MPL.direction = -1.
    
    tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM, sDM), t_span = [-80., 40.], y0 = [Mi, asi], 
                        events=MPL, rtol=1.e-5, atol=1.e-15)
    
    taut = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV 
        
    return 10.**taut

#-------------------------------#
#     Mass evolution on time    #
#-------------------------------#

def fun_aM_K(M_in, a_in, t, func):

    ftau_red, tsol_max, fM_max, fa_max = func
    
    if a_in < 0. or a_in > 1.:
        
        return [bh.MPL, 0.]
        
    else:
    
        tau_Mi = 10.**ftau_red(a_in)*M_in**3
    
        if t <= tau_Mi:
    
            x = tsol_max(a_in)
    
            t_new = (1-x)*(t/tau_Mi) + x
    
            M_r = fM_max(x)

            Mt = fM_max(t_new)*M_in/M_r

            a = fa_max(t_new)
        
        else:
            Mt = bh.MPL
            a  = 0.
        
        return [Mt, a]

#----------------------------------------------------------#
#        Integrand for PBH energy density evolution        #
#----------------------------------------------------------#

def Int_rPBH_L(mu_in, a_in, Mc, astc, typ, pars, mDM, sDM, t, ftau_red, tsol_max, fM_max, fa_max):

    func = [ftau_red, tsol_max, fM_max, fa_max]
    
    M_in = 10.**mu_in

    Mt, at = fun_aM_K(M_in, a_in, t, func)
    
    FBH = fBH_M_a(M_in, a_in, Mc, astc, typ, pars)
    
    if Mt > bh.MPL:
    
        Int = Mt * FBH
    else:
        Int = 0.
    
    return Int * log(10.) * (M_in/bh.GeV_in_g)

#------------------------------------------------------#
#        Integrand for SM radiation source term        #
#------------------------------------------------------#

def Int_drRaddt_L(mu_in, a_in, Mc, astc, typ, pars, mDM, sDM, t, ftau_red, tsol_max, fM_max, fa_max):

    func = [ftau_red, tsol_max, fM_max, fa_max]
    
    M_in = 10.**mu_in

    Mt, at = fun_aM_K(M_in, a_in, t, func)
    
    FBH = fBH_M_a(M_in, a_in, Mc, astc, typ, pars)
    
    if Mt > bh.MPL:
        
        FSM = bh.fSM(Mt, at)      # SM contribution
        FDM = bh.fDM(Mt, at, mDM, sDM) # DM contribution
        FT  = FSM + FDM            # Total Energy contribution
    
        Int = - (FSM/FT) * dMdt(Mt, at, mDM, sDM) *  FBH 
    else:
        Int = 0.

    return Int * log(10.) * (M_in/bh.GeV_in_g)


#----------------------------------------------------#
#       Integrand for DM particle production Eq      #
#----------------------------------------------------#

def Int_Gamma_L(mu_in, a_in, Mc, astc, typ, pars, mDM, sDM, t, ftau_red, tsol_max, fM_max, fa_max):

    func = [ftau_red, tsol_max, fM_max, fa_max]
    
    M_in = 10.**mu_in

    Mt, at = fun_aM_K(M_in, a_in, t, func)
    
    FBH = fBH_M_a(M_in, a_in, Mc, astc, typ, pars)
    
    if Mt > bh.MPL:
        Int = bh.Gamma_DM(Mt, at, mDM, sDM) * FBH
    else:
        Int = 0.
    
    return Int * log(10.) * (M_in/bh.GeV_in_g)

def Int_L(mu_in, a_in, pars):
    
    Mc, astc, typ, pars, mDM, sDM, t, ftau_red, tsol_max, fM_max, fa_max = pars

    func = [ftau_red, tsol_max, fM_max, fa_max]
    
    M_in = 10.**mu_in

    Mt, at = fun_aM_K(M_in, a_in, t, func)
    
    if Mt > bh.MPL:
    
        FBH = fBH_M_a(M_in, a_in, Mc, astc, typ, pars)

        FSM = bh.fSM(Mt, at)      # SM contribution
        FDR = bh.fDM(Mt, at, mDM, sDM) # DM contribution
        FT  = FSM + FDR           # Total Energy contribution
    
        Int = np.array([Mt*FBH, -(FSM/FT)*dMdt(Mt, at, mDM, sDM)*FBH, bh.Gamma_DM(Mt, at, mDM, sDM) * FBH])
        
    else:
        Int = np.array([0., 0., 0.])
    
    return Int * log(10.) * (M_in/bh.GeV_in_g) 

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, rPBH_f, drPBHdt_f, Gam_fBH, rRadi, nPBHi, nphi):
    
    t     = v[0] # Time in GeV^-1
    rRad  = v[1] # Radiation energy density in GeV^4
    Tp    = v[2] # Temperature in GeV
    NDMH  = v[3] # PBH-induced DM number density
    
    #----------------#
    #   Parameters   #
    #----------------#
    
    H   = sqrt(8 * pi * bh.GCF * (nPBHi*rPBH_f(log10(t)) * 10.**(-3*x) + rRad * 10.**(-4*x))/3.) 
    # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#
    
    dtdx    = 1./H
    drRaddx = + (nPBHi*drPBHdt_f(log10(t)))* (10**x/H)
    dTdx    = - (Tp/Del) * (1.0 - (bh.gstarS(Tp)/bh.gstar(Tp))*(0.25*drRaddx/rRad))
    
    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#
    
    dNDMHdx = (nPBHi*Gam_fBH(log10(t)))*(bh.GeV_in_g/nphi)*(1./H) # PBH-induced contribution w/o contact
    
    ##########################################################    
    
    dEqsdx = [dtdx, drRaddx, dTdx, dNDMHdx]
    
    return [xeq * log(10.) for xeq in dEqsdx]

#----------------------------------#
#    Equations after evaporation   #
#----------------------------------#

def FBEqs_aBE(x, v):

    t    = v[0] # Time in GeV^-1
    rRad = v[1] # Radiation energy density
    Tp   = v[2] # Temperature
    NDMH = v[3] # Thermal DM number density w/o PBH contribution
    
    #----------------#
    #   Parameters   #
    #----------------#

    H   = sqrt(8 * pi * bh.GCF * (rRad * 10.**(-4*x))/3.)    # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp))          # Temperature parameter
    
    #----------------------------------------#
    #    Radiation + Temperature equations   #
    #----------------------------------------#
    
    dtdx    = 1./H
    drRADdx = 0.
    dTdx    = - Tp/Del
        
    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#

    dNDMHdx = 0.                              # PBH-induced contribution w/o contact
        
    dEqsdx = [dtdx, drRADdx, dTdx, dNDMHdx]

    return [xeq * log(10.) for xeq in dEqsdx]


#-------------------------------------------------------------------------------------------------------------------------------------#
#                                              Solving Friedmann-Boltzmann Equations                                                  #
#-------------------------------------------------------------------------------------------------------------------------------------#

class FBEqs_Sol:
    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM radiation + Dark Matter. See arXiv.2212.XXXXX
    We consider mass & spin distributions.
    This class returns the full evolution of the PBH, SM and DR comoving energy densities,
    together with the evolution of the PBH mass and spin as function of the log_10 @ scale factor.
    '''
    
    def __init__(self, MPBHi, aPBHi, bPBHi, typ, pars, mDM, sDM):

        self.MPBHi = MPBHi # Mass distribution scale (central mass)
        self.aPBHi = aPBHi # Spin parameter central valu
        self.bPBHi = bPBHi # Log10[beta']
        self.typ   = typ   # Array for type of mass and spin distribution
        self.pars  = pars  # Array with parameters for distribution
        self.mDM   = mDM
        self.sDM   = sDM
    
   #----------------------------------------------------------------------------------------------------------------------------------#
   #                                                       Main function                                                              #
   #----------------------------------------------------------------------------------------------------------------------------------#
    
    def Solt(self):
        
        Mi  = 10**(self.MPBHi) # Mass distribution scale (central mass)
        asi = self.aPBHi       # Spin parameter central value
        bi  = 10**(self.bPBHi) # Initial PBH fraction

        assert 0. <= asi and asi < 1., colored('initial spin factor a* is not in the range [0., 1.)', 'red')
        assert bi < np.sqrt(bh.gamma), colored('initial PBH density is larger than the total Universe\'s budget', 'red')

        # We assume a Radiation dominated Universe as initial conditions
        
        Ti     = ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4                                          # Initial radiation energy density
        rPBHi  = bi/(sqrt(bh.gamma) -  bi)*rRadi                                              # Initial PBH energy density
        nphi   = (2.*zeta(3)/pi**2)*Ti**3                                                     # Initial photon number density
        ti = (sqrt(45./(16.*pi**3.*bh.gstar(Ti)*bh.GCF))*Ti**-2)                              # Initial time
        
        NDMHi  = 0.0       # Initial DM comoving number density, in GeV^3 
        mDM = 10**self.mDM # DM mass in GeV
        sDM = self.sDM     # DM spin     
        
        typ_MD, typ_SD   = self.typ   # Mass and Spin distributions class
        pars_MD, pars_SD = self.pars  # Mass and Spin distribution parameters

        mDM = 10**self.mDM    # DM mass in GeV

        print(colored("Mi = 10^{} g, a*i = {}, mDM = 10^{} GeV, sDM = {}".format(self.MPBHi, self.aPBHi, self.mDM, sDM), 'green'))

        DisM_types = {0:"Log-Normal", 1:"Power Law", 2:"Critical Collapse", 3:"Matter preheating"}
        DisS_types = {0:"Gaussian", 1:"From Mergers"}
        
        print(colored("Mass Distribution type = {}, pars = {}".format(DisM_types[typ_MD], pars_MD),'cyan'))
        print(colored("Spin Distribution type = {}, pars = {}\n".format(DisS_types[typ_SD], pars_SD),'cyan'))

        # Integration Limits

        if typ_MD == 0:
            assert np.isscalar(pars_MD), colored("Log-normal distribution requires 1 parameter, sigma_M", 'red')
            sig_M = pars_MD
            Min = max([2.*bh.MPL, Mi/(exp(sig_M))**4])  # Minimal mass for integration
            Mfn = Mi*(exp(sig_M))**4                    # Maximal mass for integration

        elif typ_MD == 1:
            assert len(pars_MD) == 2, colored("Power-law distribution requires 2 parameters, sigma_M, alpha", 'red')
            sig_M, alpha = pars_MD
            Min = Mi            # Minimal mass for integration
            Mfn = Mi*10.**sig_M # Maximal mass for integration

        elif typ_MD == 2:
            Min = max([2.*bh.MPL, 0.01*Mi])  # Minimal mass for integration
            Mfn = 5.*Mi                      # Maximal mass for integration

        if typ_SD == 0:   # Gaussian
            assert np.isscalar(pars_SD), colored("Gaussian distribution requires 1 parameter, sigma_a*", 'red')
            sig_a = pars_SD
            ain = max([1.e-9, asi-4.*sig_a])
            afn = min([asi+4.*sig_a, 0.99999])
        elif typ_SD == 1: # Merger
            ain = 0.4
            afn = 0.94
        
        Int_i = dblquad(Int_rPBH, log10(Min), log10(Mfn), lambda x: 0., lambda x: 1., args=(Mi, asi, self.typ, self.pars))

        nPBH_i = rPBHi/(Int_i[0]/bh.GeV_in_g) ## Initial PBH number density, adjusted to give rPBHi defined above
        
        #******************************************************************#
        #       Solving mass evolution given a particle physics model      # 
        #******************************************************************#
        
        print(colored("Solving mass evolution given a particle physics model...", 'blue'))

        start = time.time()

        Mbi=1.e14
        asbi=0.9999999

        tBE, MBHBE, astBE, taut = PBH_time_ev(Mbi, asbi, mDM, sDM)
         
        # Interpolating the results from the solver to get a function of M(t)/Min in general
        
        fM_max = interp1d(10.**tBE/10.**taut, MBHBE/Mbi)
        fa_max = interp1d(10.**tBE/10.**taut, astBE/asbi)
        tsol_max = interp1d(astBE/asbi, 10.**tBE/10.**taut) # interpolation of y = -log(a*) vs x=t/tau

        end = time.time()

        print(colored(f"Time is {end - start} s\n", 'magenta'))

        #*************************************************************************#
        #       Finding the BH lifetimes for a fixed mass and varying the a*      # 
        #*************************************************************************#

        print(colored("Finding the BH lifetimes for a fixed mass and varying the a*...", 'blue'))

        start = time.time()
        
        nts=50

        an = 0.
        ax = 1.
        da = (ax - an)/nts
        
        def log_tau_a(i): return log10(tau(1., an + da*i, mDM, sDM))
        
        a_ar = [an + da*i for i in range(nts+1)]
        
        with Pool(12) as pool:        
            tau_PBH_red = pool.map(log_tau_a, [i for i in range(nts+1)])

        ftau_red = interp1d(a_ar, tau_PBH_red)
            
        end = time.time()
            
        print(colored(f"Time is {end - start} s\n", 'magenta'))
       
        #******************************************************************#
        #              Integrating and interpolating the PBH mass          #
        #               terms dependent on the mass distribution           # 
        #******************************************************************#

        # Time range for performing the integration

        tin = log10(0.5*ti)
        tmd = log10(10.**ftau_red(0.01*afn)*(0.01*Min)**3)
        tpk = log10(10.**ftau_red(afn)*(Min)**3)
        tfn = log10(10.**ftau_red(ain)*(Mfn)**3)
        tlt = log10(10.**ftau_red(0.)*(5.*Mfn)**3)
        
        Nf1 = 15
        Nf2 = 52
        Nf3 = 170
        Nf4 = 16
        
        dt1  = (tmd-tin)/(Nf1-1)
        dt2  = (tpk-tmd)/(Nf2-1)
        dt3  = (tfn-tpk)/(Nf3-1)
        dt4  = (tlt-tfn)/(Nf4-1)
        tlow = [tin + i*dt1 for i in range(Nf1)]
        tmed = [tmd + i*dt2 for i in range(1, Nf2)]
        thig = [tpk + i*dt3 for i in range(1, Nf3)]
        tlat = [tfn + i*dt4 for i in range(1, Nf4)]
        temp = np.sort(concatenate((tlow, tmed, thig, tlat)))  # Final time array
            
        ttot=temp.shape[0]

        inTot = zeros((ttot, 3))
        iGamm = zeros(ttot)
        inRad = zeros(ttot)
        iPBHt = zeros(ttot)

        start = time.time()

        print(colored("Integrating and interpolating terms dependent on the mass & spin distribution...", 'blue'))
        
        # def integ_all(i): 
            
        #     iPBHt = dblquad(Int_rPBH_L, ain, afn, lambda x: log10(Min), lambda x: log10(Mfn),
        #                     args=(Mi, asi, self.typ, self.pars, mDM, sDM, 10.**temp[i], ftau_red, tsol_max, fM_max, fa_max),
        #                     epsabs=1.e-1, epsrel=1.e-1)[0]
            
        #     inRad = dblquad(Int_drRaddt_L, ain, afn, lambda x: log10(Min), lambda x: log10(Mfn),
        #                     args=(Mi, asi, self.typ, self.pars, mDM, sDM, 10.**temp[i], ftau_red, tsol_max, fM_max, fa_max),
        #                     epsabs=1.e-1, epsrel=1.e-1)[0]
            
        #     iGamm = dblquad(Int_Gamma_L, ain, afn, lambda x: log10(Min), lambda x: log10(Mfn),
        #                     args=(Mi, asi, self.typ, self.pars, mDM, sDM, 10.**temp[i], ftau_red, tsol_max, fM_max, fa_max),
        #                     epsabs=1.e-1, epsrel=1.e-1)[0]
            
        #     return np.array([iPBHt, inRad, iGamm])
        
        
        # with Pool(8) as pool: inTot = np.array(pool.map(integ_all, [i for i in range(len(temp))]))

        Nx=50 # Divisions for integration in M
        Ny=50 # Divisions for integration in a*
        
        for i in tqdm(range(ttot)):

            inTot[i] = Simp2D(Int_L, [Mi, asi, self.typ, self.pars, mDM, sDM, 10.**temp[i], ftau_red, tsol_max, fM_max, fa_max], 
                              [log10(Min), log10(Mfn), ain, afn, Nx, Ny]).integral()      

        
        iPBHt = inTot[:,0]
        
        inRad = inTot[:,1]
        
        iGamm = inTot[:,2]
            
        Gam_fBH = interp1d(temp, abs(iGamm), kind='linear')
        rPBH_fBH = interp1d(temp, abs(iPBHt), kind='linear')
        drRaddt_fBH = interp1d(temp, abs(inRad), kind='linear')

        end = time.time()

        print(colored(f"Time is {end - start} s\n", 'magenta'))
    
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        print(colored("Solving the Friedmann-Boltzmann equations...", 'blue'))

        start = time.time()
        
        #---------------------------------------------------------------#
        #         Computing scale factor in which BHs evaporate         #
        #---------------------------------------------------------------#

        tfn = log10(10.**ftau_red(0.)*Mfn**3)

        if bi > 1.e-19*(1.e9/Mi):
            xf = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tfn, 0.), method='lm', tol=1.e-50) # Scale factor 
            xflog10 = xf.x[0]           
        else:
            xfw = sqrt(1. + 4.*10.**tfn*sqrt(2.*pi*bh.GCF*rRadi/3.))
            xflog10 = log10(xfw)
        
        #-----------------------------------------#
        #          Before BH evaporation          #
        #-----------------------------------------#
        
        v0 = [ti, rRadi, Ti, 0.]

        # solve ODE
        solFBE = solve_ivp(lambda t, z: FBEqs(t, z, rPBH_fBH, drRaddt_fBH, Gam_fBH, rRadi, nPBH_i, nphi), 
                           [0., xflog10], v0, 
                           Method='BDF', rtol=1.e-7, atol=1.e-10)

        if not solFBE.success: 
            print(solFBE.message)
            print(xflog10, solFBE.t[-1])

        #-----------------------------------------#
        #           After BH evaporation          #
        #-----------------------------------------#
        
        Tfin = 1.e-3 # Final plasma temp in GeV

        xflog10 = solFBE.t[-1]
        
        xzmax = xflog10 + log10(cbrt(bh.gstarS(solFBE.y[2,-1])/bh.gstarS(Tfin))*(solFBE.y[2,-1]/Tfin))
        xfmax = max([xflog10, xzmax])
        
        v0aBE = [solFBE.y[0,-1], solFBE.y[1,-1], solFBE.y[2,-1], solFBE.y[3,-1]]
        
        # solve ODE        
        solFBE_aBE = solve_ivp(lambda t, z: FBEqs_aBE(t, z), [xflog10, xfmax], v0aBE, method='Radau')

        npaf = solFBE_aBE.t.shape[0]

        end = time.time()

        print(colored(f"Time is {end - start} s\n", 'magenta'))

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #       Joining the solutions before and after evaporation       #
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        x    = concatenate((solFBE.t[:], solFBE_aBE.t[:]), axis=None)
        
        t    = concatenate((solFBE.y[0,:], solFBE_aBE.y[0,:]), axis=None) 
        Rad  = concatenate((solFBE.y[1,:], solFBE_aBE.y[1,:]), axis=None)    
        PBH  = concatenate((nPBH_i*rPBH_fBH(log10(solFBE.y[0,:])), zeros(npaf)),  axis=None)
        TUn  = concatenate((solFBE.y[2,:], solFBE_aBE.y[2,:]), axis=None)
        NDBE = concatenate((solFBE.y[3,:], solFBE_aBE.y[3,:]), axis=None)
        
        return [x, t, Rad, PBH, TUn, NDBE]

    #------------------------------------------------------------#
    #                                                            #
    #                     Conversion to Oh^2                     #
    #                                                            #
    #------------------------------------------------------------#

    def Omega_h2(self):
        '''
        This function directly returns Omega_h2, using the solution above
        '''

        x, t, Rad, PBH, TUn, NDBE = self.Solt()
        
        nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3             # Initial photon number density
        
        rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3
        
        T0 = 2.34865e-13  # Temperature today in GeV
        
        Oh2  = NDBE[-1] * nphi * 10.**(-3.*x[-1]) * 10.**self.mDM * (bh.gstarS(T0)/bh.gstarS(TUn[-1]))*(T0/TUn[-1])**3*(1/rc)

        return Oh2
