###################################################################################################
#                                                                                                 #
#                       Primordial Black Hole + Dark Matter Generation.                           #
#                             Considering Mass Distributions f_BH(M)                              #
#                                     Only DM from evaporation                                    #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                                    Based on: arXiv:2212.XXXXX                                   #
#                                                                                                 #
###################################################################################################

import ulysses
import numpy as np
from odeintw import odeintw
import pandas as pd
from scipy import interpolate, optimize
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, quad_vec, ode, solve_ivp, fixed_quad
from scipy.optimize import root
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline
from pathos.multiprocessing import ProcessingPool as Pool
from termcolor import colored
from tqdm import tqdm

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete

import BHProp as bh # Schwarzschild and Kerr BHs library

from Integrator import Simp1D # Our 1D integrator

from collections import OrderedDict
olderr = seterr(all='ignore')

import time

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
#          - 'typ'  : Type of Mass  distribution                                                                           #
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
#                 Mass Distributions              #
#-------------------------------------------------#

def fBH_M(M, Mc, typ, pars):# M in grams

    if typ == 0: # Log-Normal

        sig = pars

        Mc *= exp(sig**2)
    
        f = (1/(sqrt(2.*pi)*sig*M))*exp(-0.5*log(M/Mc)**2/sig**2)

    elif typ == 1: # Power Law

        sig, alpha = pars
    
        Mf = Mc * 10**sig
        
        if alpha!=1:
            C  = (1.-alpha)/(Mf**(1.-alpha) - Mc**(1.-alpha))
        else:
            C = 1./np.log(Mf/Mc)
            
        if M >= 0.999*Mc and M <= 1.001*Mf:
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

def Int_rPBH(mu, Mc, typ, pars): 
    
    M = 10.**mu
    
    return M * fBH_M(M, Mc, typ, pars) * log(10.) * M

#-------------------------------------------------------#
#   dM/dt including full grebody factors, for the SM    #
#-------------------------------------------------------#

def eps(M, ast, mDM, sDM):

    FSM = bh.fSM(M, ast)           # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                # Total Energy contribution
    
    return FT

def dMdt(M, ast, mDM, sDM):

    FSM = bh.fSM(M, ast)      # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM           # Total Energy contribution
    
    return -bh.kappa * FT/(M*M)

#---------------------------------------------------------------------#
#     Solving the PBH evolution from initial mass to Planck mass      #
#---------------------------------------------------------------------#

def PBH_time_ev(Mi, asi, mDM, sDM):
    
    tBE    = []
    MBHBE  = []
    astBE  = []
    
    taut = -80.

    def PlanckMass_A(t, v, Mi):

        eps = 1.e-2

        if (eps*Mi > bh.MPL): Mst = eps*Mi
        else: Mst = bh.MPL
    
        return v[0] - Mst # Function to stop the solver if the BH is equal or smaller than the Planck mass

    while Mi >= 1.5 * bh.MPL:

        MPL_A = lambda t, x:PlanckMass_A(t, x, Mi)
        MPL_A.terminal  = True
        MPL_A.direction = -1.
            
        tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM, sDM), t_span = [-80., 40.], y0 = [Mi, asi], 
                            events=MPL_A, rtol=1.e-10, atol=1.e-15)

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
        
    tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM, sDM), t_span = [-80., 40.], y0 = [Mi, asi], 
                        rtol=1.e-5, atol=1.e-15)
         
    taut = tau_sol.t[-1]
        
    return 10.**taut

#-------------------------------#
#     Mass evolution on time    #
#-------------------------------#

def fun_M(M_in, t, ftau, fM):
    
    tau_Mi = 10.**ftau(log10(M_in))
    
    if t <= tau_Mi:
        Mt = fM(t/tau_Mi)*M_in
    else:
        Mt = bh.MPL
        
    return Mt

#-----------------------------------#
#    Initial Mass for given M(t)    #
#-----------------------------------#

def Mi_fun(M, t, ftau, fM):
    
    Mi_r = lambda x, Mt, t: Mt - fun_M(x, t, ftau, fM)

    root = optimize.toms748(Mi_r, 2.*bh.MPL, 1e14, args=(M, t))
    
    return root

#-------------------------------------------#
#    Mass distribution at a given time t    #
#-------------------------------------------#

def fBH_M_t(Mt, pars):
    
    Mi, asi, typ, pars_MD, ftau, fM, mDM, sDM, t = pars
    
    Min = Mi_fun(Mt, t, ftau, fM)
    
    FBH = (eps(Min, asi, mDM, sDM)/eps(Mt, asi, mDM, sDM)) * (Mt/Min)**2 * fBH_M(Min, Mi, typ, pars_MD)
    
    return FBH

#--------------------------------------------------------------------------#
#       Integrand for PBH, radiation densities and DM production rate      #
#--------------------------------------------------------------------------#

def Int_L(mu, pars):
    
    Mi, ast, typ, pars_MD, ftau, fM, mDM, sDM, t = pars
    
    Mt = 10.**mu
    
    FSM = bh.fSM(Mt, ast)           # SM contribution
    FDM = bh.fDM(Mt, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                 # Total Energy contribution
    
    fBH_t = fBH_M_t(Mt, pars)
    
    #in [GeV^2, GeV^2, GeV]
    Int = np.array([- dMdt(Mt, ast, mDM, sDM)/bh.GeV_in_g, - (FSM/FT) * dMdt(Mt, ast, mDM, sDM)/bh.GeV_in_g,
                    bh.Gamma_DM(Mt, ast, mDM, sDM)])
    
    return Int * fBH_t * log(10.) * Mt

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, rPBHi, rRadi, nPBHi, nphi, fR_fBH, fP_fBH, Gam_fBH):
    
    t     = v[0] # Time in GeV^-1
    rRad  = v[1] # Radiation energy density in GeV^4
    rPBH  = v[2] # PBH energy density in GeV^4
    Tp    = v[3] # Temperature in GeV
    NDMH  = v[4] # PBH-induced DM number density
    
    #----------------#
    #   Parameters   #
    #----------------#
    
    H   = sqrt(8 * pi * bh.GCF * (rPBH * 10.**(-3*x) + rRad * 10.**(-4*x))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    dtdx    = 1./H
    drRaddx = + (nPBHi*fR_fBH(log10(t))) * (10**x/H)
    drPBHdx = - (nPBHi*fP_fBH(log10(t))) * (1./H)
    dTdx    = - (Tp/Del) * (1.0 - (bh.gstarS(Tp)/bh.gstar(Tp))*(0.25*drRaddx/rRad))
    
    if rPBH < 0.:
        drRaddx *= 0.
        drPBHdx *= 0.
        
    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#
    
    dNDMHdx = (nPBHi*Gam_fBH(log10(t))/nphi)*(1./H) # PBH-induced contribution w/o contact
    
    ##########################################################    
    
    dEqsdx = [dtdx, drRaddx, drPBHdx, dTdx, dNDMHdx]
    
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

    return [x * log(10.) for x in dEqsdx]

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                              Solving Friedmann-Boltzmann Equations                                                  #
#-------------------------------------------------------------------------------------------------------------------------------------#

class FBEqs_Sol:
    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM radiation + Dark Matter. See arXiv.2212.XXXXX
    We consider mass distributions. Code valid for Schwarzschild PBHs only.
    This class returns the full evolution of the PBH, SM and DR comoving energy densities,
    together with the evolution of the PBH mass and spin as function of the log_10 @ scale factor.
    '''

    def __init__(self, MPBHi, bPBHi, typ, pars, mDM, sDM):

        self.MPBHi = MPBHi # Log10[M/1g]
        self.bPBHi = bPBHi # Log10[beta']
        self.typ   = typ   # Mass distribution class
        self.pars  = pars  # Mass distribution parameters
        self.mDM   = mDM   # Log10[DM Mass/GeV]
        self.sDM   = sDM   # Dark Matter spin
    
   #----------------------------------------------------------------------------------------------------------------------------------#
   #                                                       Main function                                                              #
   #----------------------------------------------------------------------------------------------------------------------------------#
    
    def Solt(self):
        
        Mi     = 10**(self.MPBHi) # Horizon mass in g at formation  --> Taken here as a parameter
        asi    = 0.               # PBH initial rotation a_star factor
        bi     = 10**(self.bPBHi) # Initial PBH fraction

        assert asi == 0., colored('Code valid only for Schwarzschild BHs.', 'red')
        assert bi < np.sqrt(bh.gamma), colored('initial PBH density is larger than the total Universe\'s budget', 'red')

        # We assume a Radiation dominated Universe as initial conditions
        
        Ti     = ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4                                          # Initial radiation energy density
        rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi                                         # Initial PBH energy density
        nphi   = (2.*zeta(3)/pi**2)*Ti**3                                                     # Initial photon number density
        ti     = (sqrt(45./(16.*pi**3.*bh.gstar(Ti)*bh.GCF))*Ti**-2)                          # Initial time
        
        typ     = self.typ   # Mass distribution class
        pars_MD = self.pars  # Mass distribution parameters

        NDMHi  = 0.0         # Initial DM comoving number density, in GeV^3
        mDM = 10**self.mDM   # DM mass in GeV
        sDM = self.sDM       # DM spin

        print(colored("Mi = 10^{} g, mDM = 10^{} GeV".format(self.MPBHi, self.mDM), 'green'))

        Dis_types = {0:"Log-Normal", 1:"Power Law", 2:"Critical Collapse", 3:"AV Model"}
        
        print(colored("Distribution type = {}, pars = {} \n".format(Dis_types[typ], pars_MD),'cyan'))

        # Integration Limits

        if typ == 0:
            assert np.isscalar(pars_MD), colored("Log-normal distribution requires 1 parameter, sigma_M", 'red')
            sig = pars_MD
            Min = max([2.*bh.MPL, Mi/(exp(sig))**4])  # Minimal mass for integration
            Mfn = Mi*(exp(sig))**4                    # Maximal mass for integration

        elif typ == 1:
            assert len(pars_MD) == 2, colored("Power-law distribution requires 2 parameters, sigma_M, alpha", 'red')
            sig, alpha = pars_MD
            Min = Mi           # Minimal mass for integration
            Mfn = Mi*10.**sig  # Maximal mass for integration

        elif typ == 2:
            Min = max([2.*bh.MPL, 0.01*Mi])  # Minimal mass for integration
            Mfn = 5.*Mi                      # Maximal mass for integration

        elif typ == 3:
            Min = max([2.*bh.MPL, Mi*1e-5])
            Mfn = Mi*1e2

        
        Int_i = quad(Int_rPBH, log10(Min), log10(Mfn), args=(Mi, typ, pars_MD))

        nPBH_i = rPBHi/(Int_i[0]/bh.GeV_in_g) ## Initial PBH number density, adjusted to give rPBHi defined above
        
        #******************************************************************#
        #       Solving mass evolution given a particle physics model      # 
        #******************************************************************#
        
        print(colored("Solving mass evolution given a particle physics model...", 'blue'))

        Mb = 1.e14 # Test Mass

        start = time.time()

        t_solg, M_solg, ast_solg, tau_solg = PBH_time_ev(Mb, asi, mDM, sDM)

        # Interpolating the results from the solver to get a function of M(t)/Min in general
        
        fM = interp1d(10.**t_solg/10.**tau_solg, M_solg/Mb)

        end = time.time()

        print(colored(f"Time is {end - start} s\n", 'magenta'))

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                PBH lifetime Interpolation                 #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        print(colored("Computing BH lifetimes and interpolating...", 'blue'))

        start = time.time()
        
        Mit1 = linspace(log10(2.*bh.MPL), 14., num = 50, endpoint=True)
        
        def log_tau(M): return log10(tau(10.**M, asi, mDM, sDM))
        
        with Pool(8) as pool: tau_PBH = pool.map(log_tau, Mit1)

        ftau = interpolate.interp1d(Mit1, tau_PBH)
        
        end = time.time()

        print(colored(f"Time is {end - start} s\n", 'magenta'))

        
        #******************************************************************#
        #              Integrating and interpolating the PBH mass          #
        #               terms dependent on the mass distribution           # 
        #******************************************************************#

        # Number of steps
        
        Nf1 = 25
        Nf2 = 151
        
        Nx  = 100 # Number of divisions for integration

        # Time range for performing the integration
        
        tin = log10(0.1*ti)
        tmd = log10(tau(max([10.*bh.MPL, 0.1*Min]), asi, mDM, sDM))
        tfn = log10(tau(5.*Mfn, asi, mDM, sDM))
        
        dt1=(tmd-tin)/(Nf1-1)
        dt2=(tfn-tmd)/(Nf2-1)
        tlow=[tin + i*dt1 for i in range(Nf1)]
        thig=[tmd + i*dt2 for i in range(1, Nf2)]
        temp = np.sort(concatenate((tlow, thig))) # Final time array
            
        ttot=temp.shape[0]

        inTot = zeros((ttot, 3))
        inRad = zeros(ttot)
        inPBH = zeros(ttot)
        iGamm = zeros(ttot)

        start = time.time()

        print(colored("Integrating and interpolating terms dependent on the mass distribution...", 'blue'))

        for i in tqdm(range(ttot)):

            Min_t = fun_M(Min, 10.**temp[i], ftau, fM)

            pars_I = [Mi, asi, typ, pars_MD, ftau, fM, mDM, sDM, 10.**temp[i]]

            inTot[i] = Simp1D(Int_L, pars_I, [log10(Min_t), log10(Mfn), Nx]).integral()
    
            inPBH[i] = inTot[i,0]
    
            inRad[i] = inTot[i,1]
    
            iGamm[i] = inTot[i,2]
            
        fR_fBH  = interp1d(temp, inRad)
        fP_fBH  = interp1d(temp, inPBH)
        Gam_fBH = interp1d(temp, iGamm)

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

        tfn = ftau(log10(Mfn))

        if bi > 1.e-19*(1.e9/Mi):
            xf = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tfn, 0.), method='lm', tol=1.e-50) # Scale factor 
            xflog10 = xf.x[0]           
        else:
            xfw = sqrt(1. + 4.*10.**tfn*sqrt(2.*pi*bh.GCF*rRadi/3.))
            xflog10 = log10(xfw)
            
        #-----------------------------------------#
        #          Before BH evaporation          #
        #-----------------------------------------#

        v0 = [ti, rRadi, rPBHi, Ti, 0.]
        
        # solve ODE
        solFBE = solve_ivp(lambda t, z: FBEqs(t, z, rPBHi, rRadi, nPBH_i, nphi, fR_fBH, fP_fBH, Gam_fBH),
                           [0., xflog10], v0, Method='BDF', rtol=1.e-7, atol=1.e-10)

        #if not solFBE.success: 
            #print(solFBE.message)
            #print(xflog10, solFBE.t[-1])

        #-----------------------------------------#
        #           After BH evaporation          #
        #-----------------------------------------#
        
        Tfin = 1.e-3 # Final plasma temp in GeV

        xflog10 = solFBE.t[-1]
        
        xzmax = xflog10 + log10(cbrt(bh.gstarS(solFBE.y[3,-1])/bh.gstarS(Tfin))*(solFBE.y[3,-1]/Tfin))
        xfmax = max([xflog10, xzmax])

        v0aBE = [solFBE.y[0,-1], solFBE.y[1,-1], solFBE.y[3,-1], solFBE.y[4,-1]]
        
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
        PBH  = concatenate((solFBE.y[2,:], zeros(npaf)),  axis=None)
        TUn  = concatenate((solFBE.y[3,:], solFBE_aBE.y[2,:]), axis=None)
        NDBE = concatenate((solFBE.y[4,:], solFBE_aBE.y[3,:]), axis=None)

        Tev = solFBE.y[3,-1]
                
        return [x, t, Rad, PBH, TUn, NDBE, Tev]

    #------------------------------------------------------------#
    #                                                            #
    #                     Conversion to Oh^2                     #
    #                                                            #
    #------------------------------------------------------------#
    
    def Omega_h2(self):
        '''
        This function directly returns Omega_h2, using the solution above
        '''

        x, t, Rad, PBH, TUn, NDMH, Tev = self.Solt()
        
        nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3             # Initial photon number density
        
        rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3
        
        T0 = 2.34865e-13  # Temperature today in GeV
        
        Oh2  = NDMH[-1] * nphi * 10.**(-3.*x[-1]) * 10.**self.mDM * (bh.gstarS(T0)/bh.gstarS(TUn[-1]))*(T0/TUn[-1])**3*(1/rc)

        return Oh2
