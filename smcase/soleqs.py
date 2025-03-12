###################################################################################################
#                                                                                                 #
#                                  Primordial Black Hole Evaporation                              #
#                         Including determination of Entropy from BH + Radiation                  #
#                                                                                                 #
#                                    Author: Yuber F. Perez-Gonzalez                              #
#           Based on: arXiv:2107.00013 (P1), arXiv:2107.00016 (P2), arXiv:2207.XXXXX              #
#                                                                                                 #
###################################################################################################

import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, real

from src import bhprop as bh #Schwarzschild and Kerr BHs library

from termcolor import colored

# --------------------------------------------------- Main Parameters ---------------------------------------------------- #
#
#          - 'Mi'   : Primordial BH initial Mass in grams                                                                  #
#
#          - 'ai'   : Primordial BH initial angular momentum a*                                                            # 
#
#          - 'bi'   : Primordial BH initial fraction beta^prime                                                            # 
#
#          - 'spin_DR' : Dark Radiation spin                                                                               #
#
#------------------------------------------------------------------------------------------------------------------------- #

#--------------------------   Credits  -----------------------------#
#
#      If using this code, please cite:                             #
#
#      - arXiv:2107.00013,  arXiv:2107.00016, arXiv:2207.XXXXX      #
#
#-------------------------------------------------------------------#

def StopMass(t, v, Mi):
    
    eps = 0.01
        
    if (eps*Mi > bh.MPL): Mst = eps*Mi
    else: Mst = bh.MPL

    return v[0] - Mst

def SBH_Rad_eq(a, v):

    SBH   = v[2] # PBH Bekenstein-Hawking entropy
    SRad  = v[3] # PBH Radiation entropy
    
    return SBH - SRad # Function to find PBH - Radiation entropy equality

def PBH_Rad_eq(x, v):

    rRad  = v[4] # Radiation energy density
    rPBH  = v[5] # PBH energy density

    a = 10.**x

    rRad = rRad * a**(-4)
    rpbh = rPBH * a**(-3)
    
    return rRad - rpbh # Function to find PBH - Radiation energy density equality

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, xilog10):

    M     = v[0] # PBH mass
    ast   = v[1] # PBH ang mom
    SBH   = v[2] # PBH Bekenstein-Hawking entropy
    SRad  = v[3] # PBH Radiation entropy
    rRad  = v[4] # Radiation energy density
    rPBH  = v[5] # PBH energy density
    Tp    = v[6] # Temperature
    t     = v[7] # time in GeV^-1

    M_GeV = (M/bh.GeV_in_g) # PBH mass in GeV

    xff = x + xilog10

    a = 10.**xff

    #----------------#
    #   Parameters   #
    #----------------#
    
    FSM = bh.fSM(M, ast) + 2.0 * bh.phi_g(M, ast, 0.) # SM + graviton contribution
    GSM = bh.gSM(M, ast) + 2.0 * bh.gam_g(M, ast, 0.) # SM + graviton contribution
    ZSM = bh.zSM(M, ast) + 2.0 * bh.zet_g(M, ast, 0.) # SM + graviton contribution
    
    H   = np.sqrt(8 * pi * bh.GN * (rPBH * a**(-3) + rRad * a**(-4))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter

    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    dM_GeVdx = - FSM/(bh.GN**2 * M_GeV**2)/H   
    dastdx   = - ast * (GSM - 2.*FSM)/(bh.GN**2 * M_GeV**3)/H
    
    dSBHdx   = - 2. * pi * (2.*FSM + (2.*FSM - ast**2 * GSM)/sqrt(1. - ast**2))/(bh.GN * M_GeV)/H
    dSRaddx  =   ZSM/(bh.GN * M_GeV)/H
    
    drRaddx  = - (dM_GeVdx/M_GeV) * 10**xff * rPBH
    drPBHdx  = + (dM_GeVdx/M_GeV) * rPBH
    
    dTdx     = - (Tp/Del) * (1.0 - (bh.gstar(Tp)/bh.gstarS(Tp))*(drRaddx/(4.*rRad)))
    
    dtdx    = 1./H

    ##########################################################

    kappa = bh.GeV_in_g # Conversion factor to have Mass equation rate for PBH mass in g
    
    dEqsdx = [kappa * dM_GeVdx, dastdx, dSBHdx, dSRaddx, drRaddx, drPBHdx, dTdx, dtdx]

    return [xeq * log(10.) for xeq in dEqsdx]

#----------------------------------#
#    Equations after evaporation   #
#----------------------------------#

def FBEqs_aBE(x, v):

    rRad = v[0] # Radiation energy density
    Tp   = v[1] # Temperature
    t    = v[2] # Time in GeV^-1

    a = 10.**x
    
    #----------------#
    #   Parameters   #
    #----------------#

    H   = sqrt(8 * pi * bh.GN * (rRad * a**(-4))/3.)    # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp))          # Temperature parameter
    
    #----------------------------------------#
    #    Radiation + Temperature equations   #
    #----------------------------------------#

    dtdx    = 1./H
    drRaddx = 0.
    dTdx    = - Tp/Del

    ##########################################################
    
    dEqsdx = [drRaddx, dTdx, dtdx]

    return [xeq * log(10.) for xeq in dEqsdx]

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Main Class                                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

class FBEqs_Sol:

    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM Radiation. See arXiv.2207.xxxxx.
    We consider the collapse of density fluctuations as the PBH formation mechanism.
    This class returns the full evolution of the PBH, SM comoving energy densities,
    together with the evolution of the PBH mass and spin as function of the log_10 @ scale factor.

    We compute the entropy evolution from evaporation
    '''

    def __init__(self, MPBHi, aPBHi, bPBHi, et_test):

        self.MPBHi   = MPBHi  # Log10[MPBH_in/1g]
        self.aPBHi   = aPBHi  # a_star_in
        self.bPBHi   = bPBHi  # Log10[beta']
        self.et_test = et_test # Boolean for whether stopping at Page time 


    #+++++++++++++++++++++++++++++++ Main Function +++++++++++++++++++++++++++++++#
    
    def Solt(self):
        
        Mi      = 10**(self.MPBHi) # PBH initial Mass in grams
        asi     = self.aPBHi       # PBH initial rotation a_star factor
        bi      = 10**(self.bPBHi) # Initial PBH fraction
        et_test = self.et_test     # Boolean for whether stopping at Page time

        # We assume an initially Radiation dominated Universe
        
        Ti     = ((45./(16.*106.75*(pi*bh.GN)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature, in GeV
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4  # Initial Radiation energy density, in GeV^4
        rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi # Initial PBH energy density, in GeV^4

        ti   = (np.sqrt(45./(16.*np.pi**3.*bh.gstar(Ti)*bh.GN))*Ti**-2) # Initial time, in GeV^-1

        
        SRadi = 0. # Initial Radiation entropy
        SBHi  = 2.*pi*bh.GN*(Mi/bh.GeV_in_g)**2*(1. + sqrt(1. - asi**2)) # Initial Bekenstein-Hawking entropy  -- Dimensionless

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        xilog10 = 0.

        Min  = Mi

        xBE    = []
        MBHBE  = []
        astBE  = []
        SBHBE  = []
        SRDBE  = []
        RadBE  = []
        PBHBE  = []
        TBE    = []
        tmBE   = []

        t_Page = 0.
        
        Teql = 0.
        Tdec = 0.

        i = 0

        dens_out = True
        
        while Mi >= 10. * bh.MPL: # We evolve until the PBH mass is equal to the Planck mass

            #--------------------------------------------------------------------------------#
            #         Computing PBH lifetime and scale factor in which BHs evaporate         #
            #--------------------------------------------------------------------------------#
            
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauSM(t, y), t_span = [-80, 40.], y0 = [Mi, asi], 
                                rtol=1.e-5, atol=1.e-20, dense_output=True)
            
            if i == 0:
                tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV
         
            if bi > 1.e-19*(1.e9/Mi):
                xf = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tau, 0.), method='lm', tol=1.e-40) # Scale factor 
                xflog10 = xf.x[0]            
            else:
                xfw = np.sqrt(1. + 4.*10.**tau*np.sqrt(2.*np.pi*bh.GN*rRadi/3.))
                xflog10 = np.log10(xfw)
            
            #-----------------------------------------#
            #          Before BH evaporation          #
            #-----------------------------------------#

            StopM = lambda t, x:StopMass(t, x, Mi)
            StopM.terminal  = True
            StopM.direction = -1.

            SBH_Rad = lambda t, x:SBH_Rad_eq(t, x)
            SBH_Rad.terminal  = et_test
            SBH_Rad.direction = -1.

            PBH_Rad = lambda t, x:PBH_Rad_eq(t, x)
            
            v0 = [Mi, asi, SBHi, SRadi, rRadi, rPBHi, Ti, ti] # Initial condition

            if t_Page > 0.: dens_out = False
            
            # solve ODE
            solFBE = solve_ivp(lambda t, z: FBEqs(t, z, xilog10),
                               [0., 1.05*abs(xflog10)], v0, events=(StopM,PBH_Rad,SBH_Rad), dense_output=dens_out, method="BDF", rtol=1.e-7, atol=1.e-12)

            if solFBE.t[-1] < 0.:
                print(solFBE)
                print(xfw, tau, 1.05*xflog10)
                break

            if i == 0 and solFBE.y_events[1].shape[0] > 0: 
                Teql = solFBE.y_events[1][0,6]
                Tdec = solFBE.y_events[1][1,6]

            if solFBE.t_events[2].shape[0] > 0:
                x_Page = solFBE.t_events[2][0]
                t_Page = solFBE.sol(solFBE.t_events[2][0])[7]
                if not et_test:
                    print(colored("Warning : Page time passed, proceed with caution", "red"))
                    print(colored("Page time = {0:.6E} * t_ev, PBH mass at Page time = {1:.6E} * Min".format(t_Page/10.**tau, solFBE.sol(x_Page)[0]/Min),'blue'))

                if et_test: break

            xBE    = np.append(xBE,    solFBE.t[:] + xilog10)
            MBHBE  = np.append(MBHBE,  solFBE.y[0,:])
            astBE  = np.append(astBE,  solFBE.y[1,:])
            SBHBE  = np.append(SBHBE,  solFBE.y[2,:])
            SRDBE  = np.append(SRDBE,  solFBE.y[3,:])
            RadBE  = np.append(RadBE,  solFBE.y[4,:])
            PBHBE  = np.append(PBHBE,  solFBE.y[5,:])
            TBE    = np.append(TBE,    solFBE.y[6,:])
            tmBE   = np.append(tmBE,   solFBE.y[7,:])
            
            Mi    = solFBE.y[0,-1]
            asi   = solFBE.y[1,-1]
            SBHi  = solFBE.y[2,-1]
            SRadi = solFBE.y[3,-1]
            rRadi = solFBE.y[4,-1]
            rPBHi = solFBE.y[5,-1]
            Ti    = solFBE.y[6,-1]
            ti    = solFBE.y[7,-1]
            
            xilog10 += solFBE.t[-1]

            i += 1

            if i > 100:
                xflog10 = xilog10
                print("I'm stuck!", Mi, bi)
                print()
                break

        else:
            xflog10 = xilog10 # We update the value of log(a) at which PBHs evaporate

        Tev = TBE[-1]

        return [xBE, tmBE, MBHBE, astBE, SBHBE, SRDBE, RadBE, PBHBE, TBE, Teql, Tdec, Tev, x_Page, t_Page]
