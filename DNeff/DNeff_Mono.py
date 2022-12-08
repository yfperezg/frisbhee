###################################################################################################
#                                                                                                 #
#                       Primordial Black Hole + Dark Radiation Generation.                        #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#           Based on: arXiv:2107.00013 (P1), arXiv:2107.00016 (P2), arXiv:2207.XXXXX              #
#                                                                                                 #
###################################################################################################

import numpy as np
from odeintw import odeintw
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

import BHProp as bh #Schwarzschild and Kerr BHs library

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

def PlanckMass_A(t, v, Mi):

    eps = 1.e-2

    if (eps*Mi > bh.MPL): Mst = eps*Mi
    else: Mst = bh.MPL
    
    return v[0] - Mst # Function to stop the solver if the BH is equal or smaller than the Planck mass

def PlanckMass_B(t, v, Mi):

    eps = 1.e-2

    if (eps*Mi > bh.MPL): Mst = eps*Mi
    else: Mst = bh.MPL
    
    return Mi*10.**v[1] - Mst # Function to stop the solver if the BH is equal or smaller than the Planck mass

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(a, v, Mi, rRin, ailog10, spinDR):

    t     = v[0] # time in GeV^-1 
    Mt    = v[1] # Log_10 of Initial Mass
    ast   = v[2] # PBH spin
    rRAD  = v[3] # Comoving radiation energy density in GeV^4
    rPBH  = v[4] # Comoving PBH energy density
    Tp    = v[5] # Temperature
    rDRD  = v[6] # Comoving dark radiation energy density in GeV^4

    M = Mi * 10.**Mt # PBH mass

    #----------------#
    #   Parameters   #
    #----------------#
    
    FSM = bh.fSM(M, ast)         # SM contribution
    FDR = bh.fDR(M, ast, spinDR) # DR contribution
    FT  = FSM + FDR              # Total Energy contribution

    GSM = bh.gSM(M, ast)         # SM contribution
    GDR = bh.gDR(M, ast, spinDR) # DR contribution
    GT  = GSM + GDR              # Total Angular Momentum contribution
    
    H   = np.sqrt(8 * pi * bh.GCF * (rPBH * 10.**(-3*(a + ailog10)) + rRAD * 10.**(-4*(a + ailog10)))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    dtda    = 1./H
    dMtda   = - bh.kappa * FT/(M*M*M)/H/log(10.)   
    dastda  = - ast * bh.kappa * (GT - 2.*FT)/(M*M*M)/H
    drRADda = - (FSM/FT) * (dMtda * log(10.)) * 10.**(a + ailog10) * rPBH
    drPBHda = + (dMtda * log(10.)) * rPBH
    dTda    = - (Tp/Del) * (1.0 - (bh.gstarS(Tp)/bh.gstar(Tp))*(0.25*drRADda/(rRAD + rRin*rDRD)))

    #-----------------------------------------#
    #          Dark Radiation Equation        #
    #-----------------------------------------#
    
    drDRDda =  - (FDR/FT) * (dMtda * log(10.)) * 10.**(a + ailog10) * (rPBH/rRin)

    ##########################################################    
    
    dEqsda = [dtda, dMtda, dastda, drRADda, drPBHda, dTda, drDRDda]

    return [x * log(10.) for x in dEqsda]

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Main Class                                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

class FBEqs_Sol:

    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM radiation + Dark Radiation. See arXiv.2207.xxxxx.
    We consider the collapse of density fluctuations as the PBH formation mechanism.
    This class returns the full evolution of the PBH, SM and DR comoving energy densities,
    together with the evolution of the PBH mass and spin as function of the log_10 @ scale factor.
    '''

    def __init__(self, MPBHi, aPBHi, bPBHi, spinDR):

        self.MPBHi  = MPBHi  # Log10[MPBH_in/1g]
        self.aPBHi  = aPBHi  # a_star_in
        self.bPBHi  = bPBHi  # Log10[beta']
        self.spinDR = spinDR # Dark Radiation spin


    #+++++++++++++++++++++++++++++++ Main Function +++++++++++++++++++++++++++++++#
    
    def Solt(self):
        
        Mi     = 10**(self.MPBHi) # PBH initial Mass in grams
        asi    = self.aPBHi       # PBH initial rotation a_star factor
        bi     = 10**(self.bPBHi) # Initial PBH fraction
        spinDR = self.spinDR      # Dark Radiation spin

        # We assume an initially radiation dominated Universe
        
        Ti     = ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature, in GeV
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4  # Initial radiation energy density, in GeV^4
        rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi # Initial PBH energy density, in GeV^4

        ti = (np.sqrt(45./(16.*np.pi**3.*bh.gstar(Ti)*bh.GCF))*Ti**-2) # Initial time, in GeV^-1
        TBHi   = bh.TBH(Mi, asi)  # Initial BH temperature, in GeV
        
        RDRHi  = 0.0

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        ailog10 = 0.

        Min  = Mi
        asin = asi
        rRin = rRadi

        aBE    = []
        MBHBE  = []
        astBE  = []
        RadBE  = []
        PBHBE  = []
        TBE    = []
        RDRHBE = []
        tmBE   = []

        i = 0
        
        while Mi >= 2. * bh.MPL: # We evolve until the PBH mass is equal to the Planck mass

            #--------------------------------------------------------------------------------#
            #         Computing PBH lifetime and scale factor in which BHs evaporate         #
            #--------------------------------------------------------------------------------#
            
            MPL_A = lambda t, x:PlanckMass_A(t, x, Mi)
            MPL_A.terminal  = True
            MPL_A.direction = -1.
            
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauDR(t, y, spinDR), t_span = [-80, 40.], y0 = [Mi, asi], 
                                 events=MPL_A, rtol=1.e-5, atol=1.e-20, dense_output=True)
            
            if i == 0:
                Sol_t = tau_sol.sol # Solutions for obtaining <p>
                tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV
            
            if bi > 1.e-19*(1.e9/Mi):
                af = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tau, 0.), method='lm', tol=1.e-40) # Scale factor 
                aflog10 = af.x[0]            
            else:
                afw = np.sqrt(1. + 4.*10.**tau*np.sqrt(2.*np.pi*bh.GCF*rRadi/3.))
                aflog10 = np.log10(afw)
            
            #-----------------------------------------#
            #          Before BH evaporation          #
            #-----------------------------------------#

            MPL_B = lambda t, x:PlanckMass_B(t, x, Mi)
            MPL_B.terminal  = True
            MPL_B.direction = -1.
            
            v0 = [ti, 0., asi, rRadi, rPBHi, Ti, RDRHi] # Initial condition
            
            # solve ODE
            solFBE = solve_ivp(lambda t, z: FBEqs(t, z, Mi, rRin, ailog10, spinDR),
                               [0., 1.05*abs(aflog10)], v0, method="BDF", events=MPL_B, rtol=1.e-5, atol=1.e-10) 

            if solFBE.t[-1] < 0.:
                print(solFBE)
                print(afw, tau, 1.05*aflog10)
                break
            
            aBE    = np.append(aBE,   solFBE.t[:] + ailog10)

            tmBE   = np.append(tmBE,   solFBE.y[0,:])
            MBHBE  = np.append(MBHBE,  10.**(solFBE.y[1,:])*Mi)
            astBE  = np.append(astBE,  solFBE.y[2,:])
            RadBE  = np.append(RadBE,  solFBE.y[3,:])
            PBHBE  = np.append(PBHBE,  solFBE.y[4,:])
            TBE    = np.append(TBE,    solFBE.y[5,:])
            RDRHBE = np.append(RDRHBE, solFBE.y[6,:])

            ti    = solFBE.y[0,-1]
            Mi    = 10.**(solFBE.y[1,-1])*Mi
            asi   = solFBE.y[2,-1]
            rRadi = solFBE.y[3,-1]
            rPBHi = solFBE.y[4,-1]
            Ti    = solFBE.y[5,-1]
            RDRHi = solFBE.y[6,-1]
            
            ailog10 += solFBE.t[-1]

            i += 1

            if i > 100:
                aflog10 = ailog10
                print("I'm stuck!", Mi, bi)
                print()
                break

        else:
            aflog10 = ailog10 # We update the value of log(a) at which PBHs evaporate

        return [aBE, tmBE, MBHBE, astBE, RadBE, PBHBE, TBE, rRin*RDRHBE]

    #------------------------------------------------------------#
    #                                                            #
    #                     Determining DNeff                      #
    #                                                            #
    #------------------------------------------------------------#
    
    def DNeff(self):
        '''
        This function directly returns DNeff, using the solution above
        '''

        x, t, MBH, ast, Rad, PBH, TUn, DRad = self.Solt()
        
        rDR_rRad = (DRad[-1]/(Rad[-1]))

        EV_EQ = (bh.gstar(TUn[-1])/bh.gstar(0.75e-9)) * (bh.gstarS(0.75e-9)/bh.gstarS(TUn[-1]))**(4./3.)
        
        DelNeff = ((8./7.)*(4./11.)**(-4./3.) + 3.045) * rDR_rRad * EV_EQ

        return DelNeff
