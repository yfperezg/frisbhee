###################################################################################################
#                                                                                                 #
#                       Primordial Black Hole + Dark Radiation Generation.                        #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#           Based on: arXiv:2107.00013 (P1), arXiv:2107.00016 (P2), arXiv:2207.XXXXX              #
#                                                                                                 #
###################################################################################################

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, real

from termcolor import colored

from src import bhprop as bh #Schwarzschild and Kerr BHs library

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

    return v[0] - Mst # Function to stop the solver if the BH is equal or smaller than the Planck mass

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, rRin, xilog10, spinDR):

    M     = v[0] # PBH mass
    ast   = v[1] # PBH spin
    rRad  = v[2] # Comoving Radiation energy density in GeV^4
    rPBH  = v[3] # Comoving PBH energy density
    Tp    = v[4] # Temperature
    rDRD  = v[5] # Comoving dark Radiation energy density in GeV^4
    t     = v[6] # time in GeV^-1 

    xff = (x + xilog10)

    a = 10.**xff

    #----------------#
    #   Parameters   #
    #----------------#

    M_GeV = M/bh.GeV_in_g          # PBH mass in GeV
    
    FSM = bh.fSM(M, ast)         # SM contribution
    FDR = bh.fDR(M, ast, spinDR) # DR contribution
    FT  = FSM + FDR              # Total Energy contribution

    GSM = bh.gSM(M, ast)         # SM contribution
    GDR = bh.gDR(M, ast, spinDR) # DR contribution
    GT  = GSM + GDR              # Total Angular Momentum contribution
    
    H   = np.sqrt(8 * pi * bh.GN * (rPBH * a**(-3) + rRad * a**(-4))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter Delta
    Sig = Tp * bh.dgstardT(Tp)/bh.gstar(Tp)               # Temperature parameter Sigma
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    dM_GeVdx = - FT/(bh.GN**2 * M_GeV**2)/H   
    dastdx   = - ast * (GT - 2.*FT)/(bh.GN**2 * M_GeV**3)/H

    drRaddx  = + rRad * (4*(Del - 1) - Sig)/Del - (FSM/FT) * (dM_GeVdx/M_GeV) * a * rPBH
    drPBHdx  = + (dM_GeVdx/M_GeV) * rPBH
    dTdx     = - (Tp/Del) * (1.0 + 0.25*(bh.gstar(Tp)/bh.gstarS(Tp))*(FSM/FT)*(dM_GeVdx/M_GeV)*a*rPBH/rRad)

    dtdx     = 1./H

    #-----------------------------------------#
    #          Dark Radiation Equation        #
    #-----------------------------------------#
    
    drDRDdx =  - (FDR/FT) * (dM_GeVdx/M_GeV) * a * rPBH

    ##########################################################    
    
    dEqsdx = [bh.GeV_in_g * dM_GeVdx, dastdx, drRaddx, drPBHdx, dTdx, drDRDdx, dtdx]

    return [xeq * log(10.) for xeq in dEqsdx]

#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                          Main Class                                                                 #
#-------------------------------------------------------------------------------------------------------------------------------------#

class FBEqs_Sol:

    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM Radiation + Dark Radiation. See arXiv.2207.xxxxx.
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

        # We assume an initially Radiation dominated Universe
        
        Ti     = ((45./(16.*106.75*(pi*bh.GN)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature, in GeV
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4  # Initial Radiation energy density, in GeV^4
        rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi # Initial PBH energy density, in GeV^4

        ti = (np.sqrt(45./(16.*np.pi**3.*bh.gstar(Ti)*bh.GN))*Ti**-2) # Initial time, in GeV^-1
        TBHi   = bh.TBH(Mi, asi)  # Initial BH temperature, in GeV
        
        RDRHi  = 0.0

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        xilog10 = 0.

        rRin = rRadi

        xBE    = []
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
            
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauDR(t, y, spinDR), t_span = [-80, 40.], y0 = [Mi, asi], 
                                rtol=1.e-5, atol=1.e-20, dense_output=True)
            
            if i == 0:
                Sol_t = tau_sol.sol # Solutions for obtaining <p>
                tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV
            
            if bi > 1.e-19*(1.e9/Mi):
                xf = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tau, 0.), method='lm', tol=1.e-40) # Scale factor 
                xflog10 = xf.x[0]            
            else:
                afw = np.sqrt(1. + 4.*10.**tau*np.sqrt(2.*np.pi*bh.GN*rRadi/3.))
                xflog10 = np.log10(afw)
            
            #-----------------------------------------#
            #          Before BH evaporation          #
            #-----------------------------------------#

            StopM = lambda t, x:StopMass(t, x, Mi) # Event to stop when the mass is 1% of the initial mass
            StopM.terminal  = True
            StopM.direction = -1.
            
            v0 = [Mi, asi, rRadi, rPBHi, Ti, RDRHi, ti] # Initial condition
            
            # solve ODE
            solFBE = solve_ivp(lambda t, z: FBEqs(t, z, rRin, xilog10, spinDR),
                               [0., 1.05*abs(xflog10)], v0, method="BDF", events=StopM, rtol=1.e-5, atol=1.e-10) 

            if not solFBE.success: 
                print(solFBE)
                print(colored("didn't work!",'red'))
                exit()

            if solFBE.t[-1] < 0.:
                print(solFBE)
                print(afw, tau, 1.05*xflog10)
                break

            # Concatenating solutions
            
            xBE    = np.append(xBE,   solFBE.t[:] + xilog10)

            MBHBE  = np.append(MBHBE,  solFBE.y[0,:])
            astBE  = np.append(astBE,  solFBE.y[1,:])
            RadBE  = np.append(RadBE,  solFBE.y[2,:])
            PBHBE  = np.append(PBHBE,  solFBE.y[3,:])
            TBE    = np.append(TBE,    solFBE.y[4,:])
            RDRHBE = np.append(RDRHBE, solFBE.y[5,:])
            tmBE   = np.append(tmBE,   solFBE.y[6,:])

            Mi    = solFBE.y[0,-1]
            asi   = solFBE.y[1,-1]
            rRadi = solFBE.y[2,-1]
            rPBHi = solFBE.y[3,-1]
            Ti    = solFBE.y[4,-1]
            RDRHi = solFBE.y[5,-1]
            ti    = solFBE.y[6,-1]
            
            xilog10 += solFBE.t[-1]

            i += 1

            if i > 100:
                xflog10 = xilog10
                print("I'm stuck!", Mi, bi)
                print()
                break

        else:
            xflog10 = xilog10 # We update the value of log(a) at which PBHs evaporate

        return [xBE, tmBE, MBHBE, astBE, RadBE, PBHBE, TBE, RDRHBE]

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
