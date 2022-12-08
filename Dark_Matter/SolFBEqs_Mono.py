###################################################################################################
#                                                                                                 #
#                         Primordial Black Hole + Dark Matter Generation.                         #
#                                    Only DM from evaporation                                     #
#                                                                                                 #
#         Authors: Andrew Cheek, Lucien Heurtier, Yuber F. Perez-Gonzalez, Jessica Turner         #
#                    Based on: arXiv:2107.00013 (P1) and  arXiv:2107.00016 (P2)                   #
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

from math import sqrt, log, exp, log10, pi, atan

import BHProp as bh #Schwarzschild and Kerr BHs library

from collections import OrderedDict
olderr = np.seterr(all='ignore')

# -------------------- Main Parameters ---------------------------
#
#
#          - 'Mi'   : Primordial BH initial Mass in grams
#
#          - 'ai'   : Primordial BH initial angular momentum a*
#
#          - 'bi'   : Primordial BH initial fraction beta^prim
#
#          - 'mDM'  : DM Mass in GeV
#
#          - 'sDM'  : DM spin -> [0.0, 0.5, 1.0, 2.0]
#
#          - 'g_DM' : DM degrees of freedom
#
#-----------------------------------------------------------------

#--------------------------   Credits  -----------------------------#
#
#      If using this code, please cite:
#
#      - arXiv:2107.00013,  arXiv:2107.00016                        #
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

def FBEqs(x, v, nphi, mDM, sDM, Mi, xilog10):

    M    = v[0] # PBH mass
    ast  = v[1] # PBH ang mom
    rRAD = v[2] # Radiation energy density
    rPBH = v[3] # PBH energy density
    Tp   = v[4] # Temperature
    NDMH = v[5] # PBH-induced DM number density
    t    = v[6] # time in GeV^-1

    xff = (x + xilog10)

    #----------------#
    #   Parameters   #
    #----------------#

    M_GeV = M/bh.GeV_in_g     # PBH mass in GeV
    
    FSM = bh.fSM(M, ast)           # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                # Total Energy contribution

    GSM = bh.gSM(M, ast)           # SM contribution
    GDM = bh.gDM(M, ast, mDM, sDM) # DM contribution
    GT  = GSM + GDM                # Total Angular Momentum contribution
    
    H   = np.sqrt(8 * pi * bh.GCF * (rPBH * 10.**(-3*xff) + rRAD * 10.**(-4*xff))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    dMdx    = - bh.kappa * FT/(M*M)/H
    dastdx  = - ast * bh.kappa * (GT - 2.*FT)/(M*M*M)/H
    drRADdx = - (FSM/FT) * (dMdx/M) * 10.**xff * rPBH
    drPBHdx = + (dMdx/M) * rPBH
    dTdx    = - (Tp/Del) * (1.0 - (bh.gstarS(Tp)/bh.gstar(Tp))*(0.25*drRADdx/rRAD))
    
    dtdx    = 1./H 

    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#
    
    dNDMHdx = (bh.Gamma_DM(M, ast, mDM, sDM)/H)*(rPBH/(M/bh.GeV_in_g))/nphi # PBH-induced contribution w/o contact
    
    ##########################################################    
    
    dEqsdx = [dMdx, dastdx, drRADdx, drPBHdx, dTdx, dNDMHdx, dtdx]

    return [xeq * log(10.) for xeq in dEqsdx]

#----------------------------------#
#    Equations after evaporation   #
#----------------------------------#

def FBEqs_aBE(x, v):

    t    = v[0] # Time in GeV^-1
    rRAD = v[1] # Radiation energy density
    Tp   = v[2] # Temperature
    NDMH = v[3] # Thermal DM number density w/o PBH contribution
    
    #----------------#
    #   Parameters   #
    #----------------#

    H   = sqrt(8 * pi * bh.GCF * (rRAD * 10.**(-4*x))/3.)    # Hubble parameter
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

#------------------------------------------------------------------------------------------------------------------#
#                                            Input parameters                                                      #
#------------------------------------------------------------------------------------------------------------------#
class FBEqs_Sol:
    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM radiation + Dark Matter. See arXiv:2107.00013 2107.0001
    Monochromatic mass and spin scenario
    This class returns the full evolution of the PBH, SM and DR comoving energy densities,
    together with the evolution of the PBH mass and spin as function of the log_10 @ scale factor.
    '''
    
    def __init__(self, MPBHi, aPBHi, bPBHi, mDM, sDM):

        self.MPBHi  = MPBHi # Log10[M/1g]
        self.aPBHi  = aPBHi # a_star
        self.bPBHi  = bPBHi # Log10[beta']
        self.mDM    = mDM
        self.sDM    = sDM
    
#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                       Input parameters                                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#
    
    def Solt(self):

        # Main parameters
        
        Mi     = 10**(self.MPBHi) # PBH initial Mass in grams
        asi    = self.aPBHi       # PBH initial rotation a_star factor
        bi     = 10**(self.bPBHi) # Initial PBH fraction

        # We assume a Radiation dominated Universe as initial conditions
        
        Ti     = ((45./(16.*106.75*(pi*bh.GCF)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4                                          # Initial radiation energy density
        rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi                                         # Initial PBH energy density
        nphi   = (2.*zeta(3)/pi**2)*Ti**3                                                     # Initial photon number density
        ti = (np.sqrt(45./(16.*np.pi**3.*bh.gstar(Ti)*bh.GCF))*Ti**-2)                        # Initial time
        
        NDMHi  = 0.0         # Initial DM comoving number density, in GeV^3     
        mDM  = 10**self.mDM # DM mass in GeV
        sDM  = self.sDM     # DM spin

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        xilog10 = 0.

        Min  = Mi
        asin = asi

        xBE    = []
        MBHBE  = []
        astBE  = []
        RadBE  = []
        PBHBE  = []
        TBE    = []
        NDMHBE = []
        tmBE   = []

        MBHFn  = []
        NDMHFn = []
        
        taur = []

        i = 0
        
        while Mi >= 2. * bh.MPL:# Loop on the solver such that BH mass reaches M_Planck

            #--------------------------------------------------------------------------------#
            #         Computing PBH lifetime and scale factor in which BHs evaporate         #
            #--------------------------------------------------------------------------------#
            
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauFO(t, y, mDM, sDM), t_span = [-80, 40.], y0 = [Mi, asi], 
                                 rtol=1.e-5, atol=1.e-20, dense_output=True)
            
            if i == 0:
                Sol_t = tau_sol.sol # Solutions for obtaining <p>
                tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV
            
            if bi > 1.e-19*(1.e9/Mi):
                xf = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tau, 0.), method='lm', tol=1.e-40) # Scale factor 
                xflog10 = xf.x[0]            
            else:
                xfw = np.sqrt(1. + 4.*10.**tau*np.sqrt(2.*np.pi*bh.GCF*rRadi/3.))
                xflog10 = np.log10(xfw)
            
            #-----------------------------------------#
            #          Before BH evaporation          #
            #-----------------------------------------#

            StopM = lambda t, x:StopMass(t, x, Mi) # Event to stop when the mass is 1% of the initial mass
            StopM.terminal  = True
            StopM.direction = -1.
            
            v0 = [Mi, asi, rRadi, rPBHi, Ti, NDMHi, ti]

            if self.MPBHi >= 8.:
                if self.bPBHi > -15.:
                    atol=1.e-5
                    meth='BDF'
                else:
                    atol=1.e-2
                    meth='Radau'
            else:
                atol=1.e-15
                meth='BDF'
            
            # solve ODE
            solFBE = solve_ivp(lambda t, z: FBEqs(t, z, nphi, mDM, sDM, Mi, xilog10),
                               [0., 1.05*abs(xflog10)], v0, method=meth, events=StopM, rtol=1.e-7, atol=atol) 

            if solFBE.t[-1] < 0.:
                print(solFBE)
                print(afw, tau, 1.05*xflog10)
                break

            # Concatenating solutions
            
            xBE    = np.append(xBE,    solFBE.t[:] + xilog10)
            MBHBE  = np.append(MBHBE,  solFBE.y[0,:])
            astBE  = np.append(astBE,  solFBE.y[1,:])
            RadBE  = np.append(RadBE,  solFBE.y[2,:])
            PBHBE  = np.append(PBHBE,  solFBE.y[3,:])
            TBE    = np.append(TBE,    solFBE.y[4,:])
            NDMHBE = np.append(NDMHBE, solFBE.y[5,:])
            tmBE   = np.append(tmBE,   solFBE.y[6,:])

            # Updating values of initial parameters
            
            Mi    = solFBE.y[0,-1]
            asi   = solFBE.y[1,-1]
            rRadi = solFBE.y[2,-1]
            rPBHi = solFBE.y[3,-1]
            Ti    = solFBE.y[4,-1]
            NDMHi = solFBE.y[5,-1]
            ti    = solFBE.y[6,-1]
            
            xilog10 += solFBE.t[-1]

            i += 1

            if i > 100:
                xflog10 = xilog10
                print("I'm stuck!", Mi, bi)
                print()
                break

        else:
            xflog10 = xilog10# We update the value of log(a) at which PBHs evaporate

        Tev = TBE[-1]
        
        #-----------------------------------------#
        #           After BH evaporation          #
        #-----------------------------------------#
        
        Tfin = 1.e-3 # Final plasma temp in GeV
        
        xzmax = xflog10 + np.log10(np.cbrt(bh.gstarS(TBE[-1])/bh.gstarS(Tfin))*(TBE[-1]/Tfin))
        xfmax = max(xflog10, xzmax)

        v0aBE = [tmBE[-1], RadBE[-1], TBE[-1], NDMHBE[-1]]
        
        # solve ODE        
        solFBE_aBE = solve_ivp(lambda t, z: FBEqs_aBE(t, z), [xflog10, xfmax], v0aBE, method='BDF', rtol=1.e-5, atol=1.e-10)

        npaf = solFBE_aBE.t.shape[0]

        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #       Joining the solutions before and after evaporation       #
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        x    = np.concatenate((xBE, solFBE_aBE.t[:]), axis=None)
 
        MBH  = np.concatenate((MBHBE,  np.zeros(npaf)), axis=None)
        ast  = np.concatenate((astBE,  np.zeros(npaf)), axis=None)
        t    = np.concatenate((tmBE,   solFBE_aBE.y[0,:]), axis=None) 
        Rad  = np.concatenate((RadBE,  solFBE_aBE.y[1,:]), axis=None)    
        PBH  = np.concatenate((PBHBE,  np.zeros(npaf)),  axis=None)
        T    = np.concatenate((TBE,    solFBE_aBE.y[2,:]), axis=None)
        NDMH = np.concatenate((NDMHBE, solFBE_aBE.y[3,:]), axis=None)
                
        return [x, t, MBH, ast, Rad, PBH, T, NDMH, Tev]

    #------------------------------------------------------------#
    #                                                            #
    #                     Conversion to Oh^2                     #
    #                                                            #
    #------------------------------------------------------------#
    
    def Omega_h2(self):
        '''
        This function directly returns Omega_h2, using the solution above
        '''

        x, t, MBH, ast, Rad, PBH, TUn, NDMH, Tev = self.Solt()
        
        nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3             # Initial photon number density
        
        rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3
        
        T0 = 2.34865e-13  # Temperature today in GeV
        
        Oh2  = NDMH[-1] * nphi * 10.**(-3.*x[-1]) * 10.**self.mDM * (bh.gstarS(T0)/bh.gstarS(TUn[-1]))*(T0/TUn[-1])**3*(1/rc)

        return Oh2
        
        

        
