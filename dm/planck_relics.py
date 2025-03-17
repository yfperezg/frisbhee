##################################################################################
#                                                                                #
#                Primordial Black Hole + Dark Matter Generation.                 #
#                           Only DM from evaporation                             #
#                                                                                #
##################################################################################

import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline

from math import sqrt, log, exp, log10, pi, atan

from src import bhprop as bh #Schwarzschild and Kerr BHs library

from collections import OrderedDict
olderr = np.seterr(all='ignore')

def StopMass(t, v, Mi):
    
    eps = 0.01
        
    if (eps*Mi > bh.MPL): Mst = eps*Mi
    else: Mst = bh.MPL

    return v[0] - Mst # Function to stop the solver if the BH is equal or smaller than the Planck mass


#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, Mi, xilog10):

    M     = v[0] # PBH mass
    ast   = v[1] # PBH ang mom
    rRad  = v[2] # Radiation energy density
    rPBH  = v[3] # PBH energy density
    Tp    = v[4] # Temperature
    NPR   = v[5] # PBH number density -> To compute Planck Relic density
    t     = v[6] # time in GeV^-1

    xff = (x + xilog10)

    a = 10.**xff #Scale factor

    #----------------#
    #   Parameters   #
    #----------------#

    M_GeV = M/bh.GeV_in_g     # PBH mass in GeV
    
    FSM = bh.fSM(M, ast)           # SM contribution
    GSM = bh.gSM(M, ast)           # SM contribution
    
    H   = np.sqrt(8 * pi * bh.GN * (rPBH * a**(-3) + rRad * a**(-4))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    dM_GeVdx = - FSM/(bh.GN**2 * M_GeV**2)/H   
    dastdx   = - ast * (GSM - 2.*FSM)/(bh.GN**2 * M_GeV**3)/H

    drRaddx  = - (dM_GeVdx/M_GeV) * a * rPBH
    drPBHdx  = + (dM_GeVdx/M_GeV) * rPBH
    dTdx    = - (Tp/Del) * (1.0 - (bh.gstar(Tp)/bh.gstarS(Tp))*(0.25*drRaddx/rRad))
    
    dtdx    = 1./H 

    #-----------------------------------------#
    #           Planck Relic Equations         #
    #-----------------------------------------#
    
    dNPRdx = 0. # PBH-induced contribution w/o contact
    
    ##########################################################    
    
    dEqsdx = [bh.GeV_in_g * dM_GeVdx, dastdx, drRaddx, drPBHdx, dTdx, dNPRdx, dtdx]

    return [xeq * log(10.) for xeq in dEqsdx]

#------------------------------------------------------------------------------------------------------------------#
#                                            Input parameters                                                      #
#------------------------------------------------------------------------------------------------------------------#
class FBEqs_Sol:

    def __init__(self, MPBHi, aPBHi, bPBHi):

        self.MPBHi  = MPBHi # Log10[M/1g]
        self.aPBHi  = aPBHi # a_star
        self.bPBHi  = bPBHi # Log10[beta']
    
#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                       Input parameters                                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#
    
    def Solt(self):
        
        Mi     = 10**(self.MPBHi) # PBH initial Mass in grams
        asi    = self.aPBHi       # PBH initial rotation a_star factor
        bi     = 10**(self.bPBHi) # Initial PBH fraction
        Ti     = ((45./(16.*106.75*(pi*bh.GN)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4  # Initial Radiation energy density -- assuming a Radiation dominated Universe
        rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi # Initial PBH energy density
        nphi   = (2.*zeta(3)/pi**2)*Ti**3             # Initial photon number density

        #print(Mi, asi, bi, Ti, rRadi, rPBHi)
        
        NPRi = rPBHi/(Mi/bh.GeV_in_g) #Initial PBH number density

        TBHi = bh.TBH(Mi, asi)  # Initial BH temperature
        
        ti = (np.sqrt(45./(16.*np.pi**3.*bh.gstar(Ti)*bh.GN))*Ti**-2) # Initial time, assuming a Radiation dom Universe

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
        NPRBE  = []
        tmBE   = []
    
        i = 0
        
        while Mi >= 2. * bh.MPL:

            #--------------------------------------------------------------------------------#
            #         Computing PBH lifetime and scale factor in which BHs evaporate         #
            #--------------------------------------------------------------------------------#
            
            tau_sol = solve_ivp(fun=lambda t, y: bh.ItauSM(t, y), t_span = [-80, 40.], y0 = [Mi, asi], 
                                rtol=1.e-5, atol=1.e-20, dense_output=True)
            
            if i == 0:
                Sol_t = tau_sol.sol # Solutions for obtaining <p>
                tau = tau_sol.t[-1] # Log10@PBH lifetime in inverse GeV

            #print(10.**tau)
            
            if bi > 1.e-19*(1.e9/Mi):
                xf = root(bh.afin, [40.], args = (rPBHi, rRadi, 10.**tau, 0.), method='lm', tol=1.e-40) # Scale factor 
                xflog10 = xf.x[0]            
            else:
                xfw = np.sqrt(1. + 4.*10.**tau*np.sqrt(2.*np.pi*bh.GN*rRadi/3.))
                xflog10 = np.log10(xfw)
            
            #-----------------------------------------#
            #          Before BH evaporation          #
            #-----------------------------------------#

            StopM = lambda t, x:StopMass(t, x, Mi) # Event to stop when the mass is 1% of the initial mass
            StopM.terminal  = True
            StopM.direction = -1.
            
            v0 = [Mi, asi, rRadi, rPBHi, Ti, NPRi, ti]

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
        
            solFBE = solve_ivp(lambda t, z: FBEqs(t, z, Mi, xilog10),
                               [0., 1.05*abs(xflog10)], v0, method=meth, events=StopM, rtol=1.e-7, atol=atol) 

            if solFBE.t[-1] < 0.:
                print(solFBE)
                print(xfw, tau, 1.05*xflog10)
                break

            xBE   = np.append(xBE,    solFBE.t[:] + xilog10)
            MBHBE = np.append(MBHBE,  solFBE.y[0,:])
            astBE = np.append(astBE,  solFBE.y[1,:])
            RadBE = np.append(RadBE,  solFBE.y[2,:])
            PBHBE = np.append(PBHBE,  solFBE.y[3,:])
            TBE   = np.append(TBE,    solFBE.y[4,:])
            NPRBE = np.append(NPRBE,  solFBE.y[5,:])
            tmBE  = np.append(tmBE,   solFBE.y[6,:])
            
            Mi    = solFBE.y[0,-1]
            asi   = solFBE.y[1,-1]
            rRadi = solFBE.y[2,-1]
            rPBHi = solFBE.y[3,-1]
            Ti    = solFBE.y[4,-1]
            NPRi  = solFBE.y[5,-1]
            ti    = solFBE.y[6,-1]
            
            xilog10 += solFBE.t[-1]

            i += 1

            if i > 100:
                xflog10 = xilog10
                print("I'm stuck! - Mono", Mi, bi)
                print()
                break

        else:
            xflog10 = xilog10# We update the value of log(a) at which PBHs evaporate

        Tev = TBE[-1]
                
        return [xBE, tmBE, MBHBE, astBE, RadBE, PBHBE, TBE, NPRBE, Tev]
    
    #------------------------------------------------------------#
    #                                                            #
    #                     Conversion to Oh^2                     #
    #                                                            #
    #------------------------------------------------------------#
    
    def Omega_h2(self):
        '''
        This function directly returns Omega_h2, using the solution above
        '''

        x, t, MBH, ast, Rad, PBH, TUn, NPR, Tev = self.Solt()
        
        rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3
        
        T0 = 2.34865e-13  # Temperature today in GeV
        
        Oh2  = NPR[-1] * 10.**(-3.*x[-1]) * bh.mPL * (bh.gstarS(T0)/bh.gstarS(Tev))*(T0/Tev)**3*(1/rc)

        return Oh2
