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
import pandas as pd
from scipy import interpolate
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn
from scipy.interpolate import interp1d, RectBivariateSpline

from math import sqrt, log, exp, log10, pi, atan

from termcolor import colored

from src import bhprop as bh #Schwarzschild and Kerr BHs library

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

def SBH_Rad_eq(a, v):

    SBH   = v[2] # PBH Bekenstein-Hawking entropy
    SRad  = v[3] # PBH Radiation entropy
    
    return SBH - SRad # Function to find PBH - Radiation entropy equality

#----------------------------------#
#   Equations before evaporation   #
#----------------------------------#

def FBEqs(x, v, nphi, mDM, sDM, xilog10):

    M    = v[0] # PBH mass
    ast  = v[1] # PBH ang mom
    SBH  = v[2] # PBH Bekenstein-Hawking entropy
    SRad = v[3] # PBH Radiation entropy
    rRad = v[4] # Radiation energy density
    rPBH = v[5] # PBH energy density
    Tp   = v[6] # Temperature
    t    = v[7] # time in GeV^-1
    NDMH = v[8] # PBH-induced DM number density

    xff = (x + xilog10)

    a = 10.**xff

    #----------------#
    #   Parameters   #
    #----------------#

    M_GeV = M/bh.GeV_in_g          # PBH mass in GeV
    
    FSM = bh.fSM(M, ast)           # SM contribution
    FDM = bh.fDM(M, ast, mDM, sDM) # DM contribution
    FT  = FSM + FDM                # Total Energy contribution

    GSM = bh.gSM(M, ast)           # SM contribution
    GDM = bh.gDM(M, ast, mDM, sDM) # DM contribution
    GT  = GSM + GDM                # Total Angular Momentum contribution

    ZSM = bh.zSM(M, ast)           # SM contribution
    ZDM = bh.zDM(M, ast, mDM, sDM) # DM contribution
    ZT  = ZSM + ZDM                # Total Angular Momentum contribution
    
    H   = np.sqrt(8 * pi * bh.GN * (rPBH * a**(-3) + rRad * a**(-4))/3.) # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp)) # Temperature parameter
    
    #----------------------------------------------#
    #    Radiation + PBH + Temperature equations   #
    #----------------------------------------------#

    # Mass and spin evolution
    dM_GeVdx = - FT/(bh.GN**2 * M_GeV**2)/H   
    dastdx   = - ast * (GT - 2.*FT)/(bh.GN**2 * M_GeV**3)/H

    # Evolution of entropies
    dSBHdx   = - 2. * pi * (2.*FT + (2.*FT - ast**2 * GT)/sqrt(1. - ast**2))/(bh.GN * M_GeV)/H
    dSRaddx  =   ZT/(bh.GN * M_GeV)/H

    # Evolution of SM, PBH energy densities, Temperature and time
    drRaddx  = - (FSM/FT) * (dM_GeVdx/M_GeV) * a * rPBH
    drPBHdx  = + (dM_GeVdx/M_GeV) * rPBH
    dTdx     = - (Tp/Del) * (1.0 - (bh.gstar(Tp)/bh.gstarS(Tp))*(0.25*drRaddx/rRad))
    
    dtdx    = 1./H 

    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#
    
    dNDMHdx = (bh.Gamma_DM(M, ast, mDM, sDM)/H)*(rPBH/(M/bh.GeV_in_g))/nphi # PBH-induced contribution w/o contact
    
    ##########################################################    
    
    dEqsdx = [bh.GeV_in_g * dM_GeVdx, dastdx, dSBHdx, dSRaddx, drRaddx, drPBHdx, dTdx, dtdx, dNDMHdx]

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

    H   = sqrt(8 * pi * bh.GN * (rRad * 10.**(-4*x))/3.)    # Hubble parameter
    Del = 1. + Tp * bh.dgstarSdT(Tp)/(3. * bh.gstarS(Tp))          # Temperature parameter
    
    #----------------------------------------#
    #    Radiation + Temperature equations   #
    #----------------------------------------#

    dtdx    = 1./H
    drRaddx = 0.
    dTdx    = - Tp/Del
        
    #-----------------------------------------#
    #           Dark Matter Equations         #
    #-----------------------------------------#

    dNDMHdx = 0.                              # PBH-induced contribution w/o contact
        
    dEqsdx = [dtdx, drRaddx, dTdx, dNDMHdx]

    return [xeq * log(10.) for xeq in dEqsdx]

#------------------------------------------------------------------------------------------------------------------#
#                                            Input parameters                                                      #
#------------------------------------------------------------------------------------------------------------------#
class FBEqs_Sol:
    ''' 
    Friedmann - Boltzmann equation solver for Primordial Black Holes + SM Radiation + Dark Matter. See arXiv:2107.00013 2107.0001
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

    def ItauFO(self, tl, v, mDM, sDM): # Freeze Out case
    
        M   = v[0] 
        ast = v[1]

        FSM = bh.fSM(M, ast)
        FDM = bh.fDM(M, ast, mDM, sDM) # DM evaporation contribution
        FT  = FSM + FDM             # Total Evaporation contribution

        GSM = bh.gSM(M, ast)
        GDM = bh.gDM(M, ast, mDM, sDM) # DM evaporation contribution
        GT  = GSM + GDM             # Total Evaporation contribution

        M_GeV = M/bh.GeV_in_g # BH mass in GeV

        dMdtl   = - log(10.) * 10.**tl * FT * (bh.GN * M_GeV)**-2
        dastdtl = - log(10.) * 10.**tl * ast * bh.GN**-2 * M_GeV**-3 * (GT - 2.*FT)

        return [bh.GeV_in_g * dMdtl, dastdtl]
    
#-------------------------------------------------------------------------------------------------------------------------------------#
#                                                       Input parameters                                                              #
#-------------------------------------------------------------------------------------------------------------------------------------#
    
    def Solt(self):

        # Main parameters
        
        Mi     = 10**(self.MPBHi) # PBH initial Mass in grams
        asi    = self.aPBHi       # PBH initial rotation a_star factor
        bi     = 10**(self.bPBHi) # Initial PBH fraction

        # We assume a Radiation dominated Universe as initial conditions
        
        Ti     = ((45./(16.*106.75*(pi*bh.GN)**3.))**0.25) * sqrt(bh.gamma * bh.GeV_in_g/Mi) # Initial Universe temperature
        rRadi  = (pi**2./30.) * bh.gstar(Ti) * Ti**4                                          # Initial Radiation energy density
        rPBHi  = abs(bi/(sqrt(bh.gamma) -  bi))*rRadi                                         # Initial PBH energy density
        nphi   = (2.*zeta(3)/pi**2)*Ti**3                                                     # Initial photon number density
        ti = (np.sqrt(45./(16.*np.pi**3.*bh.gstar(Ti)*bh.GN))*Ti**-2)                        # Initial time
        
        NDMHi  = 0.0        # Initial DM comoving number density, in GeV^3     
        mDM  = 10**self.mDM # DM mass in GeV
        sDM  = self.sDM     # DM spin

        SRadi = 0. # Initial Radiation entropy
        SBHi  = 2.*pi*bh.GN*(Mi/bh.GeV_in_g)**2*(1. + sqrt(1. - asi**2)) # Initial Bekenstein-Hawking entropy  -- Dimensionless

        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#
        #                                           Solving the equations                                                   #
        #+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#

        xilog10 = 0.

        xBE    = []
        MBHBE  = []
        astBE  = []
        SBHBE  = []
        SRDBE  = []
        RadBE  = []
        PBHBE  = []
        TBE    = []
        NDMHBE = []
        tmBE   = []

        i  = 0

        t_Page = 0.        # Starting the Page time
        dens_out = True    # For the solver to compute the Page time
        Min    = Mi        # SAving initial mass
        
        while Mi >= 100. * bh.MPL:# Loop on the solver such that BH mass reaches 100.*M_Planck

            #--------------------------------------------------------------------------------#
            #         Computing PBH lifetime and scale factor in which BHs evaporate         #
            #--------------------------------------------------------------------------------#
            
            tau_sol = solve_ivp(fun=lambda t, y: self.ItauFO(t, y, mDM, sDM), t_span = [-80, 40.], y0 = [Mi, asi], 
                                 rtol=1.e-5, atol=1.e-20, dense_output=True)
            
            if i == 0:
                Sol_t = tau_sol.sol # Solutions for obtaining <p>
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

            StopM = lambda t, x:StopMass(t, x, Mi) # Event to stop when the mass is 1% of the initial mass
            StopM.terminal  = True
            StopM.direction = -1.

            et_test = False # Boolean for whether stopping at Page time

            SBH_Rad = lambda t, x:SBH_Rad_eq(t, x) # Event function to determine the Page time
            SBH_Rad.terminal  = et_test
            SBH_Rad.direction = -1.
            
            v0 = [Mi, asi, SBHi, SRadi, rRadi, rPBHi, Ti, ti, NDMHi]

            if self.MPBHi >= 7.:
                if self.bPBHi > -15.:
                    atol=1.e-5
                    meth='BDF'
                else:
                    atol=1.e-7
                    meth='BDF'
            else:
                atol=1.e-9
                meth='BDF'

            if t_Page > 0.: dens_out = False # Once the Page time is saved, we don't compute a continous solution 
            
            # solve ODE
            solFBE = solve_ivp(lambda t, z: FBEqs(t, z, nphi, mDM, sDM, xilog10),
                               [0., 1.05*abs(xflog10)], v0, method=meth, events=(StopM,SBH_Rad), 
                               dense_output=dens_out, rtol=1.e-7, atol=atol) 
            
            if not solFBE.success: 
                print(solFBE)
                print(colored("didn't work!",'red'))
                exit()

            if solFBE.t[-1] < 0.:
                print(solFBE)
                print(xfw, tau, 1.05*xflog10)
                break

            if solFBE.t_events[1].shape[0] > 0:
                x_Page = solFBE.t_events[1][0]
                t_Page = solFBE.sol(solFBE.t_events[1][0])[7]
                if not et_test:
                    print(colored("Warning : Page time passed, proceed with caution", "red"))
                    print(colored("Page time = {0:.6E} * t_ev, PBH mass at Page time = {1:.6E} * Min".format(t_Page/10.**tau, solFBE.sol(x_Page)[0]/Min),'blue'))

                if et_test: break

            # Concatenating solutions
            
            xBE    = np.append(xBE,    solFBE.t[:] + xilog10)
            MBHBE  = np.append(MBHBE,  solFBE.y[0,:])
            astBE  = np.append(astBE,  solFBE.y[1,:])
            SBHBE  = np.append(SBHBE,  solFBE.y[2,:])
            SRDBE  = np.append(SRDBE,  solFBE.y[3,:])
            RadBE  = np.append(RadBE,  solFBE.y[4,:])
            PBHBE  = np.append(PBHBE,  solFBE.y[5,:])
            TBE    = np.append(TBE,    solFBE.y[6,:])
            tmBE   = np.append(tmBE,   solFBE.y[7,:])
            NDMHBE = np.append(NDMHBE, solFBE.y[8,:])

            # Updating values of initial parameters
            
            Mi    = solFBE.y[0,-1]
            asi   = solFBE.y[1,-1]
            SBHi  = solFBE.y[2,-1]
            SRadi = solFBE.y[3,-1]
            rRadi = solFBE.y[4,-1]
            rPBHi = solFBE.y[5,-1]
            Ti    = solFBE.y[6,-1]
            ti    = solFBE.y[7,-1]
            NDMHi = solFBE.y[8,-1]
            
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
                        
        return [xBE, tmBE, MBHBE, astBE, SBHBE, SRDBE, RadBE, PBHBE, TBE, NDMHBE, Tev]

    #------------------------------------------------------------#
    #                                                            #
    #                     Conversion to Oh^2                     #
    #                                                            #
    #------------------------------------------------------------#
    
    def Omega_h2(self):
        '''
        This function directly returns Omega_h2, using the solution above
        '''

        x, t, MBH, ast, SBH, SRD, Rad, PBH, TUn, NDMH, Tev = self.Solt()
        
        nphi = (2.*zeta(3)/np.pi**2)*TUn[0]**3             # Initial photon number density
        
        rc = 1.053672e-5*bh.cm_in_invkeV**-3*1.e-18   # Critical density in GeV^3
        
        T0 = 2.34865e-13  # Temperature today in GeV
        
        Oh2  = NDMH[-1] * nphi * 10.**(-3.*x[-1]) * 10.**self.mDM * (bh.gstarS(T0)/bh.gstarS(Tev))*(T0/Tev)**3*(1/rc)

        return Oh2
        
        

        
