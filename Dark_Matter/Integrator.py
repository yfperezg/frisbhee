import numpy as np
import pandas as pd
from scipy import interpolate
import scipy.integrate as integrate
from scipy.integrate import quad, ode, solve_ivp, odeint
from scipy.optimize import root
from scipy.special import zeta, kn, spherical_jn, jv
from scipy.interpolate import RectBivariateSpline
import mpmath
from mpmath import polylog

from numpy import sqrt, log, exp, log10, pi, logspace, linspace, seterr, min, max, append
from numpy import loadtxt, zeros, floor, ceil, unique, sort, cbrt, concatenate, delete, real

from multiprocessing import Pool

class Simp1D:

    def __init__(self, f, pars, lims):

        self.f    = f
        self.pars = pars
        self.lims = lims
        
    def integrand(self, i):
        
        pars = self.pars
        
        xi, xf, Nb = self.lims
        
        dx = (xf - xi)/(2*Nb)
                
        return self.f(xi + i*dx, pars) 

    def integral(self):
        
        xi, xf, Nb = self.lims
        
        dx = (xf - xi)/(2*Nb)
        
        with Pool(8) as pool:        
            grid = pool.map(self.integrand, [i for i in range(2*Nb+1)])  
        
        res = grid[0] + grid [2*Nb]
        
        for j in range (1,Nb+1): res += 4.*grid[2*j - 1]
        for j in range (1,Nb):   res += 2.*grid[2*j]
        
        #res = self.f(xi, self.pars) + self.f(xf, self.pars)
        
        #for j in range (1,Nb+1): res += 4.*self.f(xi + (2*j - 1)*dx, self.pars)
        #for j in range (1,Nb):   res += 2.*self.f(xi + 2*j*dx, self.pars)
        
        return res*dx/3.
    

class Simp2D:

    def __init__(self, f, pars, lims):

        self.f    = f
        self.pars = pars
        self.lims = lims

    def weight(self):
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        wx = np.ones(2*Nx+1)
        wx[0]  = 1.
        wx[2*Nx] = 1.

        for i in range(1, 2*Nx):
            if i % 2 == 0:
                wx[i] = 2.
            else:
                wx[i] = 4.
        
        
        wy = np.ones(2*Ny+1)
        wy[0]  = 1.
        wy[2*Ny] = 1.

        for i in range(1, 2*Ny):
            if i % 2 == 0:
                wy[i] = 2.
            else:
                wy[i] = 4
        
        w = np.kron(wy, wx)
        
        return w
        
    def integrand(self, i, j):
        
        pars = self.pars
        
        xi, xf, yi, yf, Nx, Ny = self.lims
        
        dx = (xf - xi)/(2*Nx)
        dy = (yf - yi)/(2*Ny)
        
        #print(i, j, i + j*(2*Nx+1))
                
        return self.f(xi + i*dx, yi + j*dy, pars) 
    

    def integral(self):
        
        pars = self.pars
        
        xi, xf, yi, yf, Nx, Ny = self.lims
        
        dx = (xf - xi)/(2*Nx)
        dy = (yf - yi)/(2*Ny)
        
        #grid = zeros((2*Nx+1)*(2*Ny+1))
        
        with Pool(7) as pool:        
            grid = pool.starmap(self.integrand, [(i, j) for j in range(2*Ny+1) for i in range(2*Nx+1)])  
        
        w = self.weight()

        res = 0.

        for j in range(2*Nx+1):
            for k in range(2*Ny+1):
                res += w[(2*Ny+1)*j + k] * grid[(2*Ny+1)*j + k]
        
        return res*dx*dy/9.


class Simp2D_varlims:

    def __init__(self, f, pars, lims):

        self.f    = f
        self.pars = pars
        self.lims = lims

    def weight(self):
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        wx = np.ones(2*Nx+1)
        wx[0]  = 1.
        wx[2*Nx] = 1.

        for i in range(1, 2*Nx):
            if i % 2 == 0:
                wx[i] = 2.
            else:
                wx[i] = 4.
        
        
        wy = np.ones(2*Ny+1)
        wy[0]  = 1.
        wy[2*Ny] = 1.

        for i in range(1, 2*Ny):
            if i % 2 == 0:
                wy[i] = 2.
            else:
                wy[i] = 4
        
        w = np.kron(wy, wx)
        
        return w
        
    def integrand(self, i, j):
        
        pars = self.pars
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        dx = (x_max - x_min)/(2*Nx)
        
        xi = x_min + dx*i
        
        dy = (yx_max(xi) - yx_min(xi))/(2*Ny)
        
        yij = yx_min(xi) + j*dy
                
        return self.f(xi, yij, pars)*dx*dy/9.


    def integral(self):
        
        pars = self.pars
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        #grid = zeros((2*Nx+1)*(2*Ny+1))
        
        with Pool(7) as pool:        
            grid = pool.starmap(self.integrand, [(i, j) for j in range(2*Ny+1) for i in range(2*Nx+1)])

        w = self.weight()

        res = 0.

        for j in range(2*Nx+1):
            for k in range(2*Ny+1):
                res += w[(2*Ny+1)*j + k] * grid[(2*Ny+1)*j + k]
        
        return res

class Trap2D_varlims:

    def __init__(self, f, pars, lims):

        self.f    = f
        self.pars = pars
        self.lims = lims
        
    def weight(self):
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        wx = np.ones(Nx+1)
        wx[0]  = 0.5
        wx[Nx] = 0.5
        
        wy = np.ones(Ny+1)
        wy[0]  = 0.5
        wy[Ny] = 0.5
        
        w = np.kron(wy, wx)
        
        return w
        
    def integrand(self, j, k):
        
        pars = self.pars
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        dx = (x_max - x_min)/Nx
        
        xj  = x_min + dx*j
        
        dy = (yx_max(xj) - yx_min(xj))/Ny
        
        yjk = yx_min(xj) + k*dy
                
        return self.f(xj, yjk, pars)*dx*dy
    

    def integral(self):
        
        pars = self.pars
        
        x_min, x_max, yx_min, yx_max, Nx, Ny = self.lims
        
        result = 0.
        
        w = self.weight()
        
        with Pool(8) as pool: 
            grid = pool.starmap(self.integrand, [(j, k) for k in range(Ny+1) for j in range(Nx+1)])
        
        for j in range(Nx+1):
            for k in range(Ny+1):
                result += w[(Ny+1)*j + k] * grid[(Ny+1)*j + k]
        
        return result

class Trap3D_varlims:

    def __init__(self, f, pars, lims):

        self.f    = f
        self.pars = pars
        self.lims = lims
        
    def weight(self):
        
        x_min, x_max, yx_min, yx_max, zyx_min, zyx_max, Nx, Ny, Nz = self.lims
        
        wx = np.ones(Nx+1)
        wx[0]  = 0.5
        wx[Nx] = 0.5
        
        wy = np.ones(Ny+1)
        wy[0]  = 0.5
        wy[Ny] = 0.5
        
        wz = np.ones(Nz+1)
        wz[0]  = 0.5
        wz[Ny] = 0.5
        
        w = np.kron(wz, np.kron(wy, wx))
        
        return w
        
    def integrand(self, j, k, l):
        
        pars = self.pars
        
        x_min, x_max, yx_min, yx_max, zyx_min, zyx_max, Nx, Ny, Nz = self.lims
        
        dx = (x_max - x_min)/Nx
        
        xj  = x_min + dx*j
        
        dy = (yx_max(xj) - yx_min(xj))/Ny
        
        yjk = yx_min(xj) + k*dy
        
        dz = (zyx_max(xj, yjk) - zyx_min(xj, yjk))/Ny
        
        zjkl = zyx_min(xj, yjk) + l*dz
                
        return self.f(xj, yjk, zjkl, pars)*dx*dy*dz
    

    def integral(self):
        
        pars = self.pars
        
        x_min, x_max, yx_min, yx_max, zyx_min, zyx_max, Nx, Ny, Nz = self.lims
        
        result = 0.
        
        w = self.weight()
        
        with Pool(8) as pool: 
            grid = pool.starmap(self.integrand, [(j, k, l) for l in range(Nz+1) for k in range(Ny+1) for j in range(Nx+1)])
        
        for j in range(Nx+1):
            for k in range(Ny+1):
                for l in range(Nz+1):
                    result += w[(Ny+1)*(Nz+1)*j + (Nz+1)*k + l] * grid[(Ny+1)*(Nz+1)*j + (Nz+1)*k + l]
        
        return result
