# -*- coding: utf-8 -*-



# Tests suit for Hyperbolic Conservations Laws
# ============================================



import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import square

third=1.0/3.0


def slot(x, x0=[-third, third], y=[0, 1], domain=[-1, 1]):
    assert(len(x0) == 2)
    assert(len(y) == 2)
    return y[0]+(y[1]-y[0])*(np.heaviside(x-x0[0], 0.5)-np.heaviside(x-x0[1], 0.5))

def trapeze(x, x0=[-third, 0, third], y=[0, 1], domain=[-1, 1]):
    assert(len(x0) == 3)
    assert(len(y) == 2)
    epsilon = 1e-6
    xp = [domain[0], x0[0], x0[1], x0[2]-epsilon, x0[2]+epsilon, domain[1]]
    fp = [     y[0],  y[0],  y[1],  y[1]        ,  y[0]        ,      y[0]]
    return np.interp(x, xp, fp)


class TestHCL():
    """
    This is the support class for the 5 tests where are defined
    :param tag : index of the test (1 <= tag <= 5)
    :param tFinal : final time for the time integration of the equation
    :param domain : 2-elements tuple containing the domain of the simulation
    :param nx : number of grid points
    :param nu : ratio dt / dx which then define the time step
    :param flux : callable for the flux function
    :param u_star : single value (need some work for a list of values) of the value at the sonic point
    :param u0 : callable for the initial u function
    :param uFinal : discrete values at the x location of the solution at tFinal
    """
    def __init__(self,\
                 tag = None,\
                 tFinal = 0,\
                 domain = (-1.0,+1.0),\
                 nx = None,\
                 nu = 0.8,\
                 flux = None,\
                 u_star = None,\
                 u0 = None,\
                 a = None):
        assert(isinstance(tag, int))
        assert(tag > 0)
        assert(tag <= 5)
        self.tag = tag
        self.domain = domain
        self.nx = nx
        self.dx = (domain[1]-domain[0])/nx
        self.nu = nu
        self.tFinal = tFinal
        self.dt = round(nu*self.dx, 6)
        self.u0 = u0
        self.flux = flux
        self.u_star = u_star
        self.a = a
        self.x = np.linspace(domain[0], domain[1]-self.dx, nx)


    def __repr__(self):
        return f"""...................................................
{"Tag number":>24} : {self.tag}
{"Final time":>24} : {self.tFinal}
{"domain":>24} : {self.domain}
{"Number of grid points":>24} : {self.nx}
{"nu ":>24} : {self.nu}
{"dx ":>24} : {self.dx}
{"dt ":>24} : {self.dt}
{"a (wave speed) ":>24} : {self.a}"""


class Test1(TestHCL):
    def u0(self, x):
        return -np.sin(np.pi*x)

    def __init__(self):
        super().__init__(tag = 1,\
                         tFinal =30,\
                         nx = 40,\
                         flux = lambda u:u,\
                         u_star = None,\
                         u0 = self.u0,\
                         a = lambda x:1)
        self.uFinal = self.u0(self.x)


class Test2(TestHCL):
    def u0(self, x):
        return slot(x)

    def __init__(self):
        super().__init__(tag = 2,\
                         tFinal = 4,\
                         nx = 40,\
                         flux = lambda u:u,\
                         u_star = None,\
                         u0 = self.u0,\
                         a = lambda x:1)
        self.uFinal = self.u0(self.x)


class Test3(TestHCL):
    def u0(self, x):
        return slot(x)

    def __init__(self):
        super().__init__(tag = 3,\
                         tFinal = (4,40),\
                         nx = 600,\
                         flux = lambda u:u,\
                         u_star = None,\
                         u0 = self.u0,\
                         a = lambda x:1)
        self.uFinal=self.u0(self.x)


class Test4(TestHCL):
    def u0(self, x):
        return slot(x)

    def __init__(self):
        super().__init__(tag = 4,\
                         tFinal = 0.6,\
                         nx = 40,\
                         flux = lambda x:0.5*x*x,\
                         u_star = 0.,\
                         u0 = self.u0,\
                         a = lambda x:x)
        self.uFinal = trapeze(self.x, x0=[-third, -third+1*self.tFinal, third+0.5*self.tFinal], y=[0, 1])


class Test5(TestHCL):
    def u0(self, x, y=[-1, 1]):
        return slot(x, y=y)

    def __init__(self):
        super().__init__(tag = 5,\
                         tFinal = 0.3,\
                         nx = 40,\
                         flux = lambda x:0.5*x*x,\
                         u_star = 0.,\
                         u0 = self.u0,\
                         a = lambda x:x)
        self.uFinal = trapeze(self.x, x0=[-third-1*self.tFinal, -third+1*self.tFinal, third], y=[-1, 1])

