# -*- coding: utf-8 -*-



# Godunov first order upwind scheme
# ============================================



import numpy as np
import misc as mi

epsilon = 1e-6

class Godunov1():

    def __init__(self, testCase, form='vanilla'):
        self.form = form
        self.dx = testCase.dx
        self.dt = testCase.dt
        self.tFinal = testCase.tFinal
        self.nu = testCase.nu
        self.u0 = testCase.u0
        self.flux = testCase.flux
        self.u_star = testCase.u_star
        self.a = testCase.a
        self.x = testCase.x
        self.uFinal = testCase.uFinal


    def fillFlux0_viscosity(self, w, f, a):
        N = w.shape[0]

        a_ = np.empty(N)
        diff_ = np.empty(N)
        for j in range(1, N-1):
            if (np.fabs(w[j]-w[j-1]) < epsilon):
                a_[j] = a(w[j])
                diff_[j] = 0
            else:
                a_[j] = (f(w[j])-f(w[j-1]))/(w[j]-w[j-1])
                diff_[j] = (f(w[j])-2*f(0.)+f(w[j-1]))/(w[j]-w[j-1])

        e_ = np.empty(N)
        for j in range(1, N-1):
            if ((w[j-1] * w[j] > 0) & (np.fabs(w[j]-w[j-1]) > epsilon)):
                e0_ = (f(w[j])-f(w[j-1])) / (w[j]-w[j-1])
                e1_ = (f(w[j-1])-f(w[j])) / (w[j]-w[j-1])
                e_[j] = np.max([e0_, e1_])
            else:
                e_[j] = np.max([np.fabs(a_[j]), diff_[j]])

        flux = np.empty(N)
        flux[1:-1] = 0.5 *             (f(w[1:-1])+f(w[0:-2]))\
                   - 0.5 * e_[1: -1] * (  w[1:-1] -  w[0:-2])
        flux = mi.fillGhosts(flux)
        return flux


    def fillFlux1_viscosity(self, w, f, a):
        N = w.shape[0]

        a_ = np.empty(N)
        diff_ = np.empty(N)
        for j in range(1, N-1):
            if (np.fabs(w[j+1]-w[j]) < epsilon):
                a_[j] = a(w[j])
                diff_[j] = 0
            else:
                a_[j] = (f(w[j+1])-f(w[j]))/(w[j+1]-w[j])
                diff_[j] = (f(w[j+1])-2*f(0.)+f(w[j]))/(w[j+1]-w[j])

        e_ = np.empty(N)
        for j in range(1, N-1):
            if ((w[j] * w[j+1] > 0) & (np.fabs(w[j]-w[j+1]) > epsilon)):
                e0_ = (f(w[j+1])-f(w[j])) / (w[j+1]-w[j])
                e1_ = (f(w[j])-f(w[j+1])) / (w[j+1]-w[j])
                e_[j] = np.max([e0_, e1_])
            else:
                e_[j] = np.max([np.fabs(a_[j]), diff_[j]])

        flux = np.empty(N)
        flux[1:-1] = 0.5 *             (f(w[2:])+f(w[1:-1]))\
                   - 0.5 * e_[1: -1] * (  w[2:] -  w[1:-1])
        flux = mi.fillGhosts(flux)
        return flux


    def fillFlux0_vanilla(self, w, f, a):
        N = w.shape[0]
        flux = np.empty(N)
        for j in range(1, N-1):
            if w[j-1] < w[j]:
                if w[j-1] * w[j] > 0:
                    flux[j] = np.min([f(w[j-1]), f(w[j])])
                else:
                    flux[j] = np.min([f(w[j-1]), f(w[j]), 0.])
            else:
                if w[j-1] * w[j] > 0:
                    flux[j] = np.max([f(w[j-1]), f(w[j])])
                else:
                    flux[j] = np.max([f(w[j-1]), f(w[j]), 0.])
        flux = mi.fillGhosts(flux)
        return flux


    def fillFlux1_vanilla(self, w, f, a):
        N = w.shape[0]
        flux = np.empty(N)
        for j in range(1, N-1):
            if w[j] < w[j+1]:
                if w[j] * w[j+1] > 0:
                    flux[j] = np.min([f(w[j]), f(w[j+1])])
                else:
                    flux[j] = np.min([f(w[j]), f(w[j+1]), 0.])
            else:
                if w[j] * w[j+1] > 0:
                    flux[j] = np.max([f(w[j]), f(w[j+1])])
                else:
                    flux[j] = np.max([f(w[j]), f(w[j+1]), 0.])
        flux = mi.fillGhosts(flux)
        return flux


    def fillFlux0(self, w, f, a):
        if self.form == 'vanilla':
            return self.fillFlux0_vanilla(w, f, a)
        if self.form == 'viscosity':
            return self.fillFlux0_viscosity(w, f, a)


    def fillFlux1(self, w, f, a):
        if self.form == 'vanilla':
            return self.fillFlux1_vanilla(w, f, a)
        if self.form == 'viscosity':
            return self.fillFlux1_viscosity(w, f, a)


    def compute(self, tFinal):
        Nt = int(tFinal/self.dt)
        dx = self.dx

        u0w = mi.addGhosts(self.u0(self.x))
        u0w = mi.fillGhosts(u0w)

        xw = mi.addGhosts(self.x)
        xw[0] = xw[1]-dx
        xw[-1] = xw[-2]+dx

        u1w = np.empty((u0w.shape[0]))
        F0w = np.empty((u0w.shape[0]))
        F1w = np.empty((u0w.shape[0]))

        for i in range(Nt):
            F0w = self.fillFlux0(u0w, self.flux, self.a)
            F1w = self.fillFlux1(u0w, self.flux, self.a)

            F1w = mi.fillGhosts(F1w)

            u1w[1:-1] = u0w[1:-1] - self.nu * (F1w[1:-1] - F0w[1:-1])

            u1w = mi.fillGhosts(u1w)

            u0w = u1w

        self.uF = u1w[1:-1]
