import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

class oneD:
    a = 4

    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.x = np.linspace(-self.a, self.a, N)
        self.u = np.linspace(-self.a, self.a, M)
        self.hx = 2*self.a / N
        self.gaus = np.exp(-self.x**2)
        self.fx = np.sinc(self.x / np.pi)
        self.gausfft = self.myfft(self.gaus)
        self.fxfft = self.myfft(self.fx)
        self.gausfinit = self.finite(self.gaus)
        self.fxfinit = self.finite(self.fx)

    def myfft(self, f):
        result = np.zeros(self.M, dtype=complex)
        i = int((self.M - self.N) / 2)
        m = int(self.M/2)
        result[i:(self.M - i)] = f
        temp = np.copy(result)
        result[:m], result[m:] = temp[m:], temp[:m]
        result = fft(result) * self.hx
        temp = np.copy(result)
        result[:int(self.M / 2)], result[int(self.M / 2):] = temp[int(self.M / 2):], temp[:int(self.M / 2)]
        return result

    def finite(self, f):
        F = np.zeros(self.M, dtype=np.complex)
        for i in range(self.M):
            F[i] = self.hx * np.sum(f * np.exp(-2 * np.pi * np.complex(0, 1) * self.u[i] * (self.x)))
        return F

    def show(self, f, fft, finit):
        fig1, ax1 = plt.subplots()
        ax1.plot(self.x, abs(f), color="blue", label="input")
        ax1.plot(self.u, abs(fft), color="red", label="fft")
        ax1.plot(self.u, abs(finit), color="green", label="finite")
        ax1.legend()
        plt.grid()
        fig2, ax2 = plt.subplots()
        ax2.plot(self.x, np.angle(f), color="blue", label="input")
        ax2.plot(self.u, np.angle(fft), color="red", label="fft")
        ax2.plot(self.u, np.angle(finit), color="green", label="finite")
        ax2.legend()
        plt.grid()

        plt.show()

    def showGaus(self):
        self.show(self.gaus, self.gausfft, self.gausfinit)

    def showF(self):
        self.show(self.fx, self.fxfft, self.fxfinit)