import numpy as np
from numpy.fft import fft
import matplotlib.pyplot as plt

class twoD:
    a = 4
    b = -a

    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.x = np.linspace(-self.a, self.a, N)
        self.y = np.linspace(-self.a, self.a, N)
        self.u = np.linspace(-self.a, self.a, M)
        self.v = np.linspace(-self.a, self.a, M)
        self.hx = 2 * self.a / N
        self.hy = 2 * self.a / N
        self.gaus = self.gaus2D()
        self.fx = self.fx2D()
        self.gausfft = self.fft2D(self.gaus)
        self.fxfft = self.fft2D(self.fx)

    def gaus2D(self):
        f = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                f[i][j] = np.exp(-self.x[j] ** 2 - self.y[i] ** 2)
        return f

    def fx2D(self):
        f = np.zeros((self.N, self.N), dtype=complex)
        for i in range(self.N):
            for j in range(self.N):
                f[i][j] = np.sinc(self.x[i] / np.pi) * np.sinc(self.y[i] / np.pi)
        return f

    def fft1D(self, f):
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

    def fft2D(self, fx):
        F_row = np.zeros((self.N, self.M), dtype=complex)
        for i in range(0, self.N):
            F_row[i] = self.fft1D(fx[i])
        F_col = np.zeros((self.M, self.M), dtype=complex)
        for i in range(0, self.M):
            F_col[:, i] = self.fft1D(F_row[:, i])
        return F_col

    def show(self, f):
        fig0, ax0 = plt.subplots()
        fig1, ax1 = plt.subplots()

        ax0.imshow(np.abs(f), extent=[self.b, self.a, self.b, self.a])
        ax1.imshow(np.angle(f), extent=[self.b, self.a, self.b, self.a])

        plt.show()

    def showGaus(self):
        self.show(self.gaus)
        self.show(self.gausfft)

    def showF(self):
        self.show(self.fx)
        self.show(self.fxfft)
