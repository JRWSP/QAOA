#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 26 13:40:43 2020

@author: quantuminw
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "serif"
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams.update({'font.size': 15})

sample = 0
grid6 = np.load("./grid/grid6nodes_p1statevector.npy", allow_pickle=True)

val = []
for ii in grid6:
    ii = list(ii.values())
    #beta.append(ii[0])
    #gamma.append(ii[1])
    val.append(ii[2])


val2 = np.array(val)
val2 = val2.reshape(100, 200)
val2 = -1*val2
val2 = np.around(val2, 3)

"""
Slice = 25
mean = np.mean(val2[Slice])
signal = val2[Slice] - mean
fourier = np.fft.fft(signal)
plt.plot(range(len(fourier)), fourier)
"""
#x = np.arange(0, 2.0*np.pi, np.pi/100)

N = 200 #samples
dt = 2.0*np.pi/N #sample spacing
T = dt*N #Period
df = 1/T # frequency in x space
dw = 2*np.pi/T # frequency in F(x) space
f = np.array([df*n if n<N/2 else df*(n-N) for n in range(N)])
w = np.array([dw*n if n<N/2 else dw*(n-N) for n in range(N)])
ny = dw*N/2 #Nyquist frequency
x = np.linspace(0.0*np.pi, 2.0*np.pi, N)
sin = np.sin(np.pi*x)
y = np.fft.rfft(sin)
plt.plot(x[:len(y)], np.abs(y))

"""
plt.figure(figsize=(10,6))
plt.contourf(val2/52.419)
#plt.xticks([0, 50, 100, 150, 200], [r"$-1.0\pi$", r"$-0.5\pi$", r"$0$", r"$0.5\pi$", r"$1.0\pi$"])
plt.xlabel(r"$\gamma$")
#plt.yticks([0, 25, 50, 75, 100], [r"$0$", r"$0.125\pi$", r"$0.250\pi$", r"$0.375\pi$", r"$0.500\pi$"])
plt.ylabel(r"$\beta$")
plt.colorbar()
plt.title("N = 6, p=2 from optimal p1 parameters")
#plt.savefig("grid6_p2", dpi=300)

"""