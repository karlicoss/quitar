#!/usr/bin/env python3
from sympy import symbols # type: ignore
from sympy import Piecewise, log, ITE, piecewise_fold, integrate, sin, cos, pi
# from sympy.abc import x, y # type: ignore

# f = x**2
# g = log(x)
# p = Piecewise((0, x < -1), (f, x <= 1), (g, True))
x, t = symbols('x t', real=True)

q = symbols('q', real=True)

L = 1

AA = 1
f = Piecewise(
    (0    , x < 0),

    # (x    , x <= L/4),
    # (L/4-x, x <= L/2),
    (sin(pi / (L / 2) * x), x <= L/2),

    (0    , True),
)

# print(f.subs(0))
# print(f.subs(x, 1))

def A(n):
    return 2 * integrate(f * sin(pi * n * x), (x, 0, L))

def B(n):
    return 0.0

N = 5
Ns = range(1, N + 1)
a = [0] + [A(n) for n in Ns]
b = [0] + [B(n) for n in Ns]

# print(a)

# TODO shit, that can be analytical right?...

u = sum([(a[n] * cos(pi * n * t) + b[n] * sin(pi * n * t)) * sin(pi * n * x) for n in Ns])
# print(u)
# 
# # def u(x, t):
# #     return (f(x - c * t) + f(x + c * t)) / 2

# plots = [plot(u(x, tt), (x, 0, L), ymax=1, ymin=-1) for tt in sxrange(0, Tmax, Tmax/steps)]
# plots = [plot(x ** 2 * tt, (x, 0, L), ymax=10) for tt in sxrange(0, 10.0, 1.0)]

# plots = [plot(lambda xx,tt=tt: u(xx, tt), 0, L, ymax=1, ymin=-1) for tt in sxrange(0, Tmax, Tmax/steps)]
# a = animate(plots)
# a.show(delay=20, iterations=steps)
# 
# print(p)
Tmax = 4
steps = 100
points = 50

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

fig = plt.figure()
plt.xlim(0-0.1, L + 0.1)
plt.ylim(-AA-0.1, AA+0.1)
# TODO multiple threads?..
ims = []
# TODO ugh
for tt in np.arange(0.0, Tmax, Tmax/steps):
    print(f'processing {tt}')
    ut = u.subs(t, tt)

    xs = np.linspace(0.0, L, points)
    us = [ut.subs(x, xx) for xx in xs]
    # print(tt)
    pp = plt.plot(xs, us)
    ims.append(pp)
    # TODO wonder if we can do it in realtime??
    # calc the fourier coeff; play sound
    # ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

im_ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True) # , repeat_delay=3000, blit=True)
# To save this second animation with some metadata, use the following command:
# im_ani.save('im.mp4', metadata={'artist':'Guido'})

plt.show()
