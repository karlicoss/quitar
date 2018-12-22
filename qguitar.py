#!/usr/bin/env python3
from sympy import symbols # type: ignore
from sympy import Piecewise, log, ITE, piecewise_fold, integrate, sin, cos, pi
# from sympy.abc import x, y # type: ignore
import numpy as np

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
    # (sin(pi / (L / 2) * x), x <= L/2),
    (sin(pi / L * x), x <= L),

    (0    , True),
)

# print(f.subs(0))
# print(f.subs(x, 1))

def fA(n):
    return 2 * integrate(f * sin(pi * n * x), (x, 0, L))

def fB(n):
    return 0.0

N = 1
Ns = range(1, N + 1)
A = [fA(n) for n in Ns]
B = [fB(n) for n in Ns]
# TODO test with phone?...

# print(a)

# TODO shit, that can be analytical right?...

# TODO ok, fund. frequency is implicitly 1 here (the small w)

C = [a * cos(pi * n * t) + b * sin(pi * n * t) for n, a, b in zip(Ns, A, B)]
from sympy import lambdify
CC = [lambdify(t, c, "numpy") for c in C]

u = sum([c * sin(pi * n * x) for c, n in zip(C, Ns)])

u_fast = lambdify([t, x], u, "numpy")

Tmax = 6
steps = 100
points = 50


def music():
    M = 126
    mvol = M / len(CC)

    seconds = 5
    Fs = 44100
    samples = Fs * seconds


    w = 2 * np.pi / Fs

    wav = []

    F = 440
    for s in range(samples):
        if s % 1000 == 0:
            print(s)
        # TODO shit. too slow?
        tt = s / Fs
        wv = mvol * np.sum([c(tt) * np.sin(s * w * n * F) for n, c in zip(Ns, CC)])
        wav.append(wv)

    import struct
    with open('test.wav', 'wb') as fo:
        for i in wav:
            i = 128 + int(i)
            fo.write(struct.pack('B', i))

    import subprocess
    subprocess.run([
        'aplay',
        '-f', 'U8',
        '-r' + str(Fs),
        'test.wav',
       ])
# basically, c[n](t) is the amplitude of nth fund frequency at time t

def do_plots():
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation

    # TODO multiple threads?..
    ims = []
    # TODO ugh
   #  for tt in np.arange(0.0, Tmax, Tmax/steps):
   #      print(f'processing {tt}')

   #      xs = np.linspace(0.0, L, points)
   #      us = [ut.subs(x, xx) for xx in xs]
   #      # print(tt)
   #      pp = plt.plot(xs, us, title=f'{tt}')
   #      ims.append(pp)
        # TODO wonder if we can do it in realtime??
        # calc the fourier coeff; play sound
        # ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

    # im_ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True) # , repeat_delay=3000, blit=True)
    # To save this second animation with some metadata, use the following command:
    # im_ani.save('im.mp4', metadata={'artist':'Guido'})

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [], 'r.', animated=True)
    time_template = 'time = %.1fs'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def init():
        ax.set_xlim(0-0.1, L + 0.1)
        ax.set_ylim(-AA-0.1, AA+0.1)
        return ln,

    def update(tt):
        # TODO pythonlise?..
        # TODO need to erase first?..
        xs = np.linspace(0.0, L, points)
        ys = u_fast(tt, xs)

        ax.set_title(f'hi {tt}')
        ln.set_data(xs, ys)
        time_text.set_text(time_template % (tt))
        return ln, time_text

    ani = FuncAnimation(fig, update, frames=np.linspace(0, Tmax, steps), init_func=init, blit=True, interval=100)
    plt.show()


# music()
do_plots()
