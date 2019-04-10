#!/usr/bin/env python3
from sympy import symbols # type: ignore
from sympy import Piecewise
from sympy import integrate
from sympy import lambdify
from sympy import sin, cos, pi

import numpy as np # type: ignore

import matplotlib.pyplot as plt # type: ignore
import matplotlib.animation as animation # type: ignore
from matplotlib.animation import FuncAnimation # type: ignore

def clrange(a, b):
    return range(a, b + 1)


x, t = symbols('x t', real=True)

L = 1 # string length

AA = 1 # max amplitude
# TODO ok, maybe try tunneling first?..
f = Piecewise(
    (0    , x < 0),

    # TODO huh? triangle is suspiciously stable
    # (x    , x <= L/2),
    # (L-x  , x <= L),

    # (x    , x <= L/4),
    # (L/2-x  , x <= L/2),

    # (sin(pi / (L) * x)        , x <= L),
    # (sin(pi / (L / 2) * (x - L/2)), x <= L),
    (sin(pi / (L / 3) * x), x <= L / 3),
    # (sin(pi / (L / 2) * x), x <= L / 2),
    # (sin(pi / (L / 2) * (x - L/2)), x <= L),
    (0    , True),
)
# TODO specify initial time derivative?

assert f.subs(x, 0) == 0, "BC on the left is violated!"
assert f.subs(x, L) == 0, "BC on the right is violated!"

# https://ocw.mit.edu/courses/mathematics/18-303-linear-partial-differential-equations-fall-2006/lecture-notes/waveeqni.pdf
N = 5 # fourier modes
Ns = clrange(1, N)
A = [
    2 * integrate(f * sin(pi * n * x), (x, 0, L))
    for n in Ns
]
B = [
    0
    for n in Ns
]

# TODO right, so stuff comping from schrodinger equation basically going to inflate frequency (instead of 2 * freq, 2^2 * freq)



# TODO huh, funny enough, would be nicer if it didn't have indexing operator so we wouldn't try to index with 1-based mode number
# normal modes of vibration
C = [
    # exp(i (-Et)) = cos (-Et ) + i sin(Et)
    a * cos(pi * n * t) # + b * sin(pi * n * t)
    # a * cos(- n ** 2 * t)
    for n, a, b in zip(Ns, A, B)
]
C_np = [
    lambdify(t, c, "numpy")
    for c in C
]

# hmmm... so real and imag parts are evolving as normal waves? a bit boring...

C0_num = [c(0.0) for c in C_np]

# ok, it's actually these coefficients that we wanna feed into sound modulation

# TODO test with phone?...

# TODO might need faster base frequency? Or just speed up time, should be same right?

# TODO ok, fund. frequency is implicitly 1 here (the small w)
u = sum([
    c * sin(pi * n * x)
    for c, n in zip(C, Ns)
])
u_np = lambdify([t, x], u, "numpy")


Tmax = 10
time_steps = 1000


points = 500


# https://borismus.github.io/spectrogram/
# pretty good for validation!
# TODO sampling back to get fourier coefficients from quantum thing is gonna be tricky..
def music():
    F = 440 # base frequency, for the fundamental mode
    for c, n in zip(C0_num, Ns):
        print(f"{n * F:<5}: {c:.2f}")

    M = 126 # max allowed molume (maxbyte / 2)
    mvol = M / len(C0_num) # max allowed volume for each normal mode

    seconds = 5
    Fs = 11025 # 44100
    samples = Fs * seconds

    w = 2 * np.pi / Fs

    wav = []

    for s in range(samples):
        if s % 1000 == 0:
            print(s)
        tt = s / Fs
        wv = mvol * np.sum([
            c * np.sin(s * w * n * F)
            for n, c in zip(Ns, C0_num)
        ])
        wav.append(wv)

    import struct
    with open('test.wav', 'wb') as fo:
        for i in wav:
            i = 128 + int(i)
            fo.write(struct.pack('B', i))

    import subprocess
    cmd = [
        'aplay',
        '-f', 'U8',
        '-r' + str(Fs),
        'test.wav',
       ]
    while True:
        subprocess.run(cmd)
# basically, c[n](t) is the amplitude of nth fund frequency at time t

# TODO split into wave equation bit and schrodinger?..
def do_plots():
    # TODO multiple threads?..
    # TODO wonder if we can do it in realtime??
    # calc the fourier coeff; play sound
    # ims.append((plt.pcolor(x, y, base + add, norm=plt.Normalize(0, 30)),))

    # im_ani = animation.ArtistAnimation(fig, ims, interval=200, blit=True) # , repeat_delay=3000, blit=True)
    # To save this second animation with some metadata, use the following command:
    # im_ani.save('im.mp4', metadata={'artist':'Guido'})

    fig, ax = plt.subplots(figsize=(20, 5))
    xdata, ydata = [], []
    ln, = plt.plot([], [], '.', animated=True)
    cfmt = '{:.3f}'
    time_template = 'time = {:.1f}s, C = {}'
    time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
    def init():
        ax.set_xlim(0-0.1, L + 0.1)
        ax.set_ylim(-AA-0.1, AA+0.1)
        return ln,

    def update(tt):
        xs = np.linspace(0.0, L, points)
        ys = u_np(tt, xs)

        ax.set_title(f'hi {tt}')
        ln.set_data(xs, ys)

        # TODO hmm... I don't have to do any time simulations whatsoever??
        # the modes will always be there just by the nature of initial conditions..

        cs = [cfmt.format(c_np(tt)) for c_np in C_np]
        # TODO but, what we are actually interested at is

        txt = time_template.format(tt, cs)
        time_text.set_text(txt)
        return ln, time_text

    ani = FuncAnimation(fig, update, frames=np.linspace(0, Tmax, time_steps), init_func=init, blit=True, interval=100)

    # Set up formatting for the movie files
    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
    # ani.save('im.mp4', writer=writer)
    plt.show()


# music()
do_plots()


# TODO hmm... unclear how to interpret non-zero complex amplitude inside the barrier?
