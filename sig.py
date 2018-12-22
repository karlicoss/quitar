#!/usr/bin/env python3

# TODO ugh, that's a bit wrong...
def wave(coeff, F=220):
    import struct
    import numpy as np

    Fs = 44100
    samples = Fs * 5

    x = np.arange(samples)

    M = 126
    mvol = M / len(coeff)
    w = 2 * np.pi / Fs
    y = sum([
        mvol * c * np.sin(
            x * w * n * F
        )
        for n, c in zip(range(1, 100), coeff)
    ])

    with open('test.wav', 'wb') as fo:
        for i in y:
            i = 128 + int(i)
            fo.write(struct.pack('B', i))

    import subprocess
    subprocess.run([
        'aplay',
        '-f', 'U8',
        '-r' + str(Fs),
        'test.wav',
       ])


wave([0, 1], F=440)
