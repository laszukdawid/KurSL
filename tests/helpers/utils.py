import numpy as np


def random_theta(oscN, nH):
    W = np.random.random((oscN, 1)) * 20 + 10
    P = np.random.random((oscN, 1)) * 6.28
    R = np.random.random((oscN, 1)) * 5 + 1
    K = np.random.random((oscN, nH * (oscN - 1))) * 2 - 1
    return np.hstack((W, P, R, K))


def peak_norm(t, t0, amp, s):
    _t = (t - t0) / s
    return amp * np.exp(-_t * _t)


def peak_triangle(t, t0, amp, s):
    x = amp * (1 - np.abs((t - t0) / s))
    x[x < 0] = 0
    return x


def peak_lorentz(t, t0, amp, s):
    _t = (t - t0) / (0.5 * s)
    return amp / (_t * _t + 1)


def cosine(t, f, a, ph):
    return a * np.cos(f * 2 * np.pi * t + ph)
