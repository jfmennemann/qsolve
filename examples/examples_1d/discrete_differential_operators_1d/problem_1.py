import numpy as np

from numpy import pi


def f(x, Lx):

    return np.exp(np.sin(2 * pi * x / Lx))
    # return (1-np.cos((pi * x / Lx))**2)**4 * (1 + 0.2 * np.sin(201 * pi * x / Lx))


def f_x(x, Lx):

    return (2 * pi / Lx) * np.cos(2 * pi * x / Lx) * f(x, Lx)


def f_xx(x, Lx):

    return (-(2 * pi / Lx) ** 2 * np.sin(2 * pi * x / Lx) * f(x, Lx)
            + (2 * pi / Lx) * np.cos(2 * pi * x / Lx) * f_x(x, Lx))
