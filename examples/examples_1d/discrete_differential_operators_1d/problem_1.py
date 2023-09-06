import numpy as np

from numpy import pi


def f(x_numpy):

    Jx = x_numpy.shape[0]

    dx = x_numpy[1] - x_numpy[0]

    Lx = Jx * dx

    return np.exp(np.sin(2 * pi * x_numpy / Lx))


def f_x(x_numpy):

    Jx = x_numpy.shape[0]

    dx = x_numpy[1] - x_numpy[0]

    Lx = Jx * dx

    return (2 * pi / Lx) * np.cos(2 * pi * x_numpy / Lx) * f(x_numpy)


def f_xx(x_numpy):

    Jx = x_numpy.shape[0]

    dx = x_numpy[1] - x_numpy[0]

    Lx = Jx * dx

    return (-(2 * pi / Lx) ** 2 * np.sin(2 * pi * x_numpy / Lx) * f(x_numpy)
            + (2 * pi / Lx) * np.cos(2 * pi * x_numpy / Lx) * f_x(x_numpy))
