from scipy.interpolate import pchip_interpolate

import numpy as np


def compute_u(t, t_final):

    vec_t = np.array([0.0, 0.1, 0.2, 1.0]) * t_final
    vec_u = np.array([0.0, 0.0, 1.0, 1.0])

    u1 = pchip_interpolate(vec_t, vec_u, t)

    return u1
