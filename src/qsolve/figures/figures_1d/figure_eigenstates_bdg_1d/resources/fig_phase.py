import numpy as np

import matplotlib.pyplot as plt


def activation(x):

    assert(np.all(x >= 0))

    # ---------------------------------------------------------------------------------------------
    """
    x1 = 0.05
    
    y = np.ones_like(x)

    indices_selected = x < x1

    y[indices_selected] = 0.5 * (1 - np.cos(np.pi * x[indices_selected] / x1))

    return y
    """
    # ---------------------------------------------------------------------------------------------

    # ---------------------------------------------------------------------------------------------
    x1 = 0.05

    s = 0.005

    y = 1.0 / (1.0 + np.exp(-(x - x1) / s))

    assert (np.all(y >= 0))

    return y
    # ---------------------------------------------------------------------------------------------


if __name__ == "__main__":

    fig, ax = plt.subplots(figsize=(5, 5))

    x = np.linspace(0.0, 1.0, 1000)
    y = activation(x)

    # ax.plot(x, x, linewidth=1, linestyle='--', color='k')
    ax.plot(x, y, linewidth=1, linestyle='-', color='k')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()


class fig_phase(object):

    def __init__(self, ax, settings):

        Jx = settings.Jx
        Jy = settings.Jy

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        phase = np.zeros((Jx, Jy))

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        self.image_phase = ax.imshow(
            phase,
            extent=[left, right, bottom, top],
            cmap=settings.cmap_phase,
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=+1,
            origin='lower')

        # ax.set_title('phase', fontsize=settings.fontsize_titles)
        # ax.set_title(r'$|\psi(x,y)|^2$', fontsize=settings.fontsize_titles)
        ax.set_title(r'$\sigma(\rho(x,y)) \, \cos \varphi(x,y)$', fontsize=settings.fontsize_titles)


        # self.flag_1st_function_call = False
        # self.density_max = None

    def update(self, psi):

        density = np.abs(psi)**2

        density_scaled = density / np.max(density)

        # alpha = density / np.max(density)

        # image_phase = alpha * np.cos(np.angle(psi))

        # density_normalized = density / np.max(density)

        # mask = density_normalized > 1e-3

        # image_phase = mask * np.cos(np.angle(psi))

        image_phase = activation(density_scaled) * np.cos(np.angle(psi))

        self.image_phase.set_data(image_phase)
