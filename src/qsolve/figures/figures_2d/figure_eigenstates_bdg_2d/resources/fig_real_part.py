import numpy as np


class fig_real_part(object):

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

        self.image = ax.imshow(
            phase,
            extent=[left, right, bottom, top],
            cmap=settings.cmap_real_part,
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=+1,
            origin='lower')

        ax.set_title(r'$\operatorname{Re}{\psi}$ (scaled)', fontsize=settings.fontsize_titles)

    def update(self, psi):

        density = np.abs(psi)**2

        density_max = np.max(density)

        real_part = np.real(psi) / np.sqrt(density_max)

        self.image.set_data(real_part)
