import numpy as np


class fig_density(object):

    def __init__(self, ax, settings):

        Jx = settings.Jx

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        density = np.zeros((Jx, Jy))

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        self.image = ax.imshow(
            density,
            extent=[left, right, bottom, top],
            cmap=settings.cmap_density,
            aspect='auto',
            interpolation='bilinear',
            vmin=0,
            vmax=1,
            origin='lower')

        ax.set_title(r'$\rho$ (scaled)', fontsize=settings.fontsize_titles)

    def update(self, density):

        density_scaled = density / np.max(density)

        self.image.set_data(density_scaled)
