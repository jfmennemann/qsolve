import numpy as np


class FigDensity2D(object):

    def __init__(self, ax, density, settings):

        Jx = settings.Jx
        Jy = settings.Jy

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        self.image = ax.imshow(
            density / np.max(density),
            extent=[left, right, bottom, top],
            cmap=settings.cmap_density,
            aspect='auto',
            interpolation='bilinear',
            vmin=0,
            vmax=1,
            origin='lower')

        ax.set_title(r'$|\psi|^2$ (scaled)', fontsize=settings.fontsize_titles)
