import numpy as np


class fig_imaginary_part(object):

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
            cmap=settings.cmap_phase,
            aspect='auto',
            interpolation='bilinear',
            vmin=-1,
            vmax=+1,
            origin='lower')

        # ax.set_title('phase', fontsize=settings.fontsize_titles)
        # ax.set_title(r'$|\psi(x,y)|^2$', fontsize=settings.fontsize_titles)
        # ax.set_title(r'$\sigma(\rho(x,y)) \, \cos \varphi(x,y)$', fontsize=settings.fontsize_titles)
        ax.set_title(r'$\Im\, \psi$ (scaled)', fontsize=settings.fontsize_titles)

    def update(self, psi):

        density = np.abs(psi)**2

        density_max = np.max(density)

        imaginary_part = np.imag(psi) / np.sqrt(density_max)

        self.image.set_data(imaginary_part)
