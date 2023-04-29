import numpy as np


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

        # ax.set_title(r'$|\psi(x,y)|^2$', fontsize=settings.fontsize_titles)
        # ax.set_title(r'$\cos \varphi(x,y)$', fontsize=settings.fontsize_titles)
        ax.set_title('phase', fontsize=settings.fontsize_titles)

        # self.flag_1st_function_call = False
        # self.density_max = None

    def update(self, psi):

        density = np.abs(psi)**2

        alpha = density / np.max(density)

        image_phase = alpha * np.cos(np.angle(psi))

        self.image_phase.set_data(image_phase)
