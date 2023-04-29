import numpy as np


class fig_density(object):

    def __init__(self, ax, settings):

        Jx = settings.Jx
        Jy = settings.Jy

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        density = np.zeros((Jx, Jy))

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        self.image_density = ax.imshow(
            density,
            extent=[left, right, bottom, top],
            cmap=settings.cmap_density,
            aspect='auto',
            interpolation='bilinear',
            vmin=0,
            vmax=1,
            origin='lower')

        # ax.set_title(r'$|\psi(x,y)|^2$', fontsize=settings.fontsize_titles)
        # ax.set_title(r'$|\rho(x,y)|^2 \, / \, \max \, |\rho(x,y)|^2$', fontsize=settings.fontsize_titles)
        ax.set_title('density', fontsize=settings.fontsize_titles)

        # self.flag_1st_function_call = False
        # self.density_max = None

    def update(self, density):

        # if not self.flag_1st_function_call:
        #
        #     self.density_max = 2 * np.max(density)
        #
        #     self.flag_1st_function_call = True
        #
        # self.image_density_xy.set_data(density / self.density_max)

        self.image_density.set_data(density)
