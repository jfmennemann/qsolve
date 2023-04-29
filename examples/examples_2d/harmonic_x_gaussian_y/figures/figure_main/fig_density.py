import numpy as np


class fig_density(object):

    def __init__(self, ax, settings):

        ax.set_title(r'$|\psi(x,y)|$', fontsize=settings.fontsize_titles)
    
        Jx = settings.Jx
        Jy = settings.Jy

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)
        
        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        density_xy = np.zeros((Jx, Jy))

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        self.image_density = ax.imshow(density_xy,
                                          extent=[left, right, bottom, top],
                                          cmap=settings.cmap_density,
                                          aspect='auto',
                                          interpolation='bilinear',
                                          vmin=0,
                                          vmax=1,
                                          origin='lower')

    def update(self, density):

        self.image_density.set_data(density)
