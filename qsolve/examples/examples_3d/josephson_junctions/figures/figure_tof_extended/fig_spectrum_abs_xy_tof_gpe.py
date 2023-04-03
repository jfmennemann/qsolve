import numpy as np


class fig_spectrum_abs_xy_tof_gpe(object):

    def __init__(self, ax, settings):

        ax.set_title('spectrum (GPE)')

        ax.set_xlabel(r'$i_x$')
        ax.set_ylabel(r'$i_y$')

        ax.set_xlim([-settings.Jx_tof_gpe // 2, settings.Jx_tof_gpe // 2 - 1])
        ax.set_ylim([-settings.Jy_tof_gpe // 2, settings.Jy_tof_gpe // 2 - 1])

        ax.set_xticks([-settings.Jx_tof_gpe // 2, 0, settings.Jx_tof_gpe // 2 - 1])
        ax.set_yticks([-settings.Jy_tof_gpe // 2, 0, settings.Jy_tof_gpe // 2 - 1])

        spectrum_abs_xy_tof_gpe = np.zeros((settings.Jx_tof_gpe, settings.Jy_tof_gpe))

        left = -settings.Jx_tof_gpe // 2
        right = settings.Jx_tof_gpe // 2 - 1

        bottom = -settings.Jy_tof_gpe // 2
        top = settings.Jy_tof_gpe // 2 - 1

        self.image_spectrum_abs_xy_tof_gpe = ax.imshow(
            np.transpose(spectrum_abs_xy_tof_gpe),
            extent=[left, right, bottom, top],
            cmap=settings.cmap_spectrum_abs,
            aspect='auto',
            interpolation='bilinear',
            vmin=0, vmax=1,
            origin='lower')

    def update(self, spectrum_abs_xy_tof_gpe):

        self.image_spectrum_abs_xy_tof_gpe.set_data(np.transpose(spectrum_abs_xy_tof_gpe))
