import matplotlib.pyplot as plt

from PyQt5 import QtWidgets

from scipy import constants

import numpy as np

from .fig_psi_abs_squared_1d import fig_psi_abs_squared_1d
from .fig_psi_re_im_1d import fig_psi_re_im_1d

from .fig_control_inputs import fig_control_inputs

from .. style import colors


class FigureMain1D(object):

    def __init__(self, x, times, params):

        n_control_inputs = params["n_control_inputs"]

        density_min = -0.2 * params["density_max"]
        density_max = +1.2 * params["density_max"]

        V_min = params['V_min']
        V_max = params['V_max']

        x_ticks = params["x_ticks"]

        x = x / 1e-6

        times = times / 1e-3

        Jx = x.shape[0]

        dx = x[1] - x[0]

        x_min = x[0]
        x_max = x_min + Jx * dx
        
        t_min = times[0]
        t_max = times[-1]

        Jx = x.size

        # -----------------------------------------------------------------------------------------
        if t_max == 2:

            t_ticks_major = np.array([0, 1, 2])

        elif t_max == 2.5:

            t_ticks_major = np.array([0, 0.5, 1, 1.5, 2, 2.5])

        elif abs(t_max - 4) < 1e-14:

            t_ticks_major = np.array([0, 1, 2, 3, 4])

        elif t_max == 5:

            t_ticks_major = np.array([0, 1, 2, 3, 4, 5])

        elif t_max == 8:

            t_ticks_major = np.array([0, 2, 4, 6, 8])

        elif t_max == 10:

            t_ticks_major = np.array([0, 2, 4, 6, 8, 10])

        elif t_max == 20:

            t_ticks_major = np.array([0, 4, 8, 12, 16, 20])

        elif t_max == 40:

            t_ticks_major = np.array([0, 10, 20, 30, 40])

        elif t_max == 80:

            t_ticks_major = np.array([0, 20, 40, 60, 80])

        elif t_max == 160:

            t_ticks_major = np.array([0, 40, 80, 120, 160])

        elif t_max == 200:

            t_ticks_major = np.array([0, 40, 80, 120, 160, 200])

        else:

            t_ticks_major = np.array([0, t_max])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        t_ticks_minor = 0.5 * (t_ticks_major[0:-1] + t_ticks_major[1:])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        settings = type('', (), {})()

        settings.hbar = constants.hbar

        settings.n_control_inputs = n_control_inputs

        settings.density_min = density_min
        settings.density_max = density_max

        settings.real_part_min = -1.2 * np.sqrt(settings.density_max)
        settings.real_part_max = +1.2 * np.sqrt(settings.density_max)

        settings.V_min = V_min
        settings.V_max = V_max

        settings.x = x

        settings.Jx = Jx

        settings.x_ticks = x_ticks

        settings.x_min = x_min
        settings.x_max = x_max

        settings.times = times

        settings.t_min = t_min
        settings.t_max = t_max

        settings.t_ticks_major = t_ticks_major
        settings.t_ticks_minor = t_ticks_minor

        settings.linecolor_V = colors.alizarin
        settings.linewidth_V = 1.1

        settings.label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        settings.label_t = r'$t \;\, \mathrm{in} \;\, \mathrm{ms}$'

        settings.cmap_density = colors.cmap_density

        settings.cmap_real_part = colors.cmap_real_part

        settings.color_gridlines_major = colors.color_gridlines_major
        settings.color_gridlines_minor = colors.color_gridlines_minor

        settings.fontsize_titles = 10
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        plt.rcParams.update({'font.size': 10})
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_name = "figure_main"
                
        self.fig = plt.figure(self.fig_name, facecolor="white")

        window = self.fig.canvas.window()
        
        window.findChild(QtWidgets.QToolBar).setVisible(False)
        window.statusBar().setVisible(False)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        n_pixels_x = 1200
        n_pixels_y = 800

        pos_x = 2560 - n_pixels_x
        pos_y = 0

        window.setGeometry(pos_x, pos_y, n_pixels_x, n_pixels_y)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        width_ratios = [1, 1]

        self.gridspec = self.fig.add_gridspec(nrows=3, ncols=2,
                                              left=0.055, right=0.985,
                                              bottom=0.08, top=0.95,
                                              wspace=0.35,
                                              hspace=0.7,
                                              width_ratios=width_ratios,
                                              height_ratios=[1, 1, 1])

        ax_00 = self.fig.add_subplot(self.gridspec[0, 0])
        ax_10 = self.fig.add_subplot(self.gridspec[1, 0])

        ax_02 = self.fig.add_subplot(self.gridspec[0, 1])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_psi_abs_squared_1d = fig_psi_abs_squared_1d(ax_00, settings)
        self.fig_psi_re_im_1d = fig_psi_re_im_1d(ax_10, settings)

        self.fig_control_inputs = fig_control_inputs(ax_02, settings)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        plt.ion()
        
        plt.draw()
        plt.pause(0.001)
        # -----------------------------------------------------------------------------------------

    def update_data(self, psi, V):

        self.fig_psi_abs_squared_1d.update(psi, V)
        self.fig_psi_re_im_1d.update(psi, V)

    def redraw(self):

        plt.figure(self.fig_name)

        plt.draw()

        self.fig.canvas.start_event_loop(0.001)

    def export(self, filepath):

        plt.figure(self.fig_name)

        plt.draw()

        self.fig.canvas.start_event_loop(0.001)

        plt.savefig(filepath,
                    dpi=None,
                    facecolor='w',
                    edgecolor='w',
                    format='png',
                    transparent=False,
                    bbox_inches=None,
                    pad_inches=0,
                    metadata=None)
