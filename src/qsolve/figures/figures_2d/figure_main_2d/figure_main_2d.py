import matplotlib.pyplot as plt

# from PyQt5 import QtWidgets

from scipy import constants

import numpy as np

from .fig_density import fig_density

from .fig_density_x import fig_density_x
from .fig_density_y import fig_density_y

from .fig_real_part import fig_real_part

from .fig_real_part_x import fig_real_part_x
from .fig_real_part_y import fig_real_part_y

from .fig_control_inputs import fig_control_inputs

from qsolve.figures.style import colors


class FigureMain2D(object):

    def __init__(self, x, y, times, params):

        hbar = constants.hbar

        m_atom = params['m_atom']

        density_min = -0.2 * params["density_max"]
        density_max = +1.2 * params["density_max"]

        V_min = params['V_min']
        V_max = params['V_max']

        x = x / 1e-6
        y = y / 1e-6

        times = times / 1e-3

        Jx = x.shape[0]
        Jy = y.shape[0]

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        x_min = x[0]
        y_min = y[0]

        x_max = x_min + Jx * dx
        y_max = y_min + Jy * dy
        
        t_min = times[0]
        t_max = times[-1]

        Jx = x.size
        Jy = y.size

        Lx = Jx * dx
        Ly = Jy * dy

        x_ticks = params['x_ticks']
        y_ticks = params['y_ticks']

        t_ticks_major = params['t_ticks']

        # -----------------------------------------------------------------------------------------
        t_ticks_minor = 0.5 * (t_ticks_major[0:-1] + t_ticks_major[1:])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        settings = type('', (), {})()

        settings.hbar = hbar
        settings.m_atom = m_atom

        settings.density_min = density_min
        settings.density_max = density_max

        settings.real_part_min = -1.2 * np.sqrt(settings.density_max)
        settings.real_part_max = +1.2 * np.sqrt(settings.density_max)

        settings.V_min = V_min
        settings.V_max = V_max

        # settings.indices_y_restr = indices_y_restr

        settings.x = x
        settings.y = y

        settings.Jx = Jx
        settings.Jy = Jy

        settings.x_ticks = x_ticks
        settings.y_ticks = y_ticks

        settings.x_min = x_min
        settings.x_max = x_max

        settings.y_min = y_min
        settings.y_max = y_max

        settings.times = times

        settings.t_min = t_min
        settings.t_max = t_max

        settings.t_ticks_major = t_ticks_major
        settings.t_ticks_minor = t_ticks_minor

        # settings.label_V = r'$V \;\, \mathrm{in} \;\, h \times \mathrm{kHz}$'
        settings.label_V = r'$h \times \mathrm{kHz}$'
        settings.linecolor_V = colors.alizarin
        settings.linewidth_V = 1.1

        # settings.label_density = r'$\mathrm{density} \;\, \mathrm{in} \;\, \mathrm{m}^{-2}$'
        settings.label_density = r'$\mathrm{m}^{-2}$'

        settings.label_x = r'$x \;\, \mathrm{in} \;\, \mu \mathrm{m}$'
        settings.label_y = r'$y \;\, \mathrm{in} \;\, \mu \mathrm{m}$'

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

        self.fig = plt.figure(self.fig_name, figsize=(14, 8), facecolor="white")
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        if Ly > Lx:

            width_ratios = [1.25, 1, 2]

        elif Ly < Lx:

            width_ratios = [1, 1.25, 2]

        else:

            width_ratios = [1, 1, 2]

        self.gridspec = self.fig.add_gridspec(nrows=4, ncols=3,
                                              left=0.055, right=0.985,
                                              bottom=0.08, top=0.95,
                                              wspace=0.35,
                                              hspace=0.7,
                                              width_ratios=width_ratios,
                                              height_ratios=[1, 1, 1, 1])

        ax_00 = self.fig.add_subplot(self.gridspec[0, 0])
        ax_10 = self.fig.add_subplot(self.gridspec[1, 0])
        ax_20 = self.fig.add_subplot(self.gridspec[2, 0])
        ax_30 = self.fig.add_subplot(self.gridspec[3, 0])

        ax_11 = self.fig.add_subplot(self.gridspec[1, 1])
        ax_31 = self.fig.add_subplot(self.gridspec[3, 1])

        ax_02 = self.fig.add_subplot(self.gridspec[0, 2])
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        self.fig_density = fig_density(ax_00, settings)

        self.fig_density_y = fig_density_y(ax_10, settings)

        self.fig_real_part = fig_real_part(ax_20, settings)

        self.fig_real_part_y = fig_real_part_y(ax_30, settings)


        self.fig_real_part_x = fig_real_part_x(ax_31, settings)


        self.fig_density_x = fig_density_x(ax_11, settings)



        self.fig_control_inputs = fig_control_inputs(ax_02, settings)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        plt.ion()
        
        plt.draw()
        plt.pause(0.001)
        # -----------------------------------------------------------------------------------------

    def update_data(self, data):

        self.fig_density.update(data.density)

        self.fig_density_x.update(data.density_x, data.V_x)
        self.fig_density_y.update(data.density_y, data.V_y)

        self.fig_real_part.update(data.psi)

        self.fig_real_part_x.update(data.real_part_x, data.imag_part_x, data.V_x)

        self.fig_real_part_y.update(data.real_part_y, data.imag_part_y, data.V_y)

    def redraw(self):

        # plt.figure(self.fig_name)
        #
        # plt.draw()
        #
        # self.fig.canvas.start_event_loop(0.001)

        # -----------------------------------------------------------------------------------------
        # drawing updated values
        self.fig.canvas.draw()

        # This will run the GUI event
        # loop until all UI events
        # currently waiting have been processed
        self.fig.canvas.flush_events()

        # time.sleep(0.1)
        # -----------------------------------------------------------------------------------------

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
