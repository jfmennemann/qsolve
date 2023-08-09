import numpy as np

from qsolve.figures.style import colors


class fig_density_y(object):

    def __init__(self, ax, settings):

        self.hbar = settings.hbar
        self.m_atom = settings.m_atom

        self.line_density_y, = ax.plot(settings.y, np.zeros_like(settings.y),
                                       linewidth=1, linestyle='-', color=colors.wet_asphalt)

        ax.set_xlim(settings.y_min, settings.y_max)

        density_min = settings.density_min
        density_max = settings.density_max

        ax.set_ylim(density_min - 0.1 * (density_max - density_min), density_max + 0.1 * (density_max - density_min))

        ax.set_yticks(np.linspace(density_min, density_max, num=3))

        ax.set_xlabel(settings.label_y)
        
        ax.set_xticks(settings.y_ticks)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)
        
        ax.set_ylabel(settings.label_density)
        
        ax.set_anchor('W')

        # -----------------------------------------------------------------------------------------
        ax2 = ax.twinx()
        
        self.line_V_y, = ax2.plot(settings.y, np.zeros_like(settings.y), linewidth=1, linestyle='-', color=colors.alizarin)

        ax2.set_xlim(settings.y_min, settings.y_max)

        V_min = settings.V_min
        V_max = settings.V_max

        ax2.set_ylim(V_min - 0.1 * (V_max - V_min), V_max + 0.1 * (V_max - V_min))

        ax2.set_yticks(np.linspace(V_min, V_max, num=3))

        ax2.set_ylabel(settings.label_V)
        # -----------------------------------------------------------------------------------------

    def update(self, density_y, V_y):

        scaling_V = self.hbar * 2 * np.pi * 1000
        
        V_y = V_y / scaling_V

        self.line_density_y.set_ydata(density_y)

        self.line_V_y.set_ydata(V_y)
