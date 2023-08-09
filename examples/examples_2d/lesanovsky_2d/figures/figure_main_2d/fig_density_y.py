import numpy as np

from qsolve.figures.style import colors


class fig_density_y(object):

    def __init__(self, ax, settings):

        self.hbar = settings.hbar

        self.m_atom = settings.m_atom

        # -----------------------------------------------------------------------------------------
        self.line_density_y_x1, = ax.plot(
            settings.y, np.zeros_like(settings.y), linewidth=1, linestyle='-',
            color=colors.wet_asphalt, label=r'$\rho(x_1, y)$')

        self.line_density_y_x2, = ax.plot(
            settings.y, np.zeros_like(settings.y), linewidth=1, linestyle='--',
            color=colors.wet_asphalt, label=r'$\rho(x_2, y)$')

        ax.set_xlim(settings.y_min, settings.y_max)

        density_min = settings.density_min
        density_max = settings.density_max

        ax.set_ylim(density_min - 0.1 * (density_max - density_min), density_max + 0.1 * (density_max - density_min))

        ax.set_yticks(np.linspace(density_min, density_max, num=3))

        ax.set_xlabel(settings.label_y)
        
        ax.set_xticks(settings.y_ticks)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)
        
        # ax.set_ylabel(settings.label_density)
        ax.set_ylabel(r'$\mathrm{m}^{-2}$')

        ax.set_title(r'$x=0$', fontsize=settings.fontsize_titles)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        ax2 = ax.twinx()

        self.line_V_y_x1, = ax2.plot(
            settings.y, np.zeros_like(settings.y), linewidth=settings.linewidth_V, linestyle='-',
            color=settings.linecolor_V, label=r'$V(x_1, y)$')

        ax2.set_xlim(settings.y_min, settings.y_max)

        V_min = settings.V_min
        V_max = settings.V_max

        ax2.set_ylim(V_min - 0.1 * (V_max - V_min), V_max + 0.1 * (V_max - V_min))

        ax2.set_yticks(np.linspace(V_min, V_max, num=3))

        ax2.set_ylabel(settings.label_V)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2,
                   loc='upper right', bbox_to_anchor=(1.0, 1.25), fancybox=True, framealpha=1, ncol=1)
        # -----------------------------------------------------------------------------------------

    def update(self, density_y_x1, density_y_x2, V_y_x1, V_y_x2):

        scaling_V = self.hbar * 2 * np.pi * 1000

        self.line_density_y_x1.set_ydata(density_y_x1)
        self.line_density_y_x2.set_ydata(density_y_x2)

        self.line_V_y_x1.set_ydata(V_y_x1 / scaling_V)
