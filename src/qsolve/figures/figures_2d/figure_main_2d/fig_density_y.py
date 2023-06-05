from numpy import zeros_like

from numpy import pi

from qsolve.figures.style import colors


class fig_density_y(object):

    def __init__(self, ax, settings):

        self.hbar = settings.hbar

        self.m_atom = settings.m_atom

        # -----------------------------------------------------------------------------------------
        self.line_density_y, = ax.plot(settings.y, zeros_like(settings.y), linewidth=1, linestyle='-',
                                          color=colors.wet_asphalt, label=r'$\rho$')

        ax.set_xlim(settings.y_min, settings.y_max)
        
        ax.set_ylim(settings.density_min, settings.density_max)

        ax.set_xlabel(settings.label_y)
        
        ax.set_xticks(settings.y_ticks)
        
        ax.grid(visible=True, which='major', color=settings.color_gridlines_major, linestyle='-', linewidth=0.5)
        
        ax.set_ylabel(settings.label_density)

        ax.set_title(r'$x=0$', fontsize=settings.fontsize_titles)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        ax2 = ax.twinx()

        self.line_V_y, = ax2.plot(settings.y, zeros_like(settings.y), linewidth=settings.linewidth_V, linestyle='-',
                                     color=settings.linecolor_V, label=r'$V$')

        ax2.set_xlim(settings.y_min, settings.y_max)
        ax2.set_ylim(settings.V_min, settings.V_max)

        ax2.set_ylabel(settings.label_V)
        # -----------------------------------------------------------------------------------------

        # -----------------------------------------------------------------------------------------
        lines, labels = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2,
                   loc='upper right', bbox_to_anchor=(1.0, 1.25), fancybox=True, framealpha=1, ncol=1)
        # -----------------------------------------------------------------------------------------

    def update(self, density_y, V_y):

        scaling_V = self.hbar * 2 * pi * 1000

        V_y = V_y / scaling_V

        self.line_density_y.set_ydata(density_y)

        self.line_V_y.set_ydata(V_y)
