import numpy as np

from matplotlib import cm


class FigPotential2D(object):

    def __init__(self, ax, V, settings):

        Jx = settings.Jx
        Jy = settings.Jy

        x = settings.x
        y = settings.y

        # Y, X = np.meshgrid(x, y, indexing='ij')

        ax.set_xlabel(settings.label_y)
        ax.set_ylabel(settings.label_x)

        ax.set_xticks(settings.y_ticks)
        ax.set_yticks(settings.x_ticks)

        left = settings.y_min
        right = settings.y_max

        bottom = settings.x_min
        top = settings.x_max

        ax.imshow(
            (V / np.max(np.abs(V)))**0.5,
            extent=[left, right, bottom, top],
            cmap=cm.turbo,
            aspect='auto',
            interpolation='bilinear',
            vmin=0,
            vmax=1,
            origin='lower')

        # alizarin_rgb = hex2rgb(alizarin.lstrip('#'))

        # from matplotlib import ticker, cm

        # cs = ax.contourf(X, Y, V, locator=ticker.LogLocator(), cmap=cm.Greys)
        # cs = ax.contourf(X, Y, V, locator=ticker.LogLocator(), cmap=cm.hot)
        # cs = ax.contourf(X, Y, V, levels=[0, 0.1], cmap=cm.tab20c)

        ax.set_title(r'$V$ (scaled)', fontsize=settings.fontsize_titles)
