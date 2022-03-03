import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import LogNorm

from . import utils

FONT_SIZE = 8


class Plottable(object):
    """Base PyOptica class implementing plotting functionality.

    1. Implementing subclasses:
    Each subclass must implement `PLOTTING_OPTIONS` as a class constant as a
    dict with key names referring to class properties. Example:

    >>> import pyoptica as po
    >>>
    >>> class PlottableExample(po.plotting.Plottable):
    >>>     PLOTTING_OPTIONS = dict(
    >>>          first=dict(x='x', y='y', title=''), second=dict(x='x', y='y', title='')
    >>>    )
    >>>    @property
    >>>    def first(self):
    >>>         return np.ones((10, 10))
    >>>
    >>>    @property
    >>>     def second(self):
    >>>         return np.ones((10, 10))
    >>>
    >>>    @property
    >>>     def x(self):
    >>>         return np.meshgrid(np.arange(10), np.arange(10))[0] * u.m
    >>>
    >>>    @property
    >>>     def y(self):
    >>>         return np.meshgrid(np.arange(10), np.arange(10))[1] * u.m

    All plotables parameters (properties!) can be checked by running:

    >>> plottable_example = PlottableExample()
    >>> plottable_example.plottables
    ['first', 'second']

    The easiest way to plot them is to use the default plotting options (the
    ones defined in `PLOTTING_OPTIONS`):

    >>> plottable_example.plot(first='default')

    Or if we want to have a figure with 2 sublots:

    >>> plottable_example.plot(first='default', second='default')

    In case we would like to modify the default options it can be done by
    providing plotting options:

    >>> plottable_example.plot(first=dict(title='My Title', log_scale=True))

    Supported options are:

    axis_unit
        The X,Y grid will be converted and plotted in the unit. Expected astropy.Quantity of type length.
    bar_ticks
        Ticks of the colorbar
    bar_ticks_labels
        Labels of the ticks of the colorbar
    cmap
        matplotlib colormap to be used. Must be str!
    colorbar
        Should colorbar explaining values be shown (bool)
    colorbar_title
        Title of the colorbar
    x
        Name of the object property to be shown on X-axis
    y
        Name of the object property to be shown on Y-axis
    vmin
        Minimum value shown in the plot (Z-axis)
    vmax
        Maximum value shown in the plot (Z-axis)
    log_scale
        Log scale (bool)
    title
        Title of the plot
    z_function
        Function to be applied before plotting.

    Lastly, global properties of the plot can be modified with `fig_options`.
    All options accepted by matplotlib.figure are accepted.

    >>> plottable_example.plot(first='default', fig_options=dict(dpi=130))
    """
    FONT_SIZE = 8
    DEFAULT_PLOTTING_OPTIONS = dict(
        colorbar_title='[arb. u.]', bar_ticks=None, bar_ticks_labels=None,
        vmin=None, vmax=None, cmap='inferno', colorbar=True, z_function=None,
        log_scale=None
    )
    PLOTTING_OPTIONS = dict()

    @property
    def plottables(self):
        """ Returns a list of of all variable names that can be plotted
        :rtype: list of strings
        """
        return list(self.PLOTTING_OPTIONS.keys())

    @classmethod
    def _plot_img(
            cls, fig, ax, z, axis_unit, bar_ticks, bar_ticks_labels, cmap,
            colorbar, colorbar_title, fontsize, title, x1, x2, y1, y2, vmax,
            vmin, log_scale):
        """A base function for plotting a 2D distribution in matplotlib.

        :param fig: Figure used for plotting
        :type fig: matplotlib.pyplot.figure
        :param ax: Axis for the plot
        :type ax: matplotlib.pyplot.axis
        :param z: array to be plotted
        :type z: np.array
        :param axis_unit: The X,Y grid will be converted and plotted in the unit
        :type axis_unit: astropy.Quantity of type length
        :param bar_ticks: Ticks of the colorbar
        :type bar_ticks: iterable
        :param bar_ticks_labels: Labels of the ticks of the colorbar
        :type bar_ticks: iterable
        :param cmap: colormap to be used
        :type cmap: matplotlib.colors.LinearSegmentedColormap
        :param colorbar: Should colorbar explaining plotted values be shown
        :type colorbar: bool
        :param colorbar_title: Title of the colorbar
        :type colorbar_title: str
        :param fontsize: Size of the font
        :type fontsize: int
        :param title: Title of the plot
        :type title: str
        :param x1: minimum value on OX-axis
        :type x1: float
        :param x2: maximum value on OX-axis
        :type x2: float
        :param y1: minimum value on OY-axis
        :type y1: float
        :param y2: maximum value on OY-axis
        :type y2: float
        :param vmin: minimum value on OZ-axis
        :type vmin: float
        :param vmax: maximum value on OZ-axis
        :type vmax: float
        :param log_scale: should logarithmic scale be used>
        :type log_scale: bool
        :return: created image
        :rtype: matplotlib.image.AxesImage
        """
        cmap = cm.get_cmap(cmap)
        cmap.set_bad('k')
        axis_title = axis_unit.to_string()
        extent = [i.to(axis_unit).value for i in [x1, x2, y1, y2]]
        if log_scale is True:
            norm = LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None
        im = ax.imshow(
            z, vmin=vmin, vmax=vmax, cmap=cmap, interpolation=None,
            extent=extent, norm=norm
        )
        ax.set_title(title)
        ax.set_xlabel(axis_title)
        ax.set_ylabel(axis_title)

        if colorbar is True:
            # The magic values make sure the colorbar looks good...
            # Found the on the internet.
            c_bar = fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax)
            c_bar.ax.set_title(colorbar_title, fontsize=fontsize)
            if bar_ticks is not None:
                c_bar.set_ticks(bar_ticks)
            if bar_ticks_labels is not None:
                c_bar.set_ticklabels(bar_ticks_labels)
            c_bar.ax.tick_params(labelsize=fontsize)

        set_font_of_plot(ax, fontsize)
        return im

    def plot(self, fig_options=None, **kwargs):
        """ A method to plot 2D property of the object.

        :param fig_options: Global parameters referring to the entire plot
        :type fig_options: dict
        :param kwargs: a dict representing properties to be plotted with
                correnspodning options
        :return: fig, [axes], [images]

        **Example**

        >>> import astropy.units as u
        >>> import pyoptica as po
        >>>
        >>> wavelength = 500 * u.nm
        >>> pixel_scale = 22 * u.um
        >>> npix = 1024
        >>> wf = po.Wavefront(wavelength, pixel_scale, npix)
        >>> _ = wf.plot(intensity='default', phase='default')
        >>> _ = wf.plot(intensity=dict(title='My title', vmax=0.3, vmin=0.1))
        >>> _ = wf.plot(fig_options=dict(dpi=130), phase=dict(cmap='bwr'))
        """
        self._check_input(kwargs)

        to_plot = [k for k in kwargs.keys() if k in self.plottables]
        plots_no = len(to_plot)
        if plots_no < 1:
            raise ValueError(
                'Please choose at least one from: '
                f'{", ".join(self.PLOTTING_OPTIONS.keys())}.'
            )
        if fig_options is None:
            fig_options = dict()
        if 'font_size' in fig_options:
            font_size = fig_options.pop('font_size')
        else:
            font_size = FONT_SIZE
        fig, axs = plt.subplots(1, plots_no, **fig_options)
        if plots_no == 1:
            axs = [axs]
        ims = []

        for w, ax in zip(to_plot, axs):
            plotting_options = self.DEFAULT_PLOTTING_OPTIONS.copy()
            plotting_options.update(self.PLOTTING_OPTIONS[w])
            if kwargs[w] != 'default':
                plotting_options.update(kwargs[w])
            z = self.__getattribute__(w)
            z_function = plotting_options.pop('z_function')
            if z_function is not None:
                if isinstance(z_function, str):
                    z = self.__getattribute__(z_function)()
                elif callable(z_function):
                    z = z_function(z)
            x = self.__getattribute__(plotting_options.pop('x'))
            y = self.__getattribute__(plotting_options.pop('y'))
            plotting_options['x1'] = x[0, 0]
            plotting_options['x2'] = x[-1, -1]
            plotting_options['y1'] = y[0, 0]
            plotting_options['y2'] = y[-1, -1]
            plotting_options['fontsize'] = font_size
            plotting_options['axis_unit'] = plotting_options.get('axis_unit',
                                                                 x.unit)
            im = self._plot_img(fig, ax, z, **plotting_options)
            ims.append(im)
        plt.tight_layout()
        return fig, axs, ims

    def _check_plotting_options(self, options):
        """Checks if provided plotting options are supported

        :param options: provided plotting options
        :type options: dict
        :raises: ValueError
        """
        supported = dict(
            axis_unit='The X,Y grid will be converted and plotted in the unit.'
                      ' Expected astropy.Quantity.',
            bar_ticks='Ticks of the colorbar.',
            bar_ticks_labels='Labels of the ticks of the colorbar.',
            cmap='matplotlib colormap to be used. Must be str!',
            colorbar='Should colorbar explaining values be shown (bool).',
            colorbar_title='Title of the colorbar.',
            x='Name of the object property to be shown on X-axis.',
            y='Name of the object property to be shown on Y-axis.',
            vmin='Minimum value shown in the plot (Z-axis).',
            vmax='Maximum value shown in the plot (Z-axis).',
            log_scale='Log scale (bool).',
            title='Title of the plot.',
            z_function='Function to be applied before plotting.'
        )
        msg = '\n'.join([
            f'\t{k}\n\t\t{v}' for k, v in sorted(supported.items())
        ])
        for opt in options:
            if opt not in supported:
                raise ValueError(
                    f"Option '{opt}' is not supported. Please choose one of "
                    f"the following: \n{msg}"
                )

    def _check_input(self, input_kwargs):
        """Verifies if the input (parameters) can be plotted by the object.

        :param input_kwargs: parameters to be plotted
        :type input_kwargs: dict
        :raises: ValueError
        """
        if len(input_kwargs) == 0:
            raise ValueError(
                "At least one parameters from the following must be chosen: "
                f"{self.plottables}. \n\tExample: \n"
                f"\t\t object_instance.plot({self.plottables[0]}='default')\n"
                f"\tor\n"
                f"\t\t object_instance.plot({self.plottables[0]}=dict("
                f"log_scale=True, title='My Title'))"

            )
        for key in input_kwargs:
            if key not in self.plottables:
                raise ValueError(
                    f"{key} is not recognized as a plottable property of "
                    f"{self.__class__.__name__}. Only {self.plottables} "
                    f"are supported. The parameters can be checked with "
                    f"`obj_instance.plottables`")
            if input_kwargs[key] == 'default':
                continue
            self._check_plotting_options(input_kwargs[key])


def plot_zernike(
        zernike_array,
        rho,
        theta,
        axis_unit=u.dimensionless_unscaled,
        colorbar=True,
        zmin=None,
        zmax=None,
        bar_ticks=None,
        bar_ticks_labels=None,
        title=None,
        fontsize=FONT_SIZE,
        subplot_layout=111,
        dpi=130,
        figsize=(5, 5),
        cmap=None,
        log_scale=False,
        **kwargs
):
    """Plots Zernike distribution in matplotlib.

    :param zernike_array: array representing a polynomial
    :type zernike_array: numpy.array
    :param rho: radial coordinates of the plane,
    :type rho: numpy.array
    :param theta: angle coordinate of the plane
    :type theta: numpy.array
    :param axis_unit: The X,Y grid will be converted and plotted in the given unit
    :type axis_unit: astropy.Quantity of type length
    :param colorbar: Should colorbar explaining plotted values be shown
    :type colorbar: bool
    :param zmin: Minimal value to be shown
    :type zmin: float
    :param zmax: Maximal value to be shown
    :type zmax: float
    :param bar_ticks: Ticks of the colorbar
    :type bar_ticks: iterable
    :param bar_ticks_labels: Labels of the ticks of the colorbar
    :type bar_ticks: iterable
    :param title: Title of the plot
    :type title: str
    :param fontsize: Size of the font
    :type fontsize: int
    :param figsize: Size of output figure in inches
    :type figsize: (int, int)
    :param dpi: dots per inch of the resulting figure
    :type dpi: int
    :param cmap: colormap to be used
    :type cmap: matplotlib.colors.LinearSegmentedColormap
    :param log_scale: should logarithmic scale be used?
    :type: bool
    :param kwargs: all kwargs passed to matplotlib.pyplot.imshow
    :type kwargs: object
    :return: plot of asked values
    :rtype: matplotlib.figure, matplotlib.axis

    **Example**

    >>> import astropy.units as u
    >>> import numpy as np
    >>> import pyoptica as po
    >>>
    >>> npix = 101
    >>> pixel_scale = 0.5 * u.m
    >>> x, y = po.utils.mesh_grid(npix, pixel_scale)
    >>> r_max = .48 * npix * pixel_scale
    >>> r, theta = po.utils.cart2pol(x, y)
    >>> r = r / r_max
    >>>
    >>> m, n = -1, 3
    >>> z = po.zernike((m, n), 'mn', r, theta, fill_value=np.nan)
    >>> _ = po.plotting.plot_zernike(z, r, theta, title=f'Zernike m={m}, n={n}')

    """
    if title is None:
        title = 'Zernike Distribution'
    x, y = utils.pol2cart(rho, theta)
    x1, x2 = x[0, 0], x[-1, -1]
    y1, y2 = y[0, 0], y[-1, -1]
    if cmap is None:
        cmap = cm.plasma
    if zmin is None and zmax is None:
        zmax = np.nanmax(zernike_array)
        zmin = np.nanmin(zernike_array)
        zmax = max(abs(zmax), abs(zmin))
        zmin = -zmax
    colorbar_title = '[arb. u.]'
    return plot_img(
        zernike_array, axis_unit, bar_ticks, bar_ticks_labels, cmap, colorbar,
        colorbar_title, dpi, figsize, fontsize, subplot_layout, title, x1, x2,
        y1, y2, zmax, zmin, log_scale, **kwargs)


def plot_img(
        z, axis_unit, bar_ticks, bar_ticks_labels, cmap, colorbar,
        colorbar_title, dpi, figsize, fontsize, subplot_layout, title, x1, x2,
        y1, y2, zmax, zmin, log_scale, **kwargs):
    """A base function for plotting a 2D distribution in matplotlib.
    :param z: array to be plotted
    :type z: np.array
    :param axis_unit: The X,Y grid will be converted and plotted in the unit
    :type axis_unit: astropy.Quantity of type length
    :param bar_ticks: Ticks of the colorbar
    :type bar_ticks: iterable
    :param bar_ticks_labels: Labels of the ticks of the colorbar
    :type bar_ticks: iterable
    :param cmap: colormap to be used
    :type cmap: matplotlib.colors.LinearSegmentedColormap
    :param colorbar: Should colorbar explaining plotted values be shown
    :type colorbar: bool
    :param colorbar_title: Title of the colorbar
    :type colorbar_title: str
    :param dpi: dots per inch of the resulting figure
    :type dpi: int
    :param figsize: Size of output figure in inches
    :type figsize: (int, int)
    :param fontsize: Size of the font
    :type fontsize: int
    :param subplot_layout: layout in case of many plots
    :type subplot_layout: int
    :param title: Title of the plot
    :type title: str
    :param x1: minimum value on OX-axis
    :type x1: float
    :param x2: maximum value on OX-axis
    :type x2: float
    :param y1: minimum value on OY-axis
    :type y1: float
    :param y2: maximum value on OY-axis
    :type y2: float
    :param zmin: minimum value on OZ-axis
    :type zmin: float
    :param zmax: maximum value on OZ-axis
    :type zmax: float
    :param log_scale: should logarithmic scale be used>
    :type log_scale: bool
    :param kwargs: all kwargs passed to matplotlib.pyplot.imshow
    :type kwargs: object
    :return: plot of asked values
    :rtype: matplotlib.figure, matplotlib.axis

    """
    axis_title = axis_unit.to_string()
    fig = plt.figure(dpi=dpi, figsize=figsize)
    ax = fig.add_subplot(subplot_layout)
    extent = [i.to(axis_unit).value for i in [x1, x2, y1, y2]]
    if log_scale is True:
        norm = LogNorm(vmin=zmin, vmax=zmax)
    else:
        norm = None
    plot = ax.imshow(
        z, vmin=zmin, vmax=zmax, cmap=cmap, interpolation=None, extent=extent,
        norm=norm, **kwargs
    )
    ax.set_title(title)
    ax.set_xlabel(axis_title)
    ax.set_ylabel(axis_title)
    if colorbar is True:
        # The magic values make sure the colorbar looks good...
        # Found the on the internet.
        c_bar = plt.colorbar(plot, fraction=0.046, pad=0.04)
        c_bar.ax.set_title(colorbar_title, fontsize=fontsize)
        if bar_ticks is not None:
            c_bar.set_ticks(bar_ticks)
        if bar_ticks_labels is not None:
            c_bar.set_ticklabels(bar_ticks_labels)
        c_bar.ax.tick_params(labelsize=fontsize)

    set_font_of_plot(ax, fontsize)
    plt.tight_layout()
    return fig, plot, ax


def set_font_of_plot(plot, font_size=FONT_SIZE):
    """Sets font of a plot
    :param plot: matplotlib.axes
        Plot on which font should be adjusted
    :param font_size: int

    """
    plot.title.set_fontsize(font_size)
    labels = [plot.xaxis.label, plot.yaxis.label] + \
             plot.get_xticklabels() + plot.get_yticklabels()
    for item in labels:
        item.set_fontsize(font_size)
