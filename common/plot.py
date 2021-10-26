from . import utils
from .log import log, LogLevel

if not utils.display() and not utils.notebook():
    log('[Error] DISPLAY not found, plot not available!', LogLevel.ERROR)


import io
import matplotlib
from matplotlib import pyplot
import numpy


# matplotlib.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# matplotlib.rc('font',**{'family':'serif','serif':['Palatino']})
# https://matplotlib.org/users/usetex.html
#matplotlib.rc('text', usetex=True)
matplotlib.rc('font', family='serif')
# https://matplotlib.org/users/customizing.html
matplotlib.rcParams['lines.linewidth'] = 1
#matplotlib.rcParams['figure.figsize'] = (30, 10)


def as_array(figure, close=True):
    """
    Render figure for numpy array.

    :param figure: figure
    :type figure: matplotlib.pyplot.figure
    :return: array
    :rtype: numpy.ndarray
    """

    buf = io.BytesIO()
    figure.savefig(buf, format='png')
    buf.seek(0)
    if close:
        figure.clear()
    return numpy.array(buf)


def rgb_html(rgb):
    rgb = (int(rgb[0]), int(rgb[1]), int(rgb[2]))
    return '#%02x%02x%02x' % rgb


def darken(rgb, alpha=0.9):
    rgb[0] = alpha*rgb[0]
    rgb[1] = alpha*rgb[1]
    rgb[2] = alpha*rgb[2]
    return rgb


def lighten(rgb, alpha=0.9):
    rgb[0] = 1 - (alpha * (1 - rgb[0]))
    rgb[1] = 1 - (alpha * (1 - rgb[1]))
    rgb[2] = 1 - (alpha * (1 - rgb[2]))
    return rgb


# from http://colorbrewer2.org/#type=qualitative&scheme=Paired&n=12
color_brewer_ = numpy.array([
    [166, 206, 227],
    [31, 120, 180],
    [251, 154, 153],
    [178, 223, 138],
    [51, 160, 44],
    [227, 26, 28],
    [253, 191, 111],
    [255, 127, 0],
    [202, 178, 214],
    [106, 61, 154],
    [245, 245, 145],
    [177, 89, 40],
    # 10
    lighten([166, 206, 227]),
    lighten([31, 120, 180]),
    lighten([251, 154, 153]),
    lighten([178, 223, 138]),
    lighten([51, 160, 44]),
    lighten([227, 26, 28]),
    lighten([253, 191, 111]),
    lighten([255, 127, 0]),
    lighten([202, 178, 214]),
    lighten([106, 61, 154]),
    lighten([255, 255, 153]),
    lighten([177, 89, 40]),
    # 30
    lighten([166, 206, 227], 0.7),
    lighten([31, 120, 180], 0.7),
    lighten([251, 154, 153], 0.7),
    lighten([178, 223, 138], 0.7),
    lighten([51, 160, 44], 0.7),
    lighten([227, 26, 28], 0.7),
    lighten([253, 191, 111], 0.7),
    lighten([255, 127, 0], 0.7),
    lighten([202, 178, 214], 0.7),
    lighten([106, 61, 154], 0.7),
    lighten([255, 255, 153], 0.7),
    lighten([177, 89, 40], 0.7),
    # 50
    lighten([166, 206, 227], 0.5),
    lighten([31, 120, 180], 0.5),
    lighten([251, 154, 153], 0.5),
    lighten([178, 223, 138], 0.5),
    lighten([51, 160, 44], 0.5),
    lighten([227, 26, 28], 0.5),
    lighten([253, 191, 111], 0.5),
    lighten([255, 127, 0], 0.5),
    lighten([202, 178, 214], 0.5),
    lighten([106, 61, 154], 0.5),
    lighten([255, 255, 153], 0.5),
    lighten([177, 89, 40], 0.5),
    # 90
    darken([166, 206, 227], 0.9),
    darken([31, 120, 180], 0.9),
    darken([251, 154, 153], 0.9),
    darken([178, 223, 138], 0.9),
    darken([51, 160, 44], 0.9),
    darken([227, 26, 28], 0.9),
    darken([253, 191, 111], 0.9),
    darken([255, 127, 0], 0.9),
    darken([202, 178, 214], 0.9),
    darken([106, 61, 154], 0.9),
    darken([255, 255, 153], 0.9),
    darken([177, 89, 40], 0.9),
    # 70
    darken([166, 206, 227], 0.7),
    darken([31, 120, 180], 0.7),
    darken([251, 154, 153], 0.7),
    darken([178, 223, 138], 0.7),
    darken([51, 160, 44], 0.7),
    darken([227, 26, 28], 0.7),
    darken([253, 191, 111], 0.7),
    darken([255, 127, 0], 0.7),
    darken([202, 178, 214], 0.7),
    darken([106, 61, 154], 0.7),
    darken([255, 255, 153], 0.7),
    darken([177, 89, 40], 0.7),
    # 50
    darken([166, 206, 227], 0.5),
    darken([31, 120, 180], 0.5),
    darken([251, 154, 153], 0.5),
    darken([178, 223, 138], 0.5),
    darken([51, 160, 44], 0.5),
    darken([227, 26, 28], 0.5),
    darken([253, 191, 111], 0.5),
    darken([255, 127, 0], 0.5),
    darken([202, 178, 214], 0.5),
    darken([106, 61, 154], 0.5),
    darken([255, 255, 153], 0.5),
    darken([177, 89, 40], 0.5),

], dtype=float)
color_brewer = color_brewer_/255.

marker_brewer = [
    'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o', 'o',
    'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P', 'P',
    '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*', '*',
    'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd', 'd',
    's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's', 's',
    'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p', 'p',
    'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X',
]


def label(ax=pyplot.gca(), legend=False, **kwargs):
    """
    Label axes, title etc.

    :param legend: whether to add and return a legend
    :type legend: bool
    :return: legend
    :rtype: None or matplotlib.legend.Legend
    """

    title = kwargs.get('title', None)
    xlabel = kwargs.get('xlabel', None)
    ylabel = kwargs.get('ylabel', None)
    xscale = kwargs.get('xscale', None)
    yscale = kwargs.get('yscale', None)

    if title is not None:
        ax.set_title(title)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if xscale is not None:
        if xscale == 'symlog':
            linthreshx = kwargs.get('linthreshx', 10 ** -10)
            ax.set_xscale(xscale, linthreshx=linthreshx)
        else:
            ax.set_xscale(xscale)
    if yscale is not None:
        if yscale == 'symlog':
            linthreshy = kwargs.get('linthreshy', 10 ** -10)
            ax.set_yscale(yscale, linthreshy=linthreshy)
        else:
            ax.set_yscale(yscale)

    xmax = kwargs.get('xmax', None)
    xmin = kwargs.get('xmin', None)
    ymax = kwargs.get('ymax', None)
    ymin = kwargs.get('ymin', None)

    if xmax is not None:
        ax.set_xbound(upper=xmax)
    if xmin is not None:
        ax.set_xbound(lower=xmin)
    if ymax is not None:
        ax.set_ybound(upper=ymax)
    if ymin is not None:
        ax.set_ybound(lower=ymin)

    ax.figure.set_size_inches(kwargs.get('w', 6), kwargs.get('h', 6))

    # This is fixed stuff.
    ax.grid(b=True, which='major', color=(0.5, 0.5, 0.5), linestyle='-')
    ax.grid(b=True, which='minor', color=(0.75, 0.75, 0.75), linestyle='--')

    legend_loc = kwargs.get('legend_loc', 'upper left')
    legend_anchor = kwargs.get('legend_anchor', (1, 1.05))

    if legend:
        legend_ = ax.legend(loc=legend_loc, bbox_to_anchor=legend_anchor)
        legend_.get_frame().set_alpha(None)
        legend_.get_frame().set_facecolor((1, 1, 1, 0.5))
        return legend_


def line(x, y, labels=None, colors=None, markers=None, ax=pyplot.gca(), **kwargs):
    """
    Line plot.

    :param data: vector of data to plot
    :type data: numpy.ndarray
    :param labels: optional labels
    :type labels: [str]
    """

    if isinstance(x, numpy.ndarray):
        assert len(x.shape) == 1 or len(x.shape) == 2, ' only one- or two-dimensional data can be line-plotted'
        assert len(y.shape) == 1 or len(y.shape) == 2, ' only one- or two-dimensional data can be line-plotted'

        if len(x.shape) == 1:
            x = x.reshape((1, -1))
        if len(y.shape) == 1:
            y = y.reshape((1, -1))

        num_labels = x.shape[0]
    elif isinstance(x, list):
        assert isinstance(y, list)
        assert len(x) == len(y)
        for i in range(len(x)):
            assert x[i].shape[0] == y[i].shape[0]
        num_labels = len(x)
    else:
        assert False

    has_labels = (labels is not None)
    if not has_labels:
        labels = [None] * num_labels
    if len(labels) > len(color_brewer):
        log('using black plots', LogLevel.WARNING)
        colors = numpy.array([[0, 0, 0]])
        colors = numpy.repeat(colors, len(labels), axis=0)
        markers = [None]*len(labels)
    else:
        if colors is None:
            colors = color_brewer
        if markers is None:
            markers = marker_brewer

    for i in range(num_labels):
        ax.plot(x[i], y[i], color=tuple(colors[i]), label=labels[i], marker=markers[i], linewidth=kwargs.get('linewidth', 1), markersize=kwargs.get('markersize', 10))
    ax.legend()
    label(ax, legend=has_labels, **kwargs)


def scatter(x, y, c=None, labels=None, ax=pyplot.gca(), **kwargs):
    """
    Scatter plot or 2D data.

    :param x: x data
    :type x: numpy.ndarray
    :param y: y data
    :type y: numpy.ndarray
    :param c: labels as N x 1
    :type c: numpy.ndarray
    :param labels: label names
    :type labels: [str]
    """

    assert len(x.shape) == len(y.shape), 'only one dimensional data arrays supported'
    assert x.shape[0] == y.shape[0], 'only two-dimensional data can be scatter-plotted'
    assert c is None or x.shape[0] == c.shape[0], 'data and labels need to have same number of rows'
    if c is not None:
        assert labels is not None, 'if classes are given, labels need also to be given'

    if c is not None:
        if len(c.shape) > 1:
            c = numpy.squeeze(c)
    elif c is None:
        c = numpy.zeros((x.shape[0]))
        labels = [0]
    c = c.astype(int)  # Important for indexing

    unique_labels = numpy.unique(c)
    assert unique_labels.shape[0] <= len(color_brewer), 'currently a maxmimum of 12 different labels are supported'
    # assert unique_labels.shape[0] == len(labels), 'labels do not match given classes'
    assert numpy.min(unique_labels) >= 0 and numpy.max(unique_labels) < len(labels), 'classes contain elements not in labels'

    for i in range(unique_labels.shape[0]):
        marker = kwargs.get('marker', marker_brewer[i])
        ax.scatter(x[c == unique_labels[i]], y[c == unique_labels[i]],
                       c=numpy.repeat(numpy.expand_dims(color_brewer[i], 0), x[c == unique_labels[i]].shape[0], axis=0),
                       marker=marker, s=kwargs.get('s', 45),
                       edgecolor='black', linewidth=kwargs.get('linewidth', 0.5), label=labels[unique_labels[i]])
    has_colors = (c is not None)
    label(ax, legend=has_colors, **kwargs)
