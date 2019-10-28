import numpy as np


def make_meshgrid(x, y, h=.1):
    """
    Makes a meshgrid object from x and y.
    :param x: Numpy array
    :param y: Numpy array
    :param h: Step size for meshgrid values
    :return: tuple containing x and y for meshgrid
    """
    x_min = min(x) - 1
    x_max = max(x) + 1
    y_min = min(y) - 1
    y_max = max(y) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_contours(clf, ax, xx, yy, **params):
    """Plot the decision boundaries for a classifier using a Contour plot.

    Parameters
    ----------
    ax: matplotlib axes object
    clf: a classifier
    xx: meshgrid ndarray
    yy: meshgrid ndarray
    params: dictionary of params to pass to contourf, optional
    """
    Z = clf.decision_function(np.c_[(xx.ravel(), yy.ravel())])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


def plot_2d_decision_boundary(clf, ax, x, support_vectors=False):
    x_min = min(x)
    x_max = max(x)

    out = ax.plot(
        [x_min, x_max],
        [clf.decision_boundary(x_min), clf.decision_boundary(x_max)])
    if support_vectors:
        ax.plot(
            [x_min, x_max],
            [clf.decision_boundary(x_min, c=1), clf.decision_boundary(x_max, c=1)])
        out = ax.plot(
            [x_min, x_max],
            [clf.decision_boundary(x_min, c=-1), clf.decision_boundary(x_max, c=-1)])

    return out


