from pylab import *
import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D


def plot3D(dist_lists, dist_name, dist_center, dist_width, title, file_name):
    fig = plt.figure(figsize=(20,20))
    ax = fig.add_subplot(projection="3d")
    ax.set_xlabel('Value')
    ax.set_ylabel('Modulation Type')
    ax.set_zlabel('Probability')
    ax.set_title('Distributions of Statistics of %s' % title)

    cmap = plt.cm.inferno
    ncol = numpy.shape(dist_lists)[0]

    ticksy = np.arange(0.5, ncol, 1)
    plt.yticks(ticksy, dist_name)

    for i, dist in enumerate(dist_lists): 
        ax.bar(left=dist_center, height=dist, width=dist_width, zs=i, zdir="y", color=cmap(i/ncol), alpha=0.50, edgecolor="grey", linewidth=0.3)

    fig.savefig(file_name, dpi=72)
