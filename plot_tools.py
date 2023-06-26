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

def plot3D_2(key_x, key_y, z_dist, file_name):
    fig = plt.figure(figsize=(15,15))
    ticksx = np.arange(len(key_x))
    ticksy = np.arange(len(key_y))
    zpos = numpy.zeros((len(key_y), len(key_x)))
    for i in range(len(key_x)):
        for j in range(len(key_y)):
            zpos[j, i] = z_dist["%s\t%s"%(key_x[i], key_y[j])]

    plt.xticks(ticksx, key_x, rotation=45)
    plt.yticks(ticksy, key_y)
    plt.xlabel("Modulation Type")
    plt.ylabel("SNR")

    # 绘制热力图
    im = plt.imshow(zpos, cmap='hot', interpolation='nearest')
    plt.colorbar(im,fraction=0.046, pad=0.04)
    plt.tight_layout()

    # 保存热力图为文件
    plt.savefig(file_name, dpi=72)
