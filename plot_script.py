from pylab import *
import matplotlib.pyplot as plt
import numpy
from mpl_toolkits.mplot3d import Axes3D
from data_processing import DataLoader
from plot_tools import plot3D_2

if __name__=='__main__':
    file = open(sys.argv[1], "r")
    key_y = dict()
    z_dist = dict()
    for line in file:
        tokens = line.strip().split("\t")
        if(len(tokens) != 5):
            continue
        z_dist["%s\t%s"%(tokens[1], tokens[2])] = float(tokens[4])
        key_y[int(tokens[2])] = 0
    file.close()
    key_y = list(map(str, sorted(list(key_y.keys()), reverse=False)))
    print(key_y)
    plot3D_2(DataLoader.Modulation_Types, key_y, z_dist, "%s-img.png" % sys.argv[1].split(".")[0])
