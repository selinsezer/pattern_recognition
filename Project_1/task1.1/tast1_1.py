import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import os

INPUT_PATH = os.path.dirname(os.getcwd()) + '/resources'
OUTPUT_PATH = os.path.dirname(os.getcwd()) + '/results'


def plotData2D(X, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # see what happens, if you uncomment the next line
    # axs.set_aspect('equal')
    
    # plot the data 
    axs.plot(X[0,:], X[1,:], 'ro', label='data')

    # set x and y limits of the plotting area
    xmin = X[0,:].min()
    xmax = X[0,:].max()
    axs.set_xlim(xmin-10, xmax+10)
    axs.set_ylim(-2, X[1,:].max()+10)

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper left', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


def is_outlier(point):
    if point[0] < 0 or point[1] < 0:
        return True
    else:
        return False


if __name__ == "__main__":
    #######################################################################
    # Using the first alternative in whExample.py for data reading
    #######################################################################
    # define type of data to be read and read data from file whose path is defined as data_path
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data_path = INPUT_PATH + '/whData.dat'
    data = np.loadtxt(data_path, dtype=dt, comments='#', delimiter=None)

    # read height, weight and gender information into 1D arrays if the point is not an outlier
    ws = np.array([d[0] for d in data if not is_outlier(d)])
    hs = np.array([d[1] for d in data if not is_outlier(d)])
    gs = np.array([d[2] for d in data if not is_outlier(d)])

    # now, plot weight vs. height by creating matrix X by stacking w and h
    X = np.vstack((ws, hs))
    plotData2D(X, OUTPUT_PATH + '/plotWH.pdf')

    # next, let's plot height vs. weight
    Z = np.vstack((hs,ws))
    plotData2D(Z, OUTPUT_PATH + '/plotHW.pdf')
