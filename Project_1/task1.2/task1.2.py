from scipy.stats import norm
import matplotlib.pyplot as plt
import numpy as np
import os
from math import ceil, floor

INPUT_PATH = os.path.dirname(os.getcwd()) + '/resources'
OUTPUT_PATH = os.path.dirname(os.getcwd()) + '/results'


def plotData(data, x, fitted_pdf, filename=None):
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    # plot the normal
    axs.plot(x, fitted_pdf, color='orange', label='normal')

    # plot the data points
    axs.scatter(data, [0] * len(data), alpha=0.3, color='blue', label="data")

    # insert a legend in the plot (using label)
    plt.legend(fancybox=True)

    # limit the x and y axis
    axs.set_xlim(floor(min(x)/10)*10, ceil(max(x)/10)*10)
    axs.set_ylim((floor(min(fitted_pdf)*100))/100, (ceil(max(fitted_pdf)*100))/100)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


if __name__ == "__main__":
    # define path of the data file, type of data to be read and read data from file
    data_path = INPUT_PATH + '/whData.dat'
    dt = np.dtype([('w', np.float), ('h', np.float), ('g', np.str_, 1)])
    data = np.loadtxt(data_path, dtype=dt, comments='#', delimiter=None)

    # read height information into 1D array while eliminating the outliers and sort the array
    heights = sorted(np.array([d[1] for d in data if d[1] > 0]))

    # calculate the mean and the standard deviation
    mean, standard_deviation = norm.fit(heights)

    # create a range where the probability density will be defined
    x = np.linspace(min(heights)-20, max(heights)+20, 100)

    # create the probability density function using the parameters we obtained above
    fitted_pdf = norm.pdf(x, loc=mean, scale=standard_deviation)

    # define the output path and plot the data
    figure_path = OUTPUT_PATH + '/task1_2.pdf'
    plotData(heights, x, fitted_pdf, figure_path)
