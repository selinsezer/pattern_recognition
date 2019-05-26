import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import inv

if __name__ == "__main__":
    # read data as 2D array of data type 'object'
    data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

    # read height and weight data into 2D array (i.e. into a matrix)
    h_w_array = data[:, 0:2].astype(np.float)

    # create a mask to remove the out liar
    outlier_mask = (h_w_array[:, 0] == -1)

    x_heights_to_be_estimated = h_w_array[outlier_mask][:, 1]

    Y = h_w_array[:, 0]

    N = 5
    # generate design matrix
    X = np.vander(h_w_array[:, 1], N + 1)

    # generate sigma0 and sigma
    sigma0Square = 3.0
    sigmaSquare = 1000.0  # this number can be changed

    # estimate W with Least Square Method
    W_LSE = np.linalg.lstsq(X, Y, rcond=None)[0]

    # estimate W with Bayesian regression
    n, m = X.shape
    I = np.identity(m)
    W_Bayesian = np.dot(np.dot(inv(np.dot(X.T, X) + (sigmaSquare / sigma0Square) * I), X.T), Y)

    # plot
    # set up and low bound
    min_x, max_x = min(X[:, 4]), max(X[:, 4])
    min_y, max_y = min(Y), max(Y)
    ax_x_min, ax_x_max = min_x - (max_x - min_x) / 3, max_x + (max_x - min_x) / 3
    ax_y_min, ax_y_max = min_y - (max_y - min_y) / 3, max_y + (max_y - min_y) / 3

    # create a fig
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.set_xlim(ax_x_min, ax_x_max)
    axs.set_ylim(ax_y_min, ax_y_max)

    plt.xlabel('Height\nSigma^2 = ' + str(sigmaSquare))
    plt.ylabel('Weight')

    # get regression lines and points
    line_x = np.linspace(ax_x_min, ax_x_max, 1000)
    axs.plot(line_x, np.polyval(W_Bayesian, line_x), label='Bayesian', color='green')
    axs.plot(line_x, np.polyval(W_LSE, line_x), label='LSE', color='blue')
    axs.scatter(X[:, 4], Y, s=10, color='orange')

    axs.scatter(x_heights_to_be_estimated, np.polyval(W_Bayesian, x_heights_to_be_estimated), label="est from Bayesian",
                s=20, color='red')
    axs.scatter(x_heights_to_be_estimated, np.polyval(W_LSE, x_heights_to_be_estimated), label="est from LSE", s=20,
                color='pink')

    # set properties of the legend of the plot
    leg = axs.legend(loc='lower right', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    plt.savefig('prediction_sigma=%d.pdf' % sigmaSquare, facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
