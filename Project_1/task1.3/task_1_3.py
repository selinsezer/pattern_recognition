import numpy as np
import math
import matplotlib.pyplot as plt
import time


def newtonParametersCalculator(k, a, histogramData):
    # We Input parameters 'k' and 'a' (alpha) into the function.
    N = len(histogramData)
    sum_log_di = np.sum(np.log(histogramData))
    sum_di_a_k = np.sum((histogramData / a) ** k)

    # Calculate all the matrix elements of the Newtonian Method.
    dl_dk = N / k - N * math.log(a) + sum_log_di - np.sum(
        ((histogramData / a) ** k) * np.log(histogramData / a))
    dl_da = (k / a) * (sum_di_a_k - N)
    d2l_dk = -N / (k ** 2) - np.sum(((histogramData / a) ** k) *
                                    (np.log(histogramData / a)) ** 2)
    d2l_da = (k / ((a) ** 2)) * (N - (k + 1) * sum_di_a_k)
    d2l_dkda = M21 = (1 / a) * sum_di_a_k + (k / a) * np.sum(
        ((histogramData / a) ** k) * np.log(histogramData / a)) - N / a
    return np.array(np.matmul(np.linalg.inv(np.matrix([[d2l_dk, d2l_dkda],
                                                       [d2l_dkda, d2l_da]])), np.array([-dl_dk, -dl_da])) +
                    np.array([k, a]))[0]


def iterationFunction(k, a, n, histogram):
    # Normally, a termination condition could also be specified, but since we
    # have observed that 'k' and 'a' converge in 20 iterations, no termination
    # condition is specified.

    # Termination condition could be of the form while newParam!=oldParam
    # (If they don't terminate exactly, we can specify a tolerance value)
    oldParameters = np.array([k, a])
    for i in range(n):
        newParameters = newtonParametersCalculator(oldParameters[0],
                                                   oldParameters[1], histogram)
        oldParameters = newParameters
    return newParameters


if __name__ == "__main__":
    # read data from file
    h = np.genfromtxt('myspace.csv', delimiter=",")[:, 1]

    # Remove leading zeros.
    while h[0] == 0:
        h = np.delete(h, 0)

    # Generate x-Values for the histogram
    xValues = np.arange(1, len(h) + 1)

    # Data generation for Newton's Method.
    histogramOfX = [];
    for i in range(len(h)):
        histogramOfX = np.append(histogramOfX, [xValues[i]] * int(h[i]))

    parameters = iterationFunction(1, 1, 20, histogramOfX)
    k = parameters[0]
    a = parameters[1]
    filename = "task_1_3.pdf"

    # plot fitted distribution with observed samples
    plt.plot(xValues, h)
    plt.axis([xValues[0], xValues[len(xValues) - 1], 0, max(h) + 10])

    # Scaling factor for the Weibull fit was derived by setting:
    # scale_factor*integral[Weibull] = Area Under the curve for Histogram
    plt.plot(sum(h) * (k / a) * ((xValues / a) ** (k - 1)) * np.exp(-1 *
                                                                    ((xValues / a) ** k)), 'r')
    plt.legend(('Google Data', 'Scaled Weibull Fit(Newton)'))

    plt.savefig(filename, facecolor='w', edgecolor='w', papertype=None,
                format='pdf', transparent=False, bbox_inches='tight',
                pad_inches=0.1)
    plt.close()
