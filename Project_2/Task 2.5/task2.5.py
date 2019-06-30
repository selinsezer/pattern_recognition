import time

import matplotlib.pyplot as plt
import numpy as np


def get_euclidean_distance(X, point):
    X = X.T
    point = point[:, np.newaxis]
    return np.sum((X - point) ** 2, axis=0)


def find_knn(test_data, training_data, k):
    """
    find_knn function gets the predicted label from findNN function and computes the accuracy of prediction.
    :param test_data:
    :param training_data:
    :param k: use k nearest neighbors to vote for the label
    :return: the accuracy of our label prediction
    """
    correct_prediction = 0.
    nd = test_data[:, 0].size
    for row in range(nd):
        label = find_nearest_neighbors(test_data[row], training_data, k)
        if np.isclose(label, test_data[row][2]):
            correct_prediction += 1
    return correct_prediction / nd


def find_nearest_neighbors(point, training_data, k):
    """
     find_nearest_neighbors function computes distance^2 between every training data and the given single test data.
     Then we use linear search to find out k nearest neighbors and vote for the label.

    :param point: the given single test data
    :param training_data: training data
    :param k: use k nearest neighbors to vote for the label
    :return: the predicted label of the given single test data
    """
    dist_squares = get_euclidean_distance(training_data[:, 0:2], point[0:2])
    predict_labels = np.zeros(k)
    for i in range(k):
        min_distance_index = np.argmin(dist_squares)
        predict_labels[i] = training_data[min_distance_index][2]
        dist_squares[min_distance_index] = float('inf')
    c1 = sum(predict_labels == -1.)
    c2 = sum(predict_labels == 1.)

    if c1 > c2:
        return -1.
    return 1.


def m_plot(xs, ys):
    plt.plot(xs, ys, 'ro')
    for i in range(len(xs)):
        round_y = "%.5f" % ys[i]
        plt.annotate(round_y, xy=(xs[i], ys[i]), xytext=(xs[i], ys[i] - 0.003))
    line, = plt.plot(xs, ys, lw=2)
    plt.axis([0, 6, 0.88, 0.94])
    plt.xlabel("k")
    plt.ylabel("accuracy(%)")
    plt.savefig("accuracy.pdf", format="pdf")
    plt.show()
    return


if __name__ == "__main__":
    training_data = np.loadtxt('data2-train.dat')
    test_data = np.loadtxt('data2-test.dat')
    k_means_list = np.zeros(3)
    accuracy_list = np.zeros(3)

    for k in range(1, 7, 2):
        startTime = time.perf_counter()
        accuracy = find_knn(test_data, training_data, k)
        print("k =", k, ", accuracy =", accuracy, ", runtime =", time.perf_counter() - startTime, "s")
        i = (k - 1) // 2
        k_means_list[i] = k
        accuracy_list[i] = accuracy

    m_plot(k_means_list, accuracy_list)
