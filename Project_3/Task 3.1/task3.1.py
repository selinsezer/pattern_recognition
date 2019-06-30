import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt
import time



# number of centroids(k)
n_clusters = 3

# Number of time the k-means algorithm will be run with different centroid seeds
n_init = 1

# save figure as pdf
save = False

# benchmark algortihms
benchmark = True


def plotData2D(data, filename, labels, centers, title, axis=['x', 'y']):

    #Plot 2D data

    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    if not save:
        filename = None

    # k < 9
    colors = "bgrcmykw"

    # same scaling from data to plot units for x and y
    axs.set_aspect('equal')

    x, y = data[:, 0], data[:, 1]
    for i in range(n_clusters):
        axs.scatter(x[labels == i], y[labels == i], alpha=0.5, c=colors[i],
                    edgecolors='none', s=20)
        axs.scatter(centers[i, 0], centers[i, 1], c=colors[i],
                    edgecolors='black', s=75, marker='s')

        # plot the data
    # axs.plot(X[:,0], X[:,1], 'ro', label='data', alpha=0.4, c=labels)
    plt.xlabel(axis[0])
    plt.ylabel(axis[1])

    # set x and y limits of the plotting area
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()

    axs.set_xlim(xmin - 0.2, xmax + 0.2)
    axs.set_ylim(ymin - 0.2, ymax + 0.2)
    axs.set_title(title)

    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()


def lloyd_kmeans(data):
    """
    Compute regular kmeans algorithm
    """

    centroid, labels = kmeans2(data, k=n_clusters, minit='points')
    print('\nlloyd_kmeans(centroids):\n')
    print(centroid)

    # plot clusters
    plotData2D(data, 'lloyd' + str(int(round(time.time() * 1000))) + '.pdf'
               , labels, centroid, 'Lloyd\'s Algorithm')


def hartigan_kmeans(data):
    """
    Compute hartigan's kmeans algorithm
    """

    # assign random cluster to data samples
    labels = np.random.randint(n_clusters, size=data.shape[0])

    # compute mu_i
    mu = [np.mean(data[labels == i], axis=0) for i in range(n_clusters)]

    converged = False
    while (not converged):
        converged = True

        for i in range(data.shape[0]):
            c_i = labels[i]
            # remove x_i
            mask = labels == c_i
            mask[i] = False
            # recompute mu_j
            mu[c_i] = np.mean(data[mask], axis=0)

            E = [0] * n_clusters
            for k in range(n_clusters):
                mu_init = mu[k]
                labels[i] = k
                mu[k] = np.mean(data[labels == k], axis=0)

                # compute energies
                for n in range(data.shape[0]):
                    E[k] += (data[n, 0] - mu[labels[n]][0]) ** 2 + (data[n, 1] -
                                                                    mu[labels[n]][1]) ** 2

                labels[i] = c_i
                mu[k] = mu_init

            # determine new label
            c_w = np.argsort(E)[0]
            if c_w != c_i:
                converged = False
                labels[i] = c_w
                mu[c_w] = np.mean(data[labels == c_w], axis=0)

    print('\nHartigan_kmeans(centroids):\n')
    print(np.asarray(mu))
    plotData2D(data, 'Hartigan' + str(int(round(time.time() * 1000))) + '.pdf',
               labels, np.asarray(mu), 'Hartigan\'s Algorithm')


def macQueen_kmeans(data):
    """
    Compute macQueen's online kmeans algorithm
    """
    mu = np.random.rand(n_clusters, 2)
    n_i = np.zeros(3)

    for i in range(data.shape[0]):
        # determine winner centroid
        E = [0] * n_clusters
        for k in range(n_clusters):
            E[k] = (data[i, 0] - mu[k, 0]) ** 2 + (data[i, 1] - mu[k, 1]) ** 2

        # update cluster size and centroid
        mu_w = np.argsort(E)[0]
        n_i[mu_w] += 1
        mu[mu_w] += (1. / n_i[mu_w]) * (data[i] - mu[mu_w])

    labels = np.zeros(data.shape[0])
    # label samples wrt. cluster centers above
    for i in range(data.shape[0]):
        E = [0] * n_clusters
        for k in range(n_clusters):
            E[k] = (data[i, 0] - mu[k, 0]) ** 2 + (data[i, 1] - mu[k, 1]) ** 2

        # update cluster size and centroid
        labels[i] = np.argsort(E)[0]

    print('\nMacQueen_kmeans(centroids):\n')
    print(np.asarray(mu))
    plotData2D(data, 'macqueen' + str(int(round(time.time() * 1000))) + '.pdf',
               labels, mu, 'MacQueen\'s Algorithm')


if __name__ == "__main__":
    # read data from file
    data = np.genfromtxt('data-clustering-1.csv', delimiter=",").T

    if benchmark:
        bench_m = np.zeros(n_clusters)
        n_iter = 10
        runtime_sum = 0
        # run all algorithms for 10 times for benchmarking.      

        start = time.time()
        for i in range(n_iter):
            lloyd_kmeans(data)
            done = time.time()
            bench_m[0] = done - start

            runtime_sum = runtime_sum + bench_m[0]

        print('Lloyed: ', bench_m[0]/10)

        runtime_sum = 0
        start = time.time()
        for i in range(n_iter):
            hartigan_kmeans(data)
            done = time.time()
            bench_m[1] = done - start

            runtime_sum = runtime_sum + bench_m[1]
        print('Hartigen: ', bench_m[1]/10)

        runtime_sum = 0
        start = time.time()
        for i in range(n_iter):
            macQueen_kmeans(data)
            done = time.time()
            bench_m[2] = done - start

            runtime_sum = runtime_sum + bench_m[2]

        print('MacQueen', bench_m[2]/10)

    else:
        lloyd_kmeans(data)
        hartigan_kmeans(data)
        macQueen_kmeans(data)
