import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.metrics.pairwise import pairwise_distances

# random seed
random_state = 0

# number of centroids(k)
n_clusters = 2

# Number of time the k-means algorithm will be run with different centroid seeds
n_init = 1

# save output figure
save = True

def plotData2D(data, filename, labels, centers=[], title=None, axis=['x', 'y']):
    """
    Plot 2D data
   """
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)
    
    if not save:
        filename = None
    
    # k < 9
    colors = "bgrcmykw"

    # same scaling from data to plot units for x and y
    axs.set_aspect('equal')
    
    x, y = data[:,0], data[:,1]
    for i in range(n_clusters):
        axs.scatter(x[labels==i], y[labels==i], alpha=0.5, c=colors[i],  
                     edgecolors='none', s=20)
        if len(centers):
            axs.scatter(centers[i,0], centers[i,1], c=colors[i],  
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
    
    axs.set_xlim(xmin-0.2, xmax+0.2)
    axs.set_ylim(ymin-0.2, ymax+0.2)
    axs.set_title(title)
    
    # either show figure on screen or write it to disk
    if filename == None:
        plt.show()
    else:
        plt.savefig(filename, facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    plt.close()
    
def spectral(data):
    """
    Run spectral clusting algorithm and plot clusters.
    """
    #Spectral clustering
    beta = 2.0

    S = np.exp(-beta * np.power(pairwise_distances(data, metric='euclidean'),2))
    D = np.diag(np.sum(S, axis=0))
    L = D - S
    
    #calculate eigenvalues and eigenvectors
    w, v = np.linalg.eigh(L)
    
    #The eigenvector u corresponding to the second smallest eigenvalue
    fiedl = v[:,np.argsort(w)[1]] 
    labels = (fiedl < 0).astype(int)
    
    plotData2D(data, 'spectral'+str(beta)+'.pdf', labels, [], 
               'Spectral Clustering(beta='+str(beta)+')' )

def hartigan_kmeans(data):
    """
    Compute hartigan's kmeans algorithm
    """
    # assign random cluster to data samples
    labels = np.random.randint(n_clusters, size=data.shape[0])
    
    #compute mu_i
    mu = [np.mean(data[labels==i], axis=0) for i in range(n_clusters)]
    
    converged = False
    while( not converged ):
        converged = True
            
        for i in range(data.shape[0]):
            c_i = labels[i]
            # remove x_i
            mask = labels==c_i
            mask[i] = False 
            # recompute mu_j
            mu[c_i] = np.mean(data[mask], axis=0)
            
            E = [0]*n_clusters
            for k in range(n_clusters):
                mu_init = mu[k]
                labels[i] = k
                mu[k] = np.mean(data[labels==k], axis=0)
                
                for n in range(data.shape[0]):
                    E[k] +=  (data[n,0] - mu[labels[n]][0])**2 + (data[n,1] - 
                     mu[labels[n]][1])**2
                        
                labels[i] = c_i
                mu[k] = mu_init
            
            # determine new label
            c_w = np.argsort(E)[0]
            if c_w != c_i:
                converged = False
                labels[i] = c_w
                mu[c_w] = np.mean(data[labels==c_w], axis=0)

    print ('\nhartigan_kmeans(centroids):\n')
    print (np.asarray(mu))
    plotData2D(data, 'hartigan'+str(int(round(time.time() * 1000)))+'.pdf',
               labels, np.asarray(mu), 'Hartigan\'s Algorithm' )
    
if __name__ == "__main__":
    # read data from file
    data = np.genfromtxt('data-clustering-2.csv', delimiter=",").T
    hartigan_kmeans(data)
    spectral(data)
