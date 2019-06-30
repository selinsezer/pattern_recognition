import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def projection_function(X, eigenset):
    return np.dot(X, eigenset.T)


def PCA(X, n_biggest_vector):
    # calculate the overall mean of data
    overall_data_mean = np.mean(X, axis=0)
    # normalize to zero mean
    zero_mean_data = X - overall_data_mean
    # transpose data
    covariance_matrix = np.cov(np.transpose(zero_mean_data))
    # derive eig values and vectors
    eigen_values, eigen_vectors = np.linalg.eigh(covariance_matrix)
    # return n-biggest eigen-vectors
    return eigen_vectors.T[np.argpartition(eigen_values, -n_biggest_vector)[-n_biggest_vector:]]


def LDA(X, class_length, n_biggest_vector):
    # calculating the overall mean of data
    overall_data_mean = np.mean(X, axis=0)
    # class_means contains the means related for each class
    class_means = np.array([np.mean(X[0:50], axis=0),
                            np.mean(X[50:100], axis=0),
                            np.mean(X[100:150], axis=0)])
    # mean_list: we use this array (150 elements) for sake of simplicity during
    # the calculation
    mean_list = np.concatenate([[class_means[i]] * class_length[i]
                                for i in range(len(class_length))])
    # here we calculate S_B which is the within class scatter matrix
    S_W = np.zeros((len(X[0]), len(X[0])))

    for i in range(len(X)):
        S_W = S_W + np.outer((X - mean_list)[i], (X - mean_list)[i])

    # Between class scatter matrix
    S_B = np.zeros((len(X[0]), len(X[0])))
    for i in range(len(class_means)):
        S_B = S_B + class_length[i] * np.outer((class_means[i] - overall_data_mean),
                                               (class_means[i] - overall_data_mean))

    eigen_values, eigen_vectors = np.linalg.eigh(np.dot(np.linalg.pinv(S_W), S_B))
    return eigen_vectors.T[np.argpartition(eigen_values, -n_biggest_vector)[-n_biggest_vector:]]


if __name__ == "__main__":

    data_x = np.transpose(np.genfromtxt("data-dimred-X.csv", delimiter=','))
    y_label = np.genfromtxt("data-dimred-y.csv")

    class_length_array = np.array(
        [len(y_label[y_label == 1]), len(y_label[y_label == 2]),
         len(y_label[y_label == 3])])

    final_pca = projection_function(data_x, PCA(data_x, 2))
    final_lda = projection_function(data_x, LDA(data_x, class_length_array, 2))

    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(final_lda.T[0][np.where(y_label == i)],
                    final_lda.T[1][np.where(y_label == i)], color=j)
    plt.title('LDA Algorithm 2D')
    plt.legend(['#1', '#2', '#3'], loc=3)
    plt.savefig('Figure_LDA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        plt.scatter(final_pca.T[0][np.where(y_label == i)],
                    final_pca.T[1][np.where(y_label == i)], color=j)
    plt.title('PCA Algorithm 2D')
    plt.legend(['#1', '#2', '#3'], loc=3)
    plt.savefig('Figure_PCA_2D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    final_pca_3d = projection_function(data_x, PCA(data_x, 3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D(final_pca_3d.T[0][np.where(y_label == i)],
                      final_pca_3d.T[1][np.where(y_label == i)],
                      final_pca_3d.T[2][np.where(y_label == i)], color=j)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('PCA Algorithm 3D')
    plt.savefig('Figure_PCA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()

    final_lda_3d = projection_function(data_x, LDA(data_x, class_length_array, 3))
    fig = plt.figure()
    axs = Axes3D(fig)
    for i, j in zip([1., 2., 3.], ['green', 'blue', 'red']):
        axs.scatter3D(final_lda_3d.T[0][np.where(y_label == i)],
                      final_lda_3d.T[1][np.where(y_label == i)],
                      final_lda_3d.T[2][np.where(y_label == i)], color=j)
    plt.legend(['Group #1', 'Group #2', 'Group #3'], loc=3)
    plt.title('LDA Algorithm 3D')
    plt.savefig('Figure_LDA_3D.pdf', facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.show()