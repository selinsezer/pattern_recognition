import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def read_data():
    #   read X data
    x_path = 'xor-X.csv'
    X = np.genfromtxt(x_path, delimiter=',').transpose()

    #   read y data
    y_path = 'xor-y.csv'
    y = np.genfromtxt(y_path, delimiter=',')

    return X, y


def f_function(x, w, theta):
    # f function as given in homework description is applied to each row of X
    func = lambda x_i: 2 * np.exp(-0.5 * (((np.dot(w.transpose(), x_i) - theta) ** 2))) - 1
    new_y = np.apply_along_axis(func, 1, x)
    return new_y


def partial_der_w(x_i, y_i, w, theta):
    #  partial derivative of the given function w.r.t. w
    return np.exp(-0.5*(-theta + np.dot(w.transpose(), x_i))**2)*x_i*(-theta + np.dot(w.transpose(),x_i))*(-1 + 2*np.exp(-0.5 *(-theta + np.dot(w.transpose(), x_i))** 2)-y_i)


def compute_delta_w(x, y, w, theta):
    #  calculation of delta w
    return -2 * sum(partial_der_w(x_i, y_i, w, theta) for x_i, y_i in zip(x, y))


def partial_der_theta(x_i, y_i, w, theta):
    #  partial derivative of the given function w.r.t. theta
    return np.exp(-0.5*(-theta + np.dot(w.transpose(), x_i)) ** 2)*(-theta + np.dot(w.transpose(), x_i))*(-1 + 2 * np.exp(-0.5*(-theta + np.dot(w.transpose(), x_i)) ** 2) - y_i)


def compute_delta_theta(x, y, w, theta):
    #  calculation of delta theta
    return 2 * sum(partial_der_theta(x_i, y_i, w, theta) for x_i, y_i in zip(x, y))


def train_non_mon_neuron(X, y, epochs=50, theta_step=0.001, w_step=0.005):
    #   randomly initialize w and theta
    w = np.random.rand(1,2)[0]
    theta = np.random.rand(1,1)[0]

    #   for a given number of times, repeat: calculate the delta w and delta theta, subtract from current w and theta
    for i in range(epochs):
        delta_w = compute_delta_w(X, y, w, theta)
        delta_theta = compute_delta_theta(X, y, w, theta)
        w -= w_step * delta_w
        theta -= theta_step * delta_theta

    #   return resulting w and theta
    return w, theta


def plot_decision_regions(X, y, w, theta, resolution=0.02):
    #   setup marker generator and color map
    colors = ('sandybrown', 'cornflowerblue')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    #   calculate the f function for some area for plotting
    x1_min, x1_max = X[:,  0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = f_function(np.array([xx1.ravel(), xx2.ravel()]).T, w, theta)
    Z = Z.reshape(xx1.shape)

    #   get x values for plotting the initial input
    x1s = X[:, 0]
    x2s = X[:, 1]

    #   setup figure and set axis
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    #   set colors for input plotting
    color = ['blue' if y_i == 1 else 'orange' for y_i in y]
    ax.scatter(x1s, x2s, color=color, alpha=0.5)
    #   plot decision region
    ax.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap, levels=1)
    plt.show()


if __name__ == '__main__':
    X, y = read_data()
    w, theta = train_non_mon_neuron(X, y, 50)
    plot_decision_regions(X, y, w, theta)