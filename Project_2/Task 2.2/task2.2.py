import matplotlib.pyplot as plt
import numpy as np
from math import pi
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from mpl_toolkits.mplot3d import Axes3D

plots_dir = 'plots'


def pdf(w, h, std_w, std_h, m_w, m_h, ro):
    Q = ((w - m_w) / std_w) ** 2 \
        - 2 * ro * (((w - m_w) / std_w) * ((h - m_h) / std_h)) \
        + ((h - m_h) / std_h) ** 2
    return np.exp(-Q / (2 * (1 - ro ** 2))) / (2 * pi * std_w * std_h * np.sqrt(1 - ro ** 2))


def plot_w_h(w, h, filename='w_h.png', w_=None, h_=None):
    plt.figure()
    plt.plot(h, w, 'ro', color='blue')
    plt.xlabel('Height')
    plt.ylabel('Weight')
    if w_ is not None and h_ is not None:
        plt.plot(h_, w_, 'bo', color='red')
    plot_path = '/'.join([plots_dir, filename])
    plt.savefig(plot_path, facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)


def plot_bi_variate_distribution(w, h, std_w, std_h, mean_w, mean_h, ro, filename):
    x_span = 0.5 * (np.max(w) - np.min(w))
    y_span = 0.5 * (np.max(h) - np.min(h))
    x = np.arange(np.min(w) - x_span, np.max(w) + x_span, 0.1)
    y = np.arange(np.min(h) - y_span, np.max(h) + y_span, 0.1)
    x, y = np.meshgrid(x, y)
    z = pdf(x, y, std_w, std_h, mean_w, mean_h, ro)
    fig = plt.figure()
    # ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    surf = ax.plot_surface(x, y, z,
                           cmap=cm.viridis,
                           linewidth=0,
                           antialiased=False)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.05f'))
    plt.xlabel('Weight')
    plt.ylabel('Height')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plot_path = '/'.join([plots_dir, filename])
    plt.savefig(plot_path, facecolor='w', edgecolor='w',
                papertype=None, format='png', transparent=False,
                bbox_inches='tight', pad_inches=0.1)


def main():
    dt = np.dtype([('weight', np.int), ('height', np.int), ('sex', np.str_, 1)])
    d = np.loadtxt(fname='whData.dat', comments='#', dtype=dt)
    # heights with missed weight
    h_ = d[np.where(d['weight'] < 0)]['height']
    # remove outliers
    d = d[np.where(d['weight'] >= 0)]
    weight = d['weight']
    height = d['height']
    # mean of weight
    weight_mean = np.mean(weight)
    # mean of height
    height_mean = np.mean(height)
    print('means of weight and height: {}, {}'.format(weight_mean, height_mean))
    # deviation of weight
    std_weight = np.std(weight)

    # deviation of height
    std_height = np.std(height)
    sigma_square_height = std_height * std_height
    sigma_square_weight = std_weight * std_weight
    print('sigma weight: {}, sigma height: {}'.format(std_weight, std_height))

    # covariance matrix 2x2
    cov = np.cov(np.array([weight, height]), bias=True)
    # covariance just for weight and height
    # cov = np.sum((weight-w_mean)*(height-h_mean))/len(weight)

    # covariance between array x and array y
    # The diagonal entries of the covariance matrix are the variances
    # and the other entries are the co-variances
    cov_w_h = cov[0][1]  # diagonal value
    print('covariance of weight and height: {}'.format(cov_w_h))

    # Correlation coefficient
    ro = cov_w_h / (std_weight * std_height)
    print('ro : %s' % ro)
    plot_w_h(weight, height, 'w_h.png')
    plot_bi_variate_distribution(weight, height, std_weight, std_height, weight_mean, height_mean, ro,
                                 'bi_variate_distr.png')
    # predict missed weights
    w_ = weight_mean + (ro * (h_ - height_mean)) / (std_weight / std_height)
    print('predicted weights: %s \n for heights: %s' % (w_, h_))
    plot_w_h(weight, height, 'filled_missings.png', w_, h_)


if __name__ == '__main__':
    main()
