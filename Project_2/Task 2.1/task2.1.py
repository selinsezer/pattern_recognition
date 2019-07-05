import matplotlib.pyplot as plt
import numpy as np

out_file = None

# read data as 2D array of data type 'object'
data = np.loadtxt('whData.dat', dtype=np.object, comments='#', delimiter=None)

# read height and weight data into 2D array (i.e. into a matrix)
X = data[:, 0:2].astype(np.float32)

# Task 2.1
mask = (X[:, 0] == -1)
h_est, h = X[mask][:, 1], X[~mask][:, 1]
y = X[~mask][:, 0]

min_h, max_h = min(h), max(h)
min_y, max_y = min(y), max(y)

ax_x_min, ax_x_max = min_h - (max_h - min_h) / 3, max_h + (max_h - min_h) / 3
ax_y_min, ax_y_max = min_y - (max_y - min_y) / 3, max_y + (max_y - min_y) / 3

for d in [1, 5, 10]:
    # create a figure and its axes
    fig = plt.figure()
    axs = fig.add_subplot(111)

    X = np.vander(h, d + 1)
    w = np.linalg.lstsq(X, y, rcond=None)[0]
    # draw plot
    x = np.linspace(ax_x_min, ax_x_max, 10000)
    axs.set_xlim(ax_x_min, ax_x_max)
    axs.set_ylim(ax_y_min, ax_y_max)

    plt.xlabel('Height')
    plt.ylabel('Weight')

    axs.plot(x, np.polyval(w, x), label='deg=%d' % d, color='green')
    axs.scatter(h, y, s=10, color='orange')
    axs.scatter(h_est.tolist(), np.polyval(w, h_est.tolist()), s=20, color='red')

    # set properties of the legend of the plot
    leg = axs.legend(loc='lower right', shadow=True, fancybox=True, numpoints=1)
    leg.get_frame().set_alpha(0.5)

    plt.savefig('prediction_GA_d=%d.pdf' % d, facecolor='w', edgecolor='w',
                papertype=None, format='pdf', transparent=False,
                bbox_inches='tight', pad_inches=0.1)
    plt.close()
