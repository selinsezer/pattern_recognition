
# coding: utf-8

# In[60]:

import math
import numpy as np
#import scipy as sp
#import scipy.stats as stats
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid.axislines import SubplotZero
from numpy import inf


# In[62]:

def plotUnitCircle(p):
    
    """ plot some 2D vectors with p-norm < 1 """
    fig = plt.figure(1)
    ax = SubplotZero(fig, 111)
    fig.add_subplot(ax)
    #print(fig)
    #print(ax)
    for direction in ["xzero", "yzero"]:
        ax.axis[direction].set_axisline_style("-|>")
        ax.axis[direction].set_visible(True)
    
    for direction in ["left", "right", "bottom", "top"]:
            ax.axis[direction].set_visible(False)
    
    x = np.linspace(-1.0, 1.0, 1000)
    y = np.linspace(-1.0, 1.0, 1000)
    X, Y = np.meshgrid(x, y)
    F = (((abs(X) ** p + abs(Y) ** p) ** (1.0 / p))-1)
    ax.contour(X, Y, F, [0])
    plt.savefig('unitCircle.pdf', facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.10)
    plt.show()


# In[63]:

plotUnitCircle(0.5)

