
# coding: utf-8

# In[59]:

import pylab
from numpy import array, linalg, random, sqrt, inf
from matplotlib import pyplot as plt

def plotUnitCircle(p):
    """ plot some 2D vectors with p-norm < 1 """
    for i in range(5000):
        x = array([random.rand()*2-1,random.rand()*2-1])
        if linalg.norm(x,p) < 1:
            plt.gca().set_aspect('equal', adjustable='box')
            pylab.plot(x[0],x[1],'ro')
    pylab.axis([-1.5, 1.5, -1.5, 1.5])
    plt.savefig('UnitCircle-4.pdf', facecolor='w', edgecolor='w',
                    papertype=None, format='pdf', transparent=False,
                    bbox_inches='tight', pad_inches=0.1)
    pylab.show()


# In[54]:

plotUnitCircle(1)


# In[56]:

plotUnitCircle(2)


# In[58]:

plotUnitCircle(3)


# In[60]:

plotUnitCircle(4)

