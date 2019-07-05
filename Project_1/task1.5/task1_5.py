"""
Pattern Recognition Assignment 1 Task 1.5
estimating the dimension of fractal objects in an image
step 1# Convert the image to binary image
step 2# Produce box scaling sets S and compute corresponding box count X, 
        S obviously from 1/2 to 1/(2^(L-2)), while L =log2(height of image)
step 3# Use linear least square, estimate it's slope D and fit a line
        Line plot is saved as "Task_1_5_"+image_name+".pdf" into current
        directory.
"""
# lightning slope D = 1.5776850270780338
# tree slope D = 1.8463900565472429

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as msc
import scipy.ndimage as img

# choose the image here
image_name = 'tree-2'

"""
Get the binary image. This function is directly copied from Professor's code.
function's name and variable names were changed to maintain the coding standard
but functionality is the same. 
"""


def convert_to_binary_image(f):
    d = img.filters.gaussian_filter(f, sigma=0.50, mode='reflect') - \
        img.filters.gaussian_filter(f, sigma=1.00, mode='reflect')
    d = np.abs(d)
    m = d.max()
    d[d < 0.1 * m] = 0
    d[d >= 0.1 * m] = 1
    return img.morphology.binary_closing(d)


"""
In every level return it's number of boxes
image = binary image 
L  = scale number, scale = 1/(2^L)
return count = number of boxes for it's given scale.
"""


def scalar_box_counting(image, L):
    scale = 2 ** L
    sub_width = image.shape[0] // scale
    box_count = 0

    # get box image
    fig, ax = plt.subplots(1)
    img = image.astype(np.float)
    img[img > 0] = 180
    img[img == 0] = 0
    ax.imshow(img)

    for x in range(0, scale):
        for y in range(0, scale):
            # calculate pixels of a sub_image from the main image's pixels
            sub_image = image[x * sub_width:(x + 1) * sub_width, y * sub_width:(y + 1) * sub_width]

            '''
            If the sub_image contains at least one foreground pixel
            then cover that in a box
            '''

            # check for foreground pixel
            if sub_image.any():
                box_count += 1
                # draw a red square in that sub image
                rectangle = patches.Rectangle((y * sub_width, x * sub_width),
                                              sub_width, sub_width,
                                              linewidth=1, edgecolor='r',
                                              facecolor='none')
                ax.add_patch(rectangle)

    filename = "output_{0}_{1}.png".format(image_name, L)
    # save the image for every iteration
    plt.savefig(filename, facecolor='w', edgecolor='w', papertype=None,
                format='png', transparent=False, bbox_inches='tight',
                pad_inches=0.1)

    return box_count


"""
box counting, given binary image g, return two list, S and count
image = binary image
s     =  scaling factor list

return l_count = list of box count corresponding to each scale.
"""


def box_counting(image):
    s = []
    l_count = []
    image_width = image.shape[0]
    L = int(np.log2(image_width))
    for i in range(1, L - 1):
        l_count.append(scalar_box_counting(image, i))
        s.append(1.0 / (2 ** i))
    return s, l_count


def plot_log_line(D, b, s, count):
    s_inverse = np.divide(1, s)
    line_x = np.linspace(1, s_inverse.max(), num=100)
    line_y = np.power(10 * np.ones(len(line_x)), D * np.log10(line_x) + b)

    # create a figure
    fig = plt.figure()
    axs = fig.add_subplot(111)
    axs.plot(s_inverse, count, 'bo', label='data')
    axs.plot(line_x, line_y, 'r', label='fitting line')

    # set properties of the legend of the plot
    leg = axs.legend(loc='upper right', shadow=True, fancybox=True,
                     numpoints=1)
    leg.get_frame().set_alpha(0.5)

    # set x,y scale as log
    plt.yscale('log')
    plt.xscale('log')

    # set x,y axis labels
    axs.set_xlabel("log(1/s)")
    axs.set_ylabel("logn")
    plt.savefig("Task_1_5_" + image_name + ".pdf", facecolor='w',
                edgecolor='w',
                papertype=None,
                format='pdf',
                transparent=False,
                bbox_inches='tight',
                pad_inches=0.1)


"""
This function uses s and box counts to fit line using least squares.
    
D*log(1/s)+b=log(count) 
Let x = log(1/s)
    y = log(l_count)
    So it becomes Dx + b = y
    Then we can use LSM to calculate for D and b
"""


def line_fitting(s, l_count):
    x = np.log10(np.divide(1, s))
    y = np.log10(l_count)
    # to solve the linear least square, we should rewrite equation 
    # Dx+b=y
    # then through Linear least square Derivation, we know
    # w =>(D,b) =  (X*X.Transpose).Inv*X*y

    X = np.vstack([x, np.ones(len(x))])  # Column Vector X = [x,1]
    XInv = X.T  # Transpose of X
    X_XInv = np.matmul(X, XInv)  # dot product of X and transpose of X
    XInv_y = np.matmul(X, y)  # dot product of X transpose and y

    D, b = np.matmul(np.linalg.inv(X_XInv), XInv_y)

    # D, b = np.linalg.lstsq(X, y, rcond=None)[0]
    plot_log_line(D, b, s, l_count)
    return D, b


if __name__ == "__main__":
    image = msc.imread(image_name + '.png', flatten=True).astype(np.float)
    binary_image = convert_to_binary_image(image)
    s, l_count = box_counting(binary_image)
    D, b = line_fitting(s, l_count)
    print("Slope D = {0}".format(D))
