import time
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage as nd
from scipy import interpolate
from scipy import linalg
from matplotlib.backend_bases import MouseButton
import matplotlib.animation as animation
from scipy import signal


def generate_circle(y_size, x_size, x_origin, y_origin, radius):
    image = np.zeros((y_size, x_size))
    x_grid, y_grid = np.meshgrid(np.arange(x_size), np.arange(y_size))
    indices = np.square((x_grid - x_origin)) + np.square((y_grid-y_origin)) < radius*radius
    image[indices] = 1
    return image

y_size, x_size = 512, 512
radius = 100
x_origin, y_origin = 256, 256
image_circle = generate_circle(y_size, x_size, x_origin, y_origin, radius)
image_rectangle = np.zeros((y_size, x_size))
image_rectangle[100:400, 150:350] = 1
image_ushape = image_circle.copy()
image_ushape[0:256, 230:280] = 0

image_circle = image_circle + np.random.randn(y_size, x_size)*0.1
image_rectangle = image_rectangle + np.random.randn(y_size, x_size)*0.1
image_ushape = image_ushape + np.random.randn(y_size, x_size)*0.1

plt.figure(dpi=200)
plt.subplot(1, 3, 1)
plt.imshow(image_circle, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(image_rectangle, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(image_ushape, cmap='gray')
plt.axis('off')
plt.show()




class ContourCreator(object):
    # TO DO
    def __init__(self, plot):
        self.plot = plot
        self.xs = []
        self.ys = []
        self.x_res = []
        self.y_res = []
        plt.imshow(plot,cmap='gray')
        plt.connect('button_press_event',self)
        plt.show()

    def __call__(self, event):
        if event.button is MouseButton.LEFT:
            self.xs.append(event.xdata)
            self.ys.append(event.ydata)
            self.redraw()
        if event.button not in [MouseButton.RIGHT, MouseButton.LEFT]:
            self.xs.append(self.xs[0])
            self.ys.append(self.ys[0])
            self.redraw()
            plt.close()

        if event.button is MouseButton.RIGHT:
            for i in range(len(self.xs)-1):
                self.x_res = self.xs[i]
                self.y_res = self.ys[i]

            self.redraw()


    def redraw(self):
        plt.plot(self.xs, self.ys, "g*-")
        plt.draw()



def click_points(image):
    contour = ContourCreator(image)
    xs, ys = contour.xs, contour.ys
    return xs, ys


xs_c, ys_c = click_points(image_circle)
xs_r, ys_r = click_points(image_rectangle)
xs_u, ys_u = click_points(image_ushape)

plt.figure(dpi=200)
plt.subplot(1, 3, 1)
plt.imshow(image_circle, cmap='gray')
plt.axis('off')
plt.plot(xs_c, ys_c, "g*-")
plt.subplot(1, 3, 2)
plt.imshow(image_rectangle, cmap='gray')
plt.axis('off')
plt.plot(xs_r, ys_r, "g*-")
plt.subplot(1, 3, 3)
plt.imshow(image_ushape, cmap='gray')
plt.axis('off')
plt.plot(xs_u, ys_u, "g*-")
plt.show()

def reinterpolate_contours(xs, ys, dmin, dmax):
    new_xs = []
    new_ys = []
    def calculateDistance(p1, p2):
        dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        return dist
    for i in range(1, len(xs)):
        p1 = (xs[i-1], ys[i-1])
        p2 = (xs[i], ys[i])
        initial_distance = calculateDistance(p1, p2)
        points_to_interpolate = int(initial_distance/dmin)
        sublist_xs = np.linspace(p1[0], p2[0], points_to_interpolate+1)
        sublist_ys = np.linspace(p1[1], p2[1], points_to_interpolate+1)
        for i in range(len(sublist_xs)):
            new_xs.append(sublist_xs[i])
            new_ys.append(sublist_ys[i])
    return new_xs, new_ys


xs_cr, ys_cr = reinterpolate_contours(xs_c, ys_c, 3, 6)
xs_rr, ys_rr = reinterpolate_contours(xs_r, ys_r, 3, 6)
xs_ur, ys_ur = reinterpolate_contours(xs_u, ys_u, 3, 6)
plt.figure(dpi=200)
plt.subplot(1, 3, 1)
plt.imshow(image_circle, cmap='gray')
plt.axis('off')
plt.plot(xs_cr, ys_cr, "g*-", markersize=0.5, linewidth=0.1)
plt.subplot(1, 3, 2)
plt.imshow(image_rectangle, cmap='gray')
plt.axis('off')
plt.plot(xs_rr, ys_rr, "g*-", markersize=0.5, linewidth=0.1)
plt.subplot(1, 3, 3)
plt.imshow(image_ushape, cmap='gray')
plt.axis('off')
plt.plot(xs_ur, ys_ur, "g*-", markersize=0.5, linewidth=0.1)
plt.show()
def generate_coefficient_matrix(N, alpha, beta):
    # TO DO
    row = np.r_[-2*alpha-6*beta,
    +alpha+4*beta,
    -beta,
    np.zeros(N-5),
    -beta,
    +alpha+4*beta]
    A = np.zeros((N,N))
    for i in range(N):
        A[i] = -np.roll(row,i)

    return A

N = 5
alpha = 0.1
beta = 0.1
print("Propagation matrix: ")
print(generate_coefficient_matrix(N, alpha, beta))




def calculate_gradient_at_contour_points(xs, ys, x_gradient, y_gradient):
    new_xs = []
    new_ys = []

    for i in range(len(xs)):
        point = (xs[i], ys[i])
        down_xs = int(np.floor(point[0]))
        up_xs = int(np.ceil(point[0]))
        down_ys = int(np.floor(point[1]))
        up_ys = int(np.ceil(point[1]))
        mean_xs = np.mean([x_gradient[down_xs][down_ys],x_gradient[down_xs][up_ys],
                           x_gradient[up_xs][down_ys],x_gradient[up_xs][up_ys]])
        mean_ys = np.mean([y_gradient[down_xs][down_ys],y_gradient[down_xs][up_ys],
                           y_gradient[up_xs][down_ys], y_gradient[up_xs][up_ys]])
        new_xs.append(mean_xs)
        new_ys.append(mean_ys)
    return new_xs, new_ys

np.random.seed(123)

y_gradient = np.random.randn(5, 5)
x_gradient = np.random.randn(5, 5)
xs = [0.5, 1.5, 2.5]
ys = [0.5, 1.2, 1.7]
print("Gradient at contour location: ")
g_xs, g_ys = calculate_gradient_at_contour_points(xs, ys, x_gradient, y_gradient)
print("X: ", g_xs)
print("Y: ", g_ys)

