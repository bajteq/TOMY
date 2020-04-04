import numpy as np
import matplotlib.pylab as plt
from skimage import color
import cv2 
from skimage.util import random_noise
from scipy import signal
import scipy

im = plt.imread("Xray-300x247.jpg")

im_gray = color.rgb2gray(im)

im_gray_u8 = np.asarray(im_gray*255, dtype=np.uint8)

def showHistogram(image, bins=255, cumulative=True):
    plt.figure()
    plt.hist(image.ravel(),bins=bins,cumulative=cumulative)
    plt.show()

showHistogram(im_gray_u8)

im_fft = np.fft.fft2(im_gray)

x_size, y_size = np.shape(im_gray_u8)


noise_img = random_noise(im_gray_u8)*255

im_fft2 = np.fft.fft2(noise_img)


def showFFT(fft):
    plt.imshow(np.log10(abs(fft)), cmap='gray')
    plt.axis('off')
    plt.show()

def showImage(img):
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()
    
showFFT(im_fft)
showFFT(im_fft2)

def isInCircle(x, y, center_x, center_y, radius):
    out = (x - center_x)**2 + (y - center_y)**2 < radius**2
    return out

def createDisk(radius, SE):
    disk = np.zeros_like(SE)
    x_SE, y_SE = np.shape(SE)
    x_center = int(np.floor(x_SE / 2))
    y_center = int(np.floor(y_SE / 2))
    for i in range(0,np.shape(SE)[0]):
        for j in range(0, np.shape(SE)[1]):
            disk[i,j] = isInCircle(i, j, x_center, y_center, radius + 0.2)
    return disk

def createSE(size = (3,3), shape='filled', radius = -1):
    SE = np.ones(size, np.uint8)
    if shape == 'filled':
        return SE
    elif shape == 'disk':
        if radius == -1:
            radius = int(np.floor(np.max(np.shape(SE)) / 2))
        return createDisk(radius, SE)

mask = createSE((3,3))

im_filtered = signal.convolve2d(noise_img, mask)
im_filtered_fft = np.fft.ifft(im_filtered)
showImage(noise_img)
showImage(im_filtered)

plt.subplot(2,3,1)
plt.imshow(im_gray_u8,cmap='gray')
plt.axis('off')
plt.title('given image')

plt.subplot(2,3,2)
plt.imshow(np.log10(abs(im_fft)),cmap='gray')
plt.axis('off')
plt.title('fft of given image')

plt.subplot(2,3,4)
plt.hist(im_gray_u8.ravel(),bins=255,cumulative=True)
plt.axis('off')
plt.title('histogram of given image')

plt.subplot(2,3,5)
plt.hist(noise_img.ravel(),bins=255,cumulative=True)
plt.axis('off')
plt.title('histogram of noised image')

plt.subplot(2,3,6)
plt.hist(im_filtered.ravel(),bins=255,cumulative=True)
plt.axis('off')
plt.title('histogram of filtered image')