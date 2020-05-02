import numpy as np
import matplotlib.pylab as plt
from skimage import color
from skimage.util import random_noise
from math import sqrt
import cv2

#przygotowywnaie obrazow
im = plt.imread("Xray-300x247.jpg")
im_gray = color.rgb2gray(im)
im_gray_u8 = np.asarray(im_gray*255, dtype=np.uint8)
im_fft_cen = np.fft.fftshift(np.fft.fft2(im_gray))
noise_img = random_noise(im_gray_u8)*255

#filtracja
def distance(point1,point2):
    return sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def idealFilterLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            if distance((y,x),center) < D0:
                base[y,x] = 1
    return base

im_fft2 = np.fft.fft2(noise_img)
im_fft2_cen = np.fft.fftshift(im_fft2)

LowPassCenter = im_fft2_cen * idealFilterLP(50,noise_img.shape)
LowPass = np.fft.ifftshift(LowPassCenter)
inverse_LowPass = np.fft.ifft2(LowPass)



plt.figure(figsize = (12,12))
plt.subplot(3,3,1)
plt.imshow(im_gray_u8,cmap='gray')
plt.axis('off')
plt.title('given image')

plt.subplot(3,3,2)
plt.imshow(noise_img,cmap='gray')
plt.axis('off')
plt.title('noised image')

plt.subplot(3,3,3)
plt.imshow(np.abs(inverse_LowPass), "gray")
plt.axis('off')
plt.title('Reconstruted image')

plt.subplot(3,3,4)
plt.hist(im_gray_u8.ravel(),bins=255,cumulative=False)
plt.title('histogram of given image')

plt.subplot(3,3,5)
plt.hist(noise_img.ravel(),bins=255,cumulative=False)
plt.title('histogram of noised image')

plt.subplot(3,3,6)
plt.hist(np.abs(inverse_LowPass).ravel(),bins=255,cumulative=False)
plt.title('histogram of filtered image')

plt.subplot(3,3,7)
plt.imshow(np.log10(abs(im_fft_cen)),cmap= "gray")
plt.axis('off')
plt.title('centered spectrum of given image')

plt.subplot(3,3,8)
plt.imshow(np.log10(abs(im_fft2_cen)), cmap='gray')
plt.axis('off')
plt.title('centered spectrum of noised image')

plt.subplot(3,3,9)
plt.imshow(np.log10(abs(LowPassCenter)),cmap= "gray")
plt.axis('off')
plt.title('centered spectrum with mask')

#roznica
noise_img_u8=np.asarray(noise_img, dtype=np.uint8)
img_diff1=cv2.absdiff(im_gray_u8,noise_img_u8)


low_pass_u8=np.asarray(np.abs(inverse_LowPass), dtype=np.uint8)
img_diff2=cv2.absdiff(im_gray_u8,low_pass_u8)

plt.figure(figsize = (12,12))

plt.subplot(1,2,1)
plt.imshow(img_diff1,"gray")
plt.axis('off')
plt.title('roznica miedzy wejciowym i zaszumionym')

plt.subplot(1,2,2)
plt.imshow(img_diff2,"gray")
plt.axis('off')
plt.title('roznica miedzy wejciowym i przefiltrowanym')