import numpy as np
import matplotlib.pylab as plt
from skimage import color
from scipy import signal
import os

vertical_mask = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
horizontal_mask = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

def edge_detection_function(image_name):
   
    #image filtered with masks
    im=plt.imread(image_name)
    if len(np.shape(im))==3:
        im=color.rgb2gray(im)
    x_filter=signal.convolve2d(im,horizontal_mask)
    y_filter=signal.convolve2d(im,vertical_mask)
    
    #magnitude
    magnitude = np.sqrt(x_filter**2+y_filter**2)
    """
    Nie zadziala jesli obraz nie bedzie numpy arrayem, w przeciwnym razie konieczne jest zamienienie listy na tablice
    """
    
    #angle
    angle = np.arctan2(x_filter,y_filter)
    """
    Funckja arctan2 radzi sobie z przypadkami dzielenia przez zero. Innym rozwiązaniem mogłoby być zastąpienie zer w y_filter na wartosc rozna od zera, lub lapanie wyjatkow
    """
    
    plt.subplot(2,4,1)
    plt.imshow(im, cmap='gray')
    plt.axis('off')
    plt.title("Given image in gray scale")
    
    plt.subplot(2,4,5)
    plt.imshow(x_filter, cmap='gray')
    plt.axis('off')
    plt.title("X axis filtration")
    
    plt.subplot(2,4,6)
    plt.imshow(y_filter, cmap='gray')
    plt.axis('off')
    plt.title("Y axis filtration")
    
    plt.subplot(2,4,7)
    plt.imshow(magnitude, cmap='magma')
    plt.axis('off')
    plt.title("magnitude")
    
    plt.subplot(2,4,8)
    plt.imshow(angle, cmap='gray')
    plt.axis('off')
    plt.title("angle")
    
edge_detection_function("Xray-300x247.jpg")