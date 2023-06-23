import numpy as np
from PIL import Image, ImageFont, ImageDraw
from PIL.ImageChops import add, subtract, multiply, difference, screen
import PIL.ImageStat as stat
from skimage.io import imread, imsave, imshow, show, imread_collection, imshow_collection
from skimage import color, exposure, img_as_float, data
from skimage.transform import SimilarityTransform, warp, swirl
from skimage.util import invert, random_noise, montage
import matplotlib.image as mpimg
import matplotlib.pylab as plt
from scipy.ndimage import affine_transform, zoom
from skimage import io
from skimage import feature
from matplotlib import pylab as pylab
from skimage import filters
from skimage.transform import hough_line, hough_line_peaks
from skimage.draw import polygon
import os
import time
import cv2


def right_orient_mammogram(img):
    orientation = 'left'
    if np.sum(img[:, 0:int(img.shape[1]/2)]) <= np.sum(img[:, int(img.shape[1]/2):]):
        img = cv2.flip(img, 1)
        orientation = 'right'
    return img, orientation


def shortlist_lines(lines):
    MIN_ANGLE = 10
    MAX_ANGLE = 80
    MIN_DIST  = 5
    MAX_DIST  = 120000
    
    shortlisted_lines = [x for x in lines if 
                          (x['dist']>=MIN_DIST) &
                          (x['dist']<=MAX_DIST) &
                          (x['angle']>=MIN_ANGLE) &
                          (x['angle']<=MAX_ANGLE)
                        ]        
    return shortlisted_lines


def remove_pectoral(shortlisted_lines):
    shortlisted_lines.sort(key = lambda x: x['dist'])
    try:
        pectoral_line = shortlisted_lines[0]
    except IndexError:
        return
    d = pectoral_line['dist']
    theta = np.radians(pectoral_line['angle'])
    
    x_intercept = d/np.cos(theta)
    y_intercept = d/np.sin(theta)
    
    return polygon([0, 0, y_intercept], [0, x_intercept, 0])


def pectoral_muscle_removal(image_path, sigma, output_path):

    # Code for applying transformation like Hough Transform 
    # or some other transform
    img = io.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img, orientation = right_orient_mammogram(img)
    edge = feature.canny(img,sigma)
    
    #sobel filtering on canny edge image
    edge = filters.sobel(edge)

    #hough transform
    h, theta, d = hough_line(edge)
    lines = list()
    #hough transform for edge line detections
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        x1 = 0
        x2 = img.shape[1]
        if np.sin(angle) == 0:
            y1 = 0
            y2 = 0
        else:
            y1 = (dist - x1 * np.cos(angle)) / np.sin(angle)
            y2 = (dist - x2 * np.cos(angle)) / np.sin(angle)
        lines.append({
            'dist': dist,
            'angle': np.degrees(angle),
            'point1': [x1, y1],
            'point2': [x2, y2]
        })
        #print("Angle: {:.2f}, Dist: {:.2f}".format(np.degrees(angle), dist))
    
    poly = remove_pectoral(shortlist_lines(lines))
    
    try:
        rr, cc = poly
    except TypeError:
        if orientation == 'right':
            img = cv2.flip(img, 1)
        cv2.imwrite(output_path, img)
    
    try:
        img[rr, cc] = 0
        if orientation == 'right':
            img = cv2.flip(img, 1)
        cv2.imwrite(output_path, img)
    except IndexError:
        if orientation == 'right':
            img = cv2.flip(img, 1)
        cv2.imwrite(output_path, img)
    except UnboundLocalError:
        if orientation == 'right':
            img = cv2.flip(img, 1)
        cv2.imwrite(output_path, img)

input_dir = 'C:/Users/CAD6/Documents/TFM/3-Preprocessed Images/1-Remove muscle/0-Muscle'
output_dir= 'C:/Users/CAD6/Documents/TFM/3-Preprocessed Images/1-Remove muscle/1-Hough'
res = os.listdir(input_dir)
for file in res:
    pectoral_muscle_removal(input_dir + '/' + file, 1, output_dir + '/' + file)