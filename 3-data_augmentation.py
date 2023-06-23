import cv2
import os
import time
import torchvision.transforms as transforms
import matplotlib.image as mpimg
from PIL import Image, ImageEnhance
import cv2
import numpy as np
import imutils


def apply_clahe(img, clip, tile):
    # Applies contrast limited adaptive histogram equalization to the image
    b, g, r = cv2.split(img)
    # The declaration of CLAHE
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)  # clipLimit -> Threshold for contrast limiting
    # Apply CLAHE to each channel
    b = clahe.apply(b)
    g = clahe.apply(g)
    r = clahe.apply(r)
    return cv2.merge([b, g, r])


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)

def erosion(img):
    kernel = np.ones((3, 3), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    #dilation = cv2.dilate(erosion, kernel, iterations = 1)
    return erosion


def jitter(img, b=0.2, c=0.2, s=0.2, h=0.1):
    """
    Randomly alter brightness, contrast, saturation, hue within given range
    """
    
    img = Image.fromarray(img)
    
    transform = transforms.ColorJitter(
    brightness=b, contrast=c, saturation=s, hue=h)
  
    # apply transform
    img = transform(img)
    
    return img


st = time.time()
# folder path were the downloaded dataset is stored
input_dir = 'C:/Users/CAD6/Documents/TFM/3-Preprocessed Images'
# folder path were the converted images will be stored
output_dir = 'C:/Users/CAD6/Documents/TFM/4-Augmented Images'

res = os.listdir(input_dir)
for file in res:
    img = cv2.imread(input_dir + '/' + file)
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-0.png'), img)
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-1.png'), cv2.flip(img, 1))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-2.png'), cv2.flip(img, 0))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-3.png'), cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-4.png'), cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-5.png'), cv2.rotate(img, cv2.ROTATE_180))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-6.png'), cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 0))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-7.png'), cv2.flip(cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE), 1))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-8.png'), gammaCorrection(img,1.5))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-9.png'), gammaCorrection(img,0.5))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-a.png'), img[0:round(img.shape[0]*0.8), 0:round(img.shape[1]*0.8)])
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-b.png'), img[-round(img.shape[0]*0.8):, -round(img.shape[1]*0.8):])
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-c.png'), imutils.rotate_bound(img, 30))
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-d.png'), imutils.rotate_bound(img, -30))
    if "FULL" in file:
        cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-e.png'), apply_clahe(img, 3, (5, 5)))
    else:
        cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-e.png'),img)
    cv2.imwrite(str(output_dir) + '/' + str(file[0:-4] + '-f.png'), imutils.translate(img, -200, 500))

    


et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')

