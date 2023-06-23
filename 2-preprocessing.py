import numpy as np
from skimage import util
import cv2
import os
import time
import pandas as pd
import matplotlib.pyplot as plt


def trim_edges(img, percentage_left, percentage_right, percentage_top, percentage_bottom):
    # Remove % of the 4 sides of the image.
    width = img.shape[1]
    height = img.shape[0]
    left = round(width * percentage_left)
    right = round(width * percentage_right)
    top = round(height * percentage_top)
    bottom = round(height * percentage_bottom)
    trimmed_image = img[top:bottom, left:right]
    return trimmed_image


def resize_image(img, scale_percent):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    return resized


def remove_noise(img):
    # Removes noise and annotations and leaves only the breast
    norm = util.img_as_ubyte((img - img.min()) / (img.max() - img.min()))  # normalize the image
    gray = cv2.cvtColor(norm, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # get the contours in the image
    max_area = max(contours, key=cv2.contourArea)  # Select the biggest contour
    # generate a mask with the largest area
    mask = np.zeros_like(img)  # generate a mask with the biggest contour
    cv2.drawContours(mask, [max_area], 0, (255, 255, 255), -1)
    breast_img = cv2.bitwise_and(img, mask)  # Extract whats inside the contour
    return breast_img


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


def erosion(img):
    kernel = np.ones((5, 5), np.uint8)
    erosion = cv2.erode(img, kernel, iterations=1)
    dilation = cv2.dilate(erosion, kernel, iterations = 1)
    return dilation


def gammaCorrection(src, gamma):
    invGamma = 1 / gamma

    table = [((i / 255) ** invGamma) * 255 for i in range(256)]
    table = np.array(table, np.uint8)

    return cv2.LUT(src, table)


def select_largest_obj(img_bin, lab_val=255, fill_holes=False, 
                       smooth_boundary=False, kernel_size=15):
    '''Select the largest object from a binary image and optionally
    fill holes inside it and smooth its boundary.
    Args:
        img_bin(2D array): 2D numpy array of binary image.
        lab_val([int]): integer value used for the label of the largest 
                        object. Default is 255.
        fill_holes([boolean]): whether fill the holes inside the largest 
                               object or not. Default is false.
        smooth_boundary([boolean]): whether smooth the boundary of the 
                                    largest object using morphological 
                                    opening or not. Default is false.
        kernel_size([int]): the size of the kernel used for morphological 
                            operation.
    '''
    n_labels, img_labeled, lab_stats, _ = cv2.connectedComponentsWithStats(
        img_bin, connectivity=8, ltype=cv2.CV_32S)
    largest_obj_lab = np.argmax(lab_stats[1:, 4]) + 1
    largest_mask = np.zeros(img_bin.shape, dtype=np.uint8)
    largest_mask[img_labeled == largest_obj_lab] = lab_val
    if fill_holes:
        bkg_locs = np.where(img_labeled == 0)
        bkg_seed = (bkg_locs[0][0], bkg_locs[1][0])
        img_floodfill = largest_mask.copy()
        h_, w_ = largest_mask.shape
        mask_ = np.zeros((h_ + 2, w_ + 2), dtype=np.uint8)
        cv2.floodFill(img_floodfill, mask_, seedPoint=bkg_seed, newVal=lab_val)
        holes_mask = cv2.bitwise_not(img_floodfill)  # mask of the holes.
        largest_mask = largest_mask + holes_mask
    if smooth_boundary:
        kernel_ = np.ones((kernel_size, kernel_size), dtype=np.uint8)
        largest_mask = cv2.morphologyEx(largest_mask, cv2.MORPH_OPEN, kernel_)
        
    return largest_mask

def remove_pectoral_muscle(mammo_breast_org):
    mammo_breast_org = cv2.cvtColor(mammo_breast_org, cv2.COLOR_BGR2GRAY)
    global_threshold = 18
    mammo_med_blurred = cv2.medianBlur(mammo_breast_org, 3)
    _, mammo_binary = cv2.threshold(mammo_med_blurred, global_threshold, 
                                    maxval=255, type=cv2.THRESH_BINARY)
    
    mammo_breast_equ = cv2.equalizeHist(mammo_breast_org)
    pect_high_inten_thres = 200  # <<= para to tune!
    _, pect_binary_thres = cv2.threshold(mammo_breast_equ, pect_high_inten_thres, 
                                        maxval=255, type=cv2.THRESH_BINARY)
    mammo_breast_mask = select_largest_obj(mammo_binary, lab_val=255, fill_holes=False, smooth_boundary=True, kernel_size=15)

    # Markers image for watershed algo.
    pect_marker_img = np.zeros(pect_binary_thres.shape, dtype=np.int32)
    # Sure foreground.
    pect_mask_init = select_largest_obj(pect_binary_thres, lab_val=255, 
                                        fill_holes=False, smooth_boundary=False)
    kernel_ = np.ones((3, 3), dtype=np.uint8)  # <<= para to tune!
    n_erosions = 7  # <<= para to tune!
    pect_mask_eroded = cv2.erode(pect_mask_init, kernel_, iterations=n_erosions)
    pect_marker_img[pect_mask_eroded > 0] = 255
    # Sure background - breast.
    n_dilations = 7  # <<= para to tune!
    pect_mask_dilated = cv2.dilate(pect_mask_init, kernel_, iterations=n_dilations)
    pect_marker_img[pect_mask_dilated == 0] = 128
    # Sure background - background.
    pect_marker_img[mammo_breast_mask == 0] = 64

    mammo_breast_equ_3c = cv2.cvtColor(mammo_breast_equ, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mammo_breast_equ_3c, pect_marker_img)
    pect_mask_watershed = pect_marker_img.copy()
    mammo_breast_equ_3c[pect_mask_watershed == -1] = (0, 0, 255)
    pect_mask_watershed[pect_mask_watershed == -1] = 0

    breast_only_mask = pect_mask_watershed.astype(np.uint8)
    breast_only_mask[breast_only_mask != 128] = 0
    breast_only_mask[breast_only_mask == 128] = 255
    kn_size = 25  # <<= para to tune!
    kernel_ = np.ones((kn_size, kn_size), dtype=np.uint8)
    breast_only_mask_smo = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN, kernel_)
    mammo_breast_only = cv2.bitwise_and(mammo_breast_org, breast_only_mask_smo)
    return mammo_breast_only


st = time.time()
input_dir = 'C:/Users/CAD6/Documents/TFM/2-PNG Images'
output_dir = 'C:/Users/CAD6/Documents/TFM/3-Preprocessed Images'


images_to_delete = [
'Calc-Test_P_00579_LEFT_MLO_FULL.png',
'Calc-Test_P_01713_RIGHT_MLO_FULL.png',
'Calc-Training_P_00321_LEFT_CC_FULL.png',
'Calc-Training_P_00321_LEFT_MLO_FULL.png',
'Calc-Training_P_00474_LEFT_MLO_FULL.png',
'Calc-Training_P_00631_RIGHT_CC_FULL.png',
'Calc-Training_P_00846_LEFT_CC_FULL.png',
'Calc-Training_P_01128_RIGHT_MLO_FULL.png',
'Calc-Training_P_01345_LEFT_CC_FULL.png',
'Calc-Training_P_01458_LEFT_MLO_FULL.png',
'Calc-Training_P_01497_LEFT_CC_FULL.png',
'Calc-Training_P_01497_RIGHT_MLO_FULL.png',
'Calc-Training_P_01647_RIGHT_CC_FULL.png',
'Calc-Training_P_01770_LEFT_MLO_FULL.png',
'Calc-Training_P_02559_RIGHT_MLO_FULL.png',
'Mass-Test_P_00126_RIGHT_CC_FULL.png',
'Mass-Test_P_00470_RIGHT_CC_FULL.png',
'Mass-Test_P_00470_RIGHT_MLO_FULL.png',
'Mass-Test_P_01661_LEFT_MLO_FULL.png',
'Mass-Training_P_00247_RIGHT_CC_FULL.png',
'Mass-Training_P_00247_RIGHT_MLO_FULL.png',
'Mass-Training_P_00248_LEFT_MLO_FULL.png',
'Mass-Training_P_00444_LEFT_CC_FULL.png',
'Mass-Training_P_00526_RIGHT_CC_FULL.png',
'Mass-Training_P_00543_RIGHT_MLO_FULL.png',
'Mass-Training_P_00545_LEFT_MLO_FULL.png',
'Mass-Training_P_00637_RIGHT_CC_FULL.png',
'Mass-Training_P_00665_LEFT_MLO_FULL.png',
'Mass-Training_P_01233_LEFT_MLO_FULL.png',
'Mass-Training_P_01350_RIGHT_MLO_FULL.png',
'Mass-Training_P_01354_RIGHT_MLO_FULL.png',
'Mass-Training_P_01355_LEFT_MLO_FULL.png',
'Mass-Training_P_01427_LEFT_MLO_FULL.png',
'Mass-Training_P_01517_LEFT_MLO_FULL.png',
]

images_to_crop_horizontal = [
'Calc-Test_P_00299_RIGHT_CC_FULL.png',
'Calc-Test_P_01752_LEFT_CC_FULL.png',
'Calc-Training_P_00049_RIGHT_CC_FULL.png',
'Calc-Training_P_00377_LEFT_MLO_FULL.png',
'Calc-Training_P_00647_RIGHT_CC_FULL.png',
'Calc-Training_P_00680_LEFT_CC_FULL.png',
'Calc-Training_P_00814_LEFT_CC_FULL.png',
'Calc-Training_P_00847_LEFT_CC_FULL.png',
'Calc-Training_P_01678_RIGHT_MLO_FULL.png',
'Calc-Training_P_01683_LEFT_CC_FULL.png',
'Mass-Test_P_00987_LEFT_CC_FULL.png',
'Mass-Test_P_01106_LEFT_MLO_FULL.png',
'Mass-Training_P_00055_LEFT_CC_FULL.png',
'Mass-Training_P_00298_LEFT_CC_FULL.png',
'Mass-Training_P_00675_LEFT_CC_FULL.png',
'Mass-Training_P_00675_LEFT_MLO_FULL.png',
'Mass-Training_P_00687_LEFT_CC_FULL.png',
'Mass-Training_P_00687_LEFT_MLO_FULL.png',
'Mass-Training_P_00859_LEFT_CC_FULL.png']

images_to_crop_vertical = [
'Mass-Training_P_00348_LEFT_CC_FULL.png',
'Mass-Training_P_00348_LEFT_MLO_FULL.png',]


res = os.listdir(input_dir)
for file in res:
    if 'FULL' in file:
        img = cv2.imread(input_dir + '/' + file)
        img = trim_edges(img, 0.05, 0.95, 0.05, 0.95)
        img = resize_image(img, 25)
        img = remove_noise(img)
        img = remove_pectoral_muscle(img)
        img = apply_clahe(img, 5, (5, 5))
        cv2.imwrite(output_dir + '/' + file, img)
    if 'MASK' in file:
        img = cv2.imread(input_dir + '/' + file)
        img = trim_edges(img, 0.05, 0.95, 0.05, 0.95)
        img = resize_image(img, 25)
        cv2.imwrite(output_dir + '/' + file, img)

res = os.listdir(output_dir)
for file in res:
    for image in images_to_delete:
        if image[0:-9] in file:
            os.remove(output_dir + '/' + file)
    for image in images_to_crop_horizontal:
        if image[0:-9] in file:
            img = cv2.imread(output_dir + '/' + file)
            img = trim_edges(img, 0.05, 0.95, 0, 1)
            cv2.imwrite(output_dir + '/' + file, img)
    for image in images_to_crop_vertical:
        if image[0:-9] in file:
            img = cv2.imread(output_dir + '/' + file)
            img = trim_edges(img, 0, 1, 0, 0.95)
            cv2.imwrite(output_dir + '/' + file, img)

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')
