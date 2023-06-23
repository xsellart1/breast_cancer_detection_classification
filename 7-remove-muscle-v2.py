import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def increase_brightness(img):
# load image
    # specify region
    #x,y,w,h = 0,0,328,530
    height = img.shape[0]
    width = img.shape[1]
    # crop region
    #crop = img[y:y+h, x:x+w]
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([[[0,0],[0,int(round(height/2))],[int(round(width/2)),0],[width,int(round(height/2))],[width,0],[0,0]]])
    cv2.fillPoly(mask, points, (255))
    gain = 1

    res = cv2.bitwise_and((gain * img.astype(np.float64)).clip(0,255).astype(np.uint8),(gain * img.astype(np.float64)).clip(0,255).astype(np.uint8),mask = mask)
    return cv2.add(res , img)

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

    # specify region
    #x,y,w,h = 0,0,328,530
    height = mammo_breast_org.shape[0]
    width = mammo_breast_org.shape[1]
    # crop region
    #crop = img[y:y+h, x:x+w]
    mask = np.zeros((height, width), dtype=np.uint8)
    points = np.array([[[0,0],[0,int(round(height/2))],[int(round(width/2)),0],[width,int(round(height/2))],[width,0],[0,0]]])
    cv2.fillPoly(mask, points, (255))
    gain = 1

    res = cv2.bitwise_and((gain * img.astype(np.float64)).clip(0,255).astype(np.uint8),(gain * img.astype(np.float64)).clip(0,255).astype(np.uint8),mask = mask)
    bright_image = cv2.add(res , img)



    global_threshold = 18

    mammo_med_blurred = cv2.medianBlur(bright_image, 3)
    _, mammo_binary = cv2.threshold(mammo_med_blurred, global_threshold, 
                                    maxval=255, type=cv2.THRESH_BINARY)

    mammo_breast_equ = cv2.equalizeHist(bright_image)
    pect_high_inten_thres = 210  # <<= para to tune!
    _, pect_binary_thres = cv2.threshold(mammo_breast_equ, pect_high_inten_thres, 
                                        maxval=255, type=cv2.THRESH_BINARY)
    #fig,axes = plt.subplots(1, 2)
    #fig.set_size_inches([12, 9])
    #res = hstack((mammo_med_blurred, mammo_binary))
    #axes[0].imshow(mammo_breast_equ, cmap='gray')
    #axes[1].imshow(pect_binary_thres, cmap='gray')
    #plt.show()

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
    # plot.
    #fig,ax = plt.subplots()
    #fig.set_size_inches([6, 9])
    #ax.imshow(pect_marker_img, cmap='gray')
    #plt.show()

    mammo_breast_equ_3c = cv2.cvtColor(mammo_breast_equ, cv2.COLOR_GRAY2BGR)
    cv2.watershed(mammo_breast_equ_3c, pect_marker_img)
    pect_mask_watershed = pect_marker_img.copy()
    mammo_breast_equ_3c[pect_mask_watershed == -1] = (0, 0, 255)
    pect_mask_watershed[pect_mask_watershed == -1] = 0
    #fig,axes = plt.subplots(1, 2)
    #fig.set_size_inches([12, 9])
    #axes[0].imshow(pect_mask_watershed, cmap='gray')
    #axes[1].imshow(mammo_breast_equ_3c)
    #plt.show()

    breast_only_mask = pect_mask_watershed.astype(np.uint8)
    breast_only_mask[breast_only_mask != 128] = 0
    breast_only_mask[breast_only_mask == 128] = 255
    kn_size = 25  # <<= para to tune!
    kernel_ = np.ones((kn_size, kn_size), dtype=np.uint8)
    breast_only_mask_smo = cv2.morphologyEx(breast_only_mask, cv2.MORPH_OPEN, kernel_)
    mammo_breast_only = cv2.bitwise_and(mammo_breast_org, breast_only_mask_smo)
    #fig,axes = plt.subplots(1, 4)
    #fig.set_size_inches([18, 9])
    #axes[0].imshow(mammo_breast_equ, cmap='gray')
    #axes[1].imshow(breast_only_mask_smo, cmap='gray')
    #axes[2].imshow(mammo_breast_only, cmap='gray')
    #axes[3].imshow(mammo_breast_org, cmap='gray')
    #plt.show()
    return mammo_breast_only


res = os.listdir('C:/Users/CAD6/Documents/TFM/3-Preprocessed Images')
for file in res:
    img = cv2.imread('C:/Users/CAD6/Documents/TFM/3-Preprocessed Images/' + file, cv2.IMREAD_GRAYSCALE)
    removed = remove_pectoral_muscle(img)
    cv2.imwrite('C:/Users/CAD6/Documents/TFM/3-Preprocessed Images/1-Remove muscle/' + file, removed)


