import pydicom
from os import walk
import cv2
import time
import numpy as np


st = time.time()
# folder path were the downloaded dataset is stored
input_path = 'C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM'
# folder path were the converted images will be stored
output_path = 'C:/Users/CAD6/Documents/TFM/2-PNG Images'

for (dir_path, dir_names, file_names) in walk(input_path):
    # Convert images of mammograms
    if 'full mammogram' in str(dir_path):
        for file in file_names:
            ds = pydicom.read_file(str(dir_path) + '/' + str(file))
            img = ds.pixel_array
            cv2.imwrite(output_path + '/' + dir_path.split('\\')[1] + '_FULL.png', img)
    elif 'ROI mask images' in str(dir_path):
        for file in file_names:
            ds = pydicom.read_file(str(dir_path) + '/' + str(file))
            img = ds.pixel_array
            if len(np.unique(img)) == 2:  # If it is the picture of the mask
                cv2.imwrite(output_path + '/' + dir_path.split('\\')[1] + '_MASK.png', img)
    elif 'cropped images' in str(dir_path):
        for file in file_names:
            ds = pydicom.read_file(str(dir_path) + '/' + str(file))
            img = ds.pixel_array
            cv2.imwrite(output_path + '/' + dir_path.split('\\')[1] + '_MASK.png', img)
            if len(np.unique(img)) == 2:  # If it is the picture of the mask
                cv2.imwrite(output_path + '/' + dir_path.split('\\')[1] + '_MASK.png', img)

img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Calc-Training_P_00005_RIGHT_CC_1/08-30-2017-DDSM-NA-09081/1.000000-cropped images-94682/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Calc-Training_P_00005_RIGHT_CC_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Calc-Training_P_00474_LEFT_MLO_1/09-06-2017-DDSM-NA-12715/1.000000-ROI mask images-91565/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Calc-Training_P_00474_LEFT_MLO_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Mass-Test_P_00016_LEFT_CC_1/10-04-2016-DDSM-NA-09887/1.000000-cropped images-26184/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Mass-Test_P_00016_LEFT_CC_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Mass-Test_P_00016_LEFT_MLO_1/10-04-2016-DDSM-NA-15563/1.000000-cropped images-77287/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Mass-Test_P_00016_LEFT_MLO_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Mass-Test_P_00145_LEFT_CC_1/09-27-2017-DDSM-NA-31267/1.000000-ROI mask images-45741/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Mass-Test_P_00145_LEFT_CC_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Mass-Test_P_00192_RIGHT_CC_1/10-04-2016-DDSM-NA-43116/1.000000-ROI mask images-84137/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Mass-Test_P_00192_RIGHT_CC_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Mass-Test_P_00381_LEFT_MLO_1/10-04-2016-DDSM-NA-32671/1.000000-ROI mask images-02900/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Mass-Test_P_00381_LEFT_MLO_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Mass-Test_P_00922_RIGHT_MLO_1/10-04-2016-DDSM-NA-30078/1.000000-ROI mask images-00265/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Mass-Test_P_00922_RIGHT_MLO_1_MASK.png', img)
img = pydicom.read_file('C:/Users/CAD6/Documents/TFM/1-Original Images/CBIS-DDSM/Mass-Test_P_01595_LEFT_CC_1/10-04-2016-DDSM-NA-42023/1.000000-ROI mask images-93832/1-1.dcm').pixel_array
cv2.imwrite('C:/Users/CAD6/Documents\TFM/2-PNG Images/Mass-Test_P_01595_LEFT_CC_1_MASK.png', img)

et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')