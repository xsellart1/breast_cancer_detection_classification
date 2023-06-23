import cv2
import time
import os
import os.path

def generate_mask(img):
    img_grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x, y, w, h = cv2.boundingRect(img_grey)
    max_w = img_grey.shape[1]
    max_h = img_grey.shape[0]
    x_center_norm = (x + w/2) / max_w
    y_center_norm = (y + h/2) / max_h
    w_norm = w / max_w
    h_norm = w / max_h
    return (x_center_norm, y_center_norm, w_norm, h_norm)


st = time.time()
input_dir = 'C:/Users/CAD6/Documents/TFM/4-Augmented Images'
output_dir = 'C:/Users/CAD6/Documents/TFM/5-Txt files'

res = os.listdir(input_dir)
for file in res:
    if 'MASK' in file:
        if 'Mass' in file:
            img = cv2.imread(input_dir + '/' + file)
            x_center_norm, y_center_norm, w_norm, h_norm = generate_mask(img)
            if os.path.isfile(output_dir + '/' + file[0:-11] + '_FULL' + '.txt'):
                file = open(output_dir + '/' + file[0:-11] + '_FULL' + '.txt', "a")
                file.write("\n")
                file.write("0 " + str(x_center_norm) + ' ' + str(y_center_norm) + ' ' + str(w_norm) + ' ' + str(h_norm))
                file.close()
            else:
                with open(output_dir + '/' + file[0:-11] + '_FULL' + '.txt', 'w') as f:
                    f.write('0 ' + str(x_center_norm) + ' ' + str(y_center_norm) + ' ' + str(w_norm) + ' ' + str(h_norm))
        elif 'Calc' in file:
            img = cv2.imread(input_dir + '/' + file)
            x_center_norm, y_center_norm, w_norm, h_norm = generate_mask(img)
            if os.path.isfile(output_dir + '/' + file[0:-11] + '_FULL' + '.txt'):
                file = open(output_dir + '/' + file[0:-11] + '_FULL' + '.txt', "a")
                file.write("\n")
                file.write("0 " + str(x_center_norm) + ' ' + str(y_center_norm) + ' ' + str(w_norm) + ' ' + str(h_norm))
                file.close()
            else:
                with open(output_dir + '/' + file[0:-11] + '_FULL' + '.txt', 'w') as f:
                    f.write('0 ' + str(x_center_norm) + ' ' + str(y_center_norm) + ' ' + str(w_norm) + ' ' + str(h_norm))
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')