import os
import shutil
import time
import glob

def delete_files(path):
    files = glob.glob(path)
    for f in files:
        os.remove(f)


st = time.time()
input_dir = 'C:/Users/CAD6/Documents/TFM/4-Augmented Images'
output_dir_train = 'C:/Users/CAD6/Documents/TFM/YOLO/images/train'
output_dir_val = 'C:/Users/CAD6/Documents/TFM/YOLO/images/val'
delete_files(output_dir_train + '/*')
delete_files(output_dir_val + '/*')
res = os.listdir(input_dir)
for file in res:
    if 'FULL' in file:
        if 'Test' in file:
            shutil.copy(input_dir + '/' + file, output_dir_val + '/' + file)
        elif 'Training' in file:
            shutil.copy(input_dir + '/' + file, output_dir_train + '/' + file)
input_dir = 'C:/Users/CAD6/Documents/TFM/5-Txt files'
output_dir_train = 'C:/Users/CAD6/Documents/TFM/YOLO/labels/train'
output_dir_val = 'C:/Users/CAD6/Documents/TFM/YOLO/labels/val'
delete_files(output_dir_train + '/*')
delete_files(output_dir_val + '/*')
res = os.listdir(input_dir)
for file in res:
    if 'Test' in file:
        shutil.copy(input_dir + '/' + file, output_dir_val + '/' + file)
    elif 'Training' in file:
        shutil.copy(input_dir + '/' + file, output_dir_train + '/' + file)
et = time.time()
elapsed_time = et - st
print('Execution time:', elapsed_time, 'seconds')