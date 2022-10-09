import cv2
import matplotlib.pyplot as plt
import numpy as np
import os

path_to_images = '/Users/aungriah/Documents/eval/bboxes'
save_dir = '/Users/aungriah/Documents/eval/test.txt'
files = os.listdir(path_to_images)
files.sort()

with open(save_dir, 'w') as f:
    for file in files:
        path = os.path.join('eval', 'images', file[:-3] + 'jpg')
        f.write(path + '\n')
f.close()