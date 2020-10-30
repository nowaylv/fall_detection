import os
import numpy as np
from torch.utils.data import Dataset
import torch
import os
import glob
import re
import cv2 as cv
from tqdm import tqdm
import timen
import string 
import math

cls_list = ['stand', 'fall']
color_list = [(255,0,0), (0,0,255)]

tag_root = f'C:/Users/18459/Desktop/ai/fall_dataset/lian1/outputs'
data_root = f'C:/Users/18459/Desktop/ai/fall_dataset/img'
txt_root = f'./txt'

class get_regr_tag(Dataset):
    def __init__(self, *root, type = 'txt', col_start = 0, col_stop = 6, line_start = 2):
        if type == 'txt':
            tag_list = []
            for dir in root:
                text = open(dir).readlines()
                # print(text)
                #remove text head
                for sub_t in tqdm(text):
                    # print(sub_t)
                    sub_t = sub_t.strip().split()
                    # print(sub_t)
                    #get useful info
                    sub_t = sub_t[col_start:]
                    for i in range(len(sub_t)):
                        if sub_t[i].find('.') == -1:
                            sub_t[i] = np.int(sub_t[i])
                        else:
                            sub_t[i] = np.float(sub_t[i])
                    tag_list.append(sub_t)

            # print(tag_list)

            self.tag = np.array(tag_list)

    def __len__(self):
        return len(self.tag)

    def __getitem__(self, index):
        return self.tag[index]

for d in os.listdir(txt_root): 
    pause = True
    img_root = f'{data_root}/{d[:-4]}.jpg'
    get_txt = get_regr_tag(f'./txt/{d}')
    img = cv.imread(img_root)
    h = len(img)
    w = len(img[0])
    for info in get_txt:
        cls = int(info[0])
        print(info)
        print(w, h)
        cx = info[1] * w
        cy = info[2] * h
        fw = info[3] * w
        fh = info[4] * h
        xmin = int(cx - (fw / 2))
        ymin = int(cy - (fh / 2))
        xmax = int(cx + (fw / 2))
        ymax = int(cy + (fh / 2))
        cv.rectangle(img, (xmin, ymin), (xmax, ymax), color_list[cls], thickness = 2)
        cv.putText(img, cls_list[cls], (xmin, (ymin - 20)), cv.FONT_ITALIC, 1, 
            (color_list[cls]), 1, lineType = cv.LINE_AA)
    cv.imshow('fa', img)
    print(d)
    while pause:
        key = cv.waitKey(1) & 0xff
        if key == ord('n'):
            pause = False

    # print(get_txt[1][0])
nn