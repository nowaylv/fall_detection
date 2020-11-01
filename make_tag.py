import json
import os
import cv2 as cv
import sys

class make_txt():
    def __init__(self, suffix = False, root = f"./falldata/labels/train", name = "demo"):
        if suffix == False:
            self.file = open(f"{root}/{name}.txt", mode = "w")
        else:
            self.file = open(f"{root}/{name}", mode = "w")
    
    def write(self, text):
        for data in text:
            for number in data:
                if isinstance(number, type('fdaa')):
                    self.file.write(number + " ")
                else:
                    self.file.write(str(number) + " ")
            self.file.write('\n')
    
    def add_head(self, *info):
        self.file.write("introduction\n")
        for data in info:
            self.file.write(data + " ")
        self.file.write('\n')

tag_root = f'C:/Users/18459/Desktop/ai/fall_dataset/lihongan/outputs'
data_root = f'C:/Users/18459/Desktop/ai/fall_dataset/lihongan/data'

cls_list = ['stand', 'fall']
color_list = [(255,0,0), (0,0,255)]

# te = [[1,2.42,3.42],[0,5.42,6.42]]

# make = make_txt(name = 'fa')
# make.write(te)
# exit()


for d in os.listdir(tag_root):  
    pause = True
    
    # f = open(f'{tag_root}/{d}', encoding = 'utf-8')
    with open(f'{tag_root}/{d}', encoding = 'utf-8') as f:
        python_data = json.load(f)
        pic_w = python_data['size']['width']
        pic_h = python_data['size']['height']
        img_root = f'{data_root}/{d[:-4]}jpg'
        info_list = python_data['outputs']['object']
        target = []
        for idx in range(len(info_list)):
            frame = info_list[idx]
            if frame['name'] == '跌倒':
                cls = 1
            else:
                cls = 0

            xmin = frame['bndbox']['xmin']
            ymin = frame['bndbox']['ymin']
            xmax = frame['bndbox']['xmax']
            ymax = frame['bndbox']['ymax']
            # if python_data['size']['width'] > 1000 or python_data['size']['height'] > 1000:
            # xmin = int(xmin * 0.16)
            # ymin = int(ymin * 0.16)
            # xmax = int(xmax * 0.16)
            # ymax = int(ymax * 0.16)
            cx = ((xmin + xmax) / 2 - 1) / pic_w
            cy = ((ymin + ymax) / 2 - 1) / pic_h
            w = (xmax - xmin) / pic_w
            h = (ymax - ymin) / pic_h
            # print(frame['bndbox']['size']['width'])
            # print(frame['bndbox']['size']['height'])

            # img = cv.imread(img_root)
            # if python_data['size']['width'] > 1000 or python_data['size']['height'] > 1000:
            #     img = cv.resize(img, (640, 480), interpolation = cv.INTER_AREA)
            # cv.imwrite(img, f'{data_root}/python_data['name']')
            # cv.rectangle(img, (xmin, ymin), (xmax, ymax), color_list[cls], thickness = 2)
            # cv.putText(img, cls_list[cls], (xmin, (ymin - 20)), cv.FONT_ITALIC, 1, 
            #     (color_list[cls]), 1, lineType = cv.LINE_AA)
            target.append([cls, '%.6f'%cx, '%.6f'%cy, '%.6f'%w, '%.6f'%h])
        make = make_txt(name = d[:-5])
        make.write(target)
        # cv.imshow('fa', img)
        # print(python_data)
        # while pause:
        #     key = cv.waitKey(1) & 0xff
        #     if key == ord('n'):
        #         pause = False
            
                

        


