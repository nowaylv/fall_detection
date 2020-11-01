import json
import os
import cv2 as cv
import sys
tag_root = f'C:/Users/18459/Desktop/ai/fall_dataset/lihongan/outputs'
data_root = f'C:/Users/18459/Desktop/ai/fall_dataset/lihongan/data'

cls_list = ['stand', 'fall']
color_list = [(255,0,0), (0,0,255)]

def resize_img():
    for d in os.listdir(tag_root):  
        pause = True
        # f = open(f'{tag_root}/{d}', encoding = 'utf-8')
        with open(f'{tag_root}/{d}', encoding = 'utf-8') as f:
            python_data = json.load(f)
            img_root = f'{data_root}/{d[:-4]}jpg'
            info_list = python_data['outputs']['object']
            target = []
            for frame in info_list:
                if frame['name'] == '跌倒':
                    cls = 1
                else:
                    cls = 0

                xmin = frame['bndbox']['xmin']
                ymin = frame['bndbox']['ymin']
                xmax = frame['bndbox']['xmax']
                ymax = frame['bndbox']['ymax']
                if python_data['size']['width'] > 1000 or python_data['size']['height'] > 1000:
                    xmin = int(xmin * 0.16)
                    ymin = int(ymin * 0.16)
                    xmax = int(xmax * 0.16)
                    ymax = int(ymax * 0.16)
                # print(frame['bndbox']['size']['width'])
                # print(frame['bndbox']['size']['height'])

                img = cv.imread(img_root)
                if python_data['size']['width'] > 1000 or python_data['size']['height'] > 1000:
                    resize_img = cv.resize(img, (640, 480), interpolation = cv.INTER_AREA)
                    cv.imwrite(img_root, resize_img)
                # cv.rectangle(img, (xmin, ymin), (xmax, ymax), color_list[cls], thickness = 2)
                # cv.putText(img, cls_list[cls], (xmin, (ymin - 20)), cv.FONT_ITALIC, 1, 
                #     (color_list[cls]), 1, lineType = cv.LINE_AA)
                # target.append([cls, xmin, ymin, xmax, ymax])
            # cv.imshow('fa', img)
            # print(python_data)
            # while pause:
            #     key = cv.waitKey(1) & 0xff
            #     if key == ord('n'):
            #         pause = False
    print("over")
            
if __name__ == '__main__':
    
                

        


