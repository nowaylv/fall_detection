from flask import Flask, request
import io
from PIL import Image
import cv2 as cv
import torch
from torchvision import transforms
from detect import detect

import argparse
import os
import shutil
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import numpy as np

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)
from utils.torch_utils import select_device, load_classifier, time_synchronized, prune, sparsity

trans = transforms.ToTensor()
device = "cuda:0" if torch.cuda.is_available() else "cpu"

cls_list = ['stand', 'fall']
app = Flask(__name__)
imgsz = 640
half = device != 'cpu'
model = attempt_load(f'./runs/exp0/weights/best.pt', map_location=device)
if half:
    model.half()

cudnn.benchmark = True  # set True to speed up constant image size inference
dataset = LoadStreams('0', img_size=imgsz)

img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
_ = model(img.half() if half else img) if device != 'cpu' else None  # run once


@app.route('/')
def test():
    return "主页"

#method大写
@app.route('/xxx', methods = ['POST', 'GET'])
def load_file():
    image = request.files.get('file')
    name = request.form.get("name1")
    image_bin = image.read()
    img = io.BytesIO(image_bin)
    data = Image.open(img)
    max_l = max(data.size)
    h = data.size[0]
    w = data.size[1]
    img_np = np.array(data)
    
    img_np = letterbox(img_np, new_shape=640)[0]
    img_np = img_np[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img_np = np.ascontiguousarray(img_np)
    img_np = img_np / 255.0

    img_t = torch.from_numpy(img_np)
    img_t = img_t.half() if half else img_t.float()  # uint8 to fp16/32
    # img_np /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img_t.ndimension() == 3:
        img_t = img_t.unsqueeze(0)

    pred = model(img_t, augment=False)[0]

    # Apply NMS
    #--------------------------------conf--iou--------------------------------
    pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=True)
    pred = torch.stack(pred, dim = 0)
    if pred[:, :, -1].sum() > 0.5:
        return 'fall'
    else:
        return 'stand'

if __name__ == "__main__":
    #指定端口
    app.run(port = 5001)
