import onnxruntime
from torchvision import transforms

import cv2 as cv
from PIL import Image
import numpy as np
from utils.datasets import LoadStreams, LoadImages, letterbox
import torch
from utils.general import (
    check_img_size, non_max_suppression, apply_classifier, scale_coords,
    xyxy2xywh, plot_one_box, strip_optimizer, set_logging)

trans = transforms.ToTensor()
img = Image.open(f'./no.jpg')
img_l = trans(img).unsqueeze(0).numpy()
img_n = img_l[0].transpose(1, 2, 0)[:,:,::-1]   

# img_np = letterbox(img, new_shape=640)[0]
# # cv.imwrite(f'./no.jpg', img_np)
# img_np = img_np[:, :, ::-1].transpose(2, 0, 1)
# img_np = np.ascontiguousarray(img_np)
# img_np = img_np / 255.0
# img_t = torch.from_numpy(img_np).unsqueeze(0).numpy()
# img_t = img_t.astype(np.float32)

ort_session = onnxruntime.InferenceSession("./runs/exp0/weights/best.onnx")

ort_input = {ort_session.get_inputs()[0].name: img_l}
print(ort_session.get_inputs()[0].name)
ort_output = ort_session.run(None, ort_input)
img_out_y = ort_output[0]
print(img_out_y.shape),exit()
out_t = torch.from_numpy(img_out_y)
print(out_t.shape),exit()
pred = non_max_suppression(out_t, 0.25, 0.45, classes=None, agnostic=True)
# pred = torch.stack(pred, dim = 0)

print(pred)   