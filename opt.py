
import torch
from models.experimental import attempt_load
from utils.torch_utils import prune

if __name__ == '__main__':
    weight = attempt_load('./yolov5s.pt', map_location = 'cpu')
    prune(weight)