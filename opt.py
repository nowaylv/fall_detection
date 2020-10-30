
import torch
from models.experimental import attempt_load
from utils.torch_utils import prune
from utils.general import kmean_anchors

if __name__ == '__main__':
    # weight = attempt_load('./yolov5s.pt', map_location = 'cpu')
    # prune(weight)
    kmean_anchors(path='./data/fall.yaml', n=9, img_size=640, thr=4.0, gen=1000, verbose=True)