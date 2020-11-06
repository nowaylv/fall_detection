import cv2
import sys
import argparse     
import time

from Processer import Processor
name_list = ['stand', 'fall']
color_list = [(0, 255, 0), (0, 0, 255)]
#from Visualizer import Visualizer

def cli():
    desc = 'Run TensorRT fall visualizer'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-model', help='trt engine file located in ./models', required=False)
    parser.add_argument('-image', help='image file path', required=False)
    args = parser.parse_args()
    model = args.model or 'best.trt'
    img = args.image or 'no.jpg'
    return { 'model': model, 'image': img }

def main():
    # parse arguments
    args = cli()

    # setup processor and visualizer
    processor = Processor(model=args['model'])
    cap = cv2.VideoCapture(0)
    # img = cv2.imread(f'../no.jpg')
    
    while True:
        t_start = time.time()
        ret, img = cap.read()
        img = img[:, ::-1, :]
        #print(img.shape,type(img))
        # inference
        output = processor.detect(img) 
        img = cv2.resize(img, (640, 480))
        
        boxes, confs, classes = processor.post_process(output)
        fps = 1 / (time.time() - t_start)
        #print('fps:', fps)
        #print(boxes, confs, classes)
        if len(boxes) != 0:
            for (f ,conf, cls) in zip(boxes, confs, classes):
                x1 = int(f[0])
                y1 = int(f[1])
                x2 = int(f[2])
                y2 = int(f[3])
                #cls = int(f[-1])
                name = name_list[int(cls)]    
                cv2.rectangle(img, (x1, y1), (x2, y2), color_list[cls], thickness = 3)
                t_size = cv2.getTextSize(name, 0, fontScale=1, thickness=2)[0]
                cv2.rectangle(img, (x1-1, y1), (x1 + t_size[0] + 34, y1 - t_size[1]), color_list[cls], -1, cv2.LINE_AA)
                cv2.putText(img, name + str('%.2f'%conf), 
                    (x1, int(y1-2)), 0, 0.8, thickness=2,  
                    color = (255, 255, 255), lineType = cv2.LINE_AA)
                cv2.putText(img, 'FPS: ' + str('%.2f'%fps), (20, 30), 0, 1, thickness = 2, color = (255, 0, 0), lineType = cv2.LINE_AA)
        
        cv2.imshow("da", img)
        cv2.waitKey(1)
    # visualizer.draw_results(img, boxes, confs, classes)

if __name__ == '__main__':
    main()   
