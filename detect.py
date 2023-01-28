import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random
import cv2
import numpy as np
import os

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from ocr import *

class Detector():
    def __init__(self):
        self.source = './upload'
        self.device = ''
        self.weights = "./weights.pt"
        self.imgsz = 1280
        self.conf_thres = 0.5
        self.iou_thres = 0.45
        self.agnostic_nms = False
        self.classes = None
        self.augment = False
        self.trace = False
        self.half = None
        self.model = None
        self.stride = None
        self.dataset = None
        self.names = None
    
    def load(self):
        # Initialize
        set_logging()
        self.device = select_device(self.device)
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride

        if self.half:
            self.model.half()  # to FP16

    def add_padding(self, path):
        # read image
        img = cv2.imread(path)
        old_image_height, old_image_width, channels = img.shape

        # create new image of desired size and color (blue) for padding
        new_image_width = old_image_width
        new_image_height = old_image_width
        color = (0,0,0)
        result = np.full((new_image_height,new_image_width, channels), color, dtype=np.uint8)

        # compute center offset
        x_center = (new_image_width - old_image_width) // 2
        y_center = (new_image_height - old_image_height) // 2

        # copy img image into center of result image
        result[y_center:y_center+old_image_height, 
            x_center:x_center+old_image_width] = img

        # save result
        cv2.imwrite(path, result)

    def downscale_image(self, path, resolution):
        image = cv2.imread(path)
        height, width = image.shape[:2]
        scaling_factor = resolution / max(height, width)
        cv2.imwrite(path, cv2.resize(image, None, fx=scaling_factor, fy=scaling_factor))

    def detect(self):
        self.imgsz = check_img_size(self.imgsz, s=self.stride)  # check img_size

        self.dataset = LoadImages(self.source, img_size=self.imgsz, stride=self.stride)
        # Get names and colors
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names

        # Run inference
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once
        old_img_w = old_img_h = self.imgsz
        old_img_b = 1

        t0 = time.time()
        images = []
        for path, img, im0s, vid_cap in self.dataset:
            img = torch.from_numpy(img).to(self.device)
            img = img.half() if self.half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            # Warmup
            if self.device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                old_img_b = img.shape[0]
                old_img_h = img.shape[2]
                old_img_w = img.shape[3]
                for i in range(3):
                    self.model(img, augment=self.augment)[0]

            # Inference
            t1 = time_synchronized()
            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = self.model(img, augment=self.augment)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)
            t3 = time_synchronized()

            detected_objects = {"path": os.path.basename(path), "objects": []}
            # Process detections
            for i, det in enumerate(pred):  # detections per image

                p, s, im0, frame = path, '', im0s, getattr(self.dataset, 'frame', 0)

                p = Path(p)  # to Path
                gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        
                        info = {"name": self.names[int(cls)], "points": xywh, "confidence": str(round(float('%.2f' % conf)*100)), "id": []}
                        
                        ## Convert x y w h from relative to absolute
                        img = cv2.imread(path)

                        c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
                        x, y, w, h = min(c1[0], c2[0]), min(c1[1], c2[1]), abs(c1[0]-c2[0]), abs(c1[1]-c2[1])
                        cropped_img = img[y:y+h, x:x+w]

                        name, extension = os.path.basename(path).split(".")[0], os.path.basename(path).split(".")[1] 
                        randomDigit = random.randint(10000, 99999)

                        info["randomDigit"] = randomDigit

                        new_path = os.path.join(os.path.dirname(__file__), "converted", name+str(randomDigit)+"."+extension)

                        cv2.imwrite(new_path, cropped_img)

                        info["id"] = []

                        try:
                            info["id"] = recognize(cropped_img)                   
                        except Exception as e:
                            print(e)
                            
                        detected_objects["objects"].append(info)
            
            images.append(detected_objects)

        return images
