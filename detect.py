import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from numpy import random
import cv2

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel


class Detector():
    def __init__(self):
        self.source = "./upload/image.jpg"
        self.device = ''
        self.weights = "./weights.pt"
        self.imgsz = cv2.imread(self.source).shape[0]
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

            detected_objects = []
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
                        info = [self.names[int(cls)], xywh, '%.2f' % conf]
                        detected_objects.append(info)

            return detected_objects
