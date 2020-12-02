# ==============================================================================
# Copyright (C) 2018-2020 Intel Corporation
#
# SPDX-License-Identifier: MIT
# ==============================================================================

import sys
import gi
import cv2
import numpy as np
from math import exp as exp
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

from gstgva import VideoFrame

DETECT_THRESHOLD = 0.9
CLASS_NUMBER = 80
input_height = 416
input_width = 416
prob_threshold = 0.6
iou_threshold = 0.6


Gst.init(sys.argv)

labels_map = ["person", "bicycle", "car", "motorbike", "aeroplane",
          "bus", "train", "truck", "boat", "traffic light",
          "fire hydrant", "stop sign", "parking meter", "bench", "bird",
          "cat", "dog", "horse", "sheep", "cow",
          "elephant", "bear", "zebra", "giraffe", "backpack",
          "umbrella", "handbag", "tie", "suitcase", "frisbee",
          "skis", "snowboard", "sports ball", "kite", "baseball bat",
          "baseball glove", "skateboard", "surfboard","tennis racket", "bottle",
          "wine glass", "cup", "fork", "knife", "spoon",
          "bowl", "banana", "apple", "sandwich", "orange",
          "broccoli", "carrot", "hot dog", "pizza", "donut",
          "cake", "chair", "sofa", "pottedplant", "bed",
          "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
          "remote", "keyboard", "cell phone", "microwave", "oven",
          "toaster", "sink", "refrigerator", "book", "clock",
          "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]

class YoloParams:
    # ------------------------------------------- Extracting layer parameters ------------------------------------------
    # Magic numbers are copied from yolo samples
    def __init__(self, side):
        self.num = 3
        self.coords = 4
        self.classes = CLASS_NUMBER
        self.side = side
        if self.side == 52:
            self.anchors = [12.0, 16.0, 19.0, 36.0, 40.0, 28.0]
        elif self.side == 13:
            self.anchors = [142.0, 110.0, 192.0, 243.0, 459.0, 401.0]
        else:
            self.anchors = [36.0, 75.0, 76.0, 55.0, 72.0, 146.0]


def scale_bbox(x, y, height, width, class_id, confidence, im_h, im_w, is_proportional):
    if is_proportional:
        scale = np.array([min(im_w/im_h, 1), min(im_h/im_w, 1)])
        offset = 0.5*(np.ones(2) - scale)
        x, y = (np.array([x, y]) - offset) / scale
        width, height = np.array([width, height]) / scale
    xmin = int((x - width / 2) * im_w)
    ymin = int((y - height / 2) * im_h)
    xmax = int(xmin + width * im_w)
    ymax = int(ymin + height * im_h)
    # Method item() used here to convert NumPy types to native types for compatibility with functions, which don't
    # support Numpy types (e.g., cv2.rectangle doesn't support int64 in color parameter)
    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id.item(), confidence=confidence.item())


def parse_yolo_region(predictions, resized_image_shape, original_im_shape, params, threshold, is_proportional):
    # ------------------------------------------ Validating output parameters ------------------------------------------
    _, _, out_blob_h, out_blob_w = predictions.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It sould be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # ------------------------------------------ Extracting layer parameters -------------------------------------------
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    size_normalizer = (resized_image_w, resized_image_h)
    bbox_size = params.coords + 1 + params.classes
    # ------------------------------------------- Parsing YOLO Region output -------------------------------------------
    for row, col, n in np.ndindex(params.side, params.side, params.num):
        # Getting raw values for each detection bounding box
        bbox = predictions[0, n*bbox_size:(n+1)*bbox_size, row, col]
        x, y, width, height, object_probability = bbox[:5]
        class_probabilities = bbox[5:]
        if object_probability < threshold:
            continue
        # Process raw value
        x = (col + x) / params.side
        y = (row + y) / params.side
        # Value for exp is very big number in some cases so following construction is using here
        try:
            width = exp(width)
            height = exp(height)
        except OverflowError:
            continue
        # Depends on topology we need to normalize sizes by feature maps (up to YOLOv3) or by input shape (YOLOv3)
        width = width * params.anchors[2 * n] / size_normalizer[0]
        height = height * params.anchors[2 * n + 1] / size_normalizer[1]

        class_id = np.argmax(class_probabilities)
        confidence = class_probabilities[class_id]*object_probability
        if confidence < threshold:
            continue
        objects.append(scale_bbox(x=x, y=y, height=height, width=width, class_id=class_id, confidence=confidence,
                                  im_h=orig_im_h, im_w=orig_im_w, is_proportional=is_proportional))
    return objects


def intersection_over_union(box_1, box_2):#add DIOU-NMS support
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])

    cw = max(box_1['xmax'], box_2['xmax'])-min(box_1['xmin'], box_2['xmin'])
    ch = max(box_1['ymax'], box_2['ymax'])-min(box_1['ymin'], box_2['ymin'])
    c_area = cw**2+ch**2+1e-16
    rh02 = ((box_2['xmax']+box_2['xmin'])-(box_1['xmax']+box_1['xmin']))**2/4+((box_2['ymax']+box_2['ymin'])-(box_1['ymax']+box_1['ymin']))**2/4

    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area
    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap
    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union-pow(rh02/c_area,0.6)



def filter_objects(objects, iou_threshold, prob_threshold):
    # Filtering overlapping boxes with respect to the --iou_threshold CLI parameter
    objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
    for i in range(len(objects)):
        if objects[i]['confidence'] == 0:
            continue
        for j in range(i + 1, len(objects)):
            if intersection_over_union(objects[i], objects[j]) > iou_threshold:
                objects[j]['confidence'] = 0

    return tuple(obj for obj in objects if obj['confidence'] >= prob_threshold)


def process_frame(frame: VideoFrame, threshold: float = DETECT_THRESHOLD) -> bool:
    global input_height, input_width, prob_threshold, iou_threshold
    width = frame.video_info().width
    height = frame.video_info().height  
    objects = list()
    for tensor in frame.tensors():
        dims = tensor.dims()
        data = tensor.data()
        layer_params = YoloParams(dims[2])
        data = data.reshape(dims[0],dims[1],dims[2],dims[3])
        objects += parse_yolo_region(data,(input_height, input_width),(height, width),layer_params,prob_threshold, is_proportional=False)
    objects = filter_objects(objects, iou_threshold, prob_threshold)
    with frame.data() as frame:
        for obj in objects:
            # Validation bbox of detected object
            obj['xmax'] = min(obj['xmax'], width)
            obj['ymax'] = min(obj['ymax'], height)
            obj['xmin'] = max(obj['xmin'], 0)
            obj['ymin'] = max(obj['ymin'], 0)
            color = (min(obj['class_id'] * 12.5, 255),
                     min(obj['class_id'] * 7, 255),
                     min(obj['class_id'] * 5, 255))
            det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else str(obj['class_id'])
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), color, 2)
            cv2.putText(frame,"#" + det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',(obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
    return True
