# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 12:02:32 2019

@author: 30753
"""
from Detection import ObjectDetection
from keras_retinanet import models
# from detectImage import Detect_img
import tensorflow as tf
import keras
import os
import cv2
import numpy as np



keras.backend.clear_session()
# 获取项目路径
execution_path = os.getcwd()
# 新建类
detector = ObjectDetection()
# 指定模型为retinanet
detector.setModelTypeAsRetinaNet()
# 指定模型.h5权值路径
detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
# 构建模型：detection_speed对应图片限制大小（影响检测速度）、类别数量num_classes等
detector.loadModel(detection_speed="faster")#通过更改detection_speed可调节识别精度，相应的识别速度减慢
# 检测视频
detections = detector.detectCustomObjectsFromVideo(frame_detection_interval=2,custom_objects=detector.CustomObjects(person=True, bicycle=True, car=True, motorcycle=True, bus=True, train=True, truck=True),input_file_path="./test_imgvideo/123.avi", output_file_path="./result")

# 检测单张图片
#detection = detector.detectCustomObjectsFromImage(custom_objects=detector.CustomObjects(person=True, bicycle=True, car=True, motorcycle=True, bus=True, train=True, truck=True),input_image=os.path.join(execution_path , "./test_imgvideo/img.jpg"), output_image_path=os.path.join(execution_path , "./test_imgvideo/result.jpg"))
