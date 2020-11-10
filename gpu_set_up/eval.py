# -*- coding: utf-8 -*-
from imageai.Detection.Custom import DetectionModelTrainer
import os
"""
to run via shell script on -gpu
"""
# data folder
INPUT_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/hololens/"

# model folder
models_folder = INPUT_DIR+"models/"
#one_model_evaluation = models_folder + "detection_model-ex-176--loss-0006.507.h5"
one_model_evaluation= "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/pretrained-yolov3.h5"

#json configuration folder
JSON_DIR = INPUT_DIR+"/json/detection_config.json"

"""## evaluate performance"""
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=INPUT_DIR)
metrics = trainer.evaluateModel(model_path=one_model_evaluation, json_path= JSON_DIR, iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
print(metrics)
