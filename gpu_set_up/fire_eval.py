# -*- coding: utf-8 -*-
from imageai.Detection.Custom import DetectionModelTrainer
import os
"""
to run via shell script on -gpu
"""
# data folder THIS DOESNT CHANGE
#INPUT_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/hololens/"
INPUT_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/"

# model folder THIS DOESNT CHANGE
models_folder ="/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/models/"
print (models_folder)
#one_model_evaluation = models_folder + "detection_model-ex-176--loss-0006.507.h5"
#one_model_evaluation= "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/pretrained-yolov3.h5"
#one_model_evaluation= "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-0_official_IMAGEAI_upgrade/data/hololens/models/detection_model-ex-005--loss-0008.398.h5"
#one_model_evaluation= "/zhome/94/5/101974/Desktop/Interactive_Nodes/YOLO - SIM_DATA/non-deformed/models/detection_model-ex-005--loss-0015.334.h5"


#model folder THIS CHANGES IF YOU WANT TO VALIDATE ONE SPECIFIC MODEL 
# model that worked
#one_model_evaluation = "/zhome/94/5/101974/Desktop/Interactive_Nodes/YOLO_full/deformed/models/detection_model-ex-197--loss-0004.431.h5"
#one_model_evaluation = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/1000_fire_corrected/models/detection_model-ex-011--loss-0008.460.h5"

# new models 
#one_model_evaluation = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/models/detection_model-ex-010--loss-0009.515.h5"

#json configuration folder THIS DOESNT CHANGE 
#JSON_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/YOLO_full/deformed/json/detection_config.json"
JSON_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/json/detection_config.json"

"""## evaluate performance"""
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=INPUT_DIR)
metrics = trainer.evaluateModel(model_path=models_folder, json_path= JSON_DIR, iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
print(metrics)
