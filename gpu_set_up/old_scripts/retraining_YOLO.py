# -*- coding: utf-8 -*-
# folllow https://imageai.readthedocs.io/en/latest/customdetection/index.html
from imageai.Detection.Custom import DetectionModelTrainer
import os
"""
test enviroment using interactive nodes (voltash for small jobs,sxm2sh for bigger ones)  
then run the full training on many epochs (min 5,up to 200) via shell script on -gpuv100
"""
INPUT_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/hololens/"

trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=INPUT_DIR)
trainer.setTrainConfig(object_names_array=["hololens"], batch_size=4, num_experiments=5, train_from_pretrained_model="pretrained-yolov3.h5")
trainer.trainModel()

"""## evaluate performance
trainer = DetectionModelTrainer()
trainer.setModelTypeAsYOLOv3()
trainer.setDataDirectory(data_directory=INPUT_DIR)
metrics = trainer.evaluateModel(model_path=INPUT_DIR+"models", json_path=INPUT_DIR+"/json/detection_config.json", iou_threshold=0.5, object_threshold=0.3, nms_threshold=0.5)
print(metrics)
"""

