from imageai.Detection.Custom import CustomObjectDetection
import os

## CHANGE THIS
#model folder
one_model= "detection_model-ex-008--loss-0009.504.h5"

## THIS DOES NOT CHANGE 
# data folder
INPUT_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/"

## THIS DOES NOT CHANGE 
# model folder
models_folder = INPUT_DIR+"models/"
one_model_evaluation=  models_folder + one_model
print ("using model for testing: ",one_model_evaluation)

# THIS DOES NOT CHANGE 
JSON_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/json/detection_config.json"

## THIS DOES NOT CHANGE 
# check reading all images from directory
test_dir = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/test/Low/"
sub_dir = os.listdir(test_dir)


# detect
detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(one_model_evaluation)
detector.setJsonPath(JSON_DIR)
detector.loadModel()

this = "Low_Image_1.jpg"
image_name = this[:-4]
image_extention = this[-4:]
print ("reading")
print (this)
#print (image_name)
#print (image_extention)

# input dir
# each from its own sub
input_file = test_dir +"/" + image_name + image_extention
print (input_file)

#output dir
# all in the same folder
output_dir = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/results/"
output_file = output_dir + image_name + "_detected" + "_SINGLE"+ image_extention 
print (output_file)

detections = detector.detectObjectsFromImage(input_image = input_file,  output_image_path= output_file)
for detection in detections:
	print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
