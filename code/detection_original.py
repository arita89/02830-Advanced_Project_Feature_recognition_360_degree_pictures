from imageai.Detection.Custom import CustomObjectDetection

## CHANGE THIS
image_name = "Low_Image_1"
image_extention = ".jpg"
image_original_path = image_name+image_extention
image_modify_path = image_name+"-detected" +image_extention

## CHANGE THIS
#model folder
one_model= "detection_model-ex-010--loss-0007.663.h5"

# data folder
input_images_dir= "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/test/Low/"
output_images_dir = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/results/"


# data folder THIS DOESNT CHANGE
INPUT_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/"


## THIS DOES NOT CHANGE 
# model folder
models_folder = INPUT_DIR+"models/"
one_model_evaluation=  models_folder + one_model
print ("using model for testing: ",one_model_evaluation)

#json configuration folder THIS DOESNT CHANGE 
JSON_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/fire_extinguisher/json/detection_config.json"

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(one_model_evaluation)
detector.setJsonPath(JSON_DIR)
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=input_images_dir+image_original_path, output_image_path= output_images_dir + image_modify_path)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
