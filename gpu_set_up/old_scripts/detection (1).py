from imageai.Detection.Custom import CustomObjectDetection
# data folder
INPUT_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/hololens/" # for training and validation image
input_images_dir= "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/test-image/"
output_images_dir = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/output_images/"

image_name = "holo1"
image_extention = ".jpg"

image_original_path = image_name+image_extention
image_modify_path = image_name+"-detected" +image_extention

# model folder
models_folder = INPUT_DIR+"models/"
#one_model_evaluation = models_folder + "detection_model-ex-176--loss-0006.507.h5"
one_model_evaluation= "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/pretrained-yolov3.h5"

print ("model: ",one_model_evaluation)

#json configuration folder
JSON_DIR = "/zhome/94/5/101974/Desktop/Interactive_Nodes/tf_2-3/data/hololens/json/detection_config.json"

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(one_model_evaluation)
detector.setJsonPath(JSON_DIR)
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=input_images_dir+image_original_path, output_image_path= output_images_dir + image_modify_path)
for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])
