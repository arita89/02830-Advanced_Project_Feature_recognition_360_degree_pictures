## 02830- advanced project in digital media engineering **
Taormina, Arianna \\ s163671
Majidi, Oldouz  \\ s163502  

Scope of this project is to train a model able to detect features in panoramic 360 degree pictures, in which lines can be more or less deformed due to the projection on the plane. 
The chosen method is: transfer learning from a pre-trained model.
The chosen model is: yolo v3 for its versatility and ability to detect more features at once (even though we trained only for one).
The chosen package is: ImageAI, which required modifications on the source code to make it compatible with supported versions of TensorFlow and Keras.

The data sources are multiple and different:
- part has been collected on our own, with a goPro (kindly given by DTU Skylab) 
- part has been scraped from the internet (using Image Downloader Chrome Extension ) 
- part has been produced via image augmentation as per correspondent Notebook(flip, split, zoom, spherical projection)
- part has been given by Google Team as resource available to students(ca 3300 panoramic images part of dataset "") 

In fact one of the major challenge to train a model to recognise deformed features, is the availability of such data to train with. 

The second challenge is the time required for training.
For reference:
- using cpu to train over 10 epochs would take ca 34 hours.
- using 1core gpu (interactive nodes) to train over 10 epochs would take roughly 100 minutes. 
- using 4cores gpu (gpuv nodes) to train over 10 epoch takes a very acceptable 20 minutes. 
Both interactive and gpuv nodes are accessible through hpc services, of course there is sometimes queue on the gpuv ones (for up to 24h) and sometimes the interactive ones are also completely used. 

The third challenge is space to store images and trained models.
My own premium Google Drive (with 100GB) has been used for the purpose and I will keep the files store at least until this semester is over. 

The fourth challenge was to create datasets with desired characteristics, by sampling from the -limited- available data.

## Project tasks, progresses and contributions
https://github.com/arita89/Feature_recognition_360_degree_pictures/projects/1

## Prerequisites

```
tensorflow-gpu==2.3.0
opencv-python
keras
cuda/10.1
cudnn/v7.6.5.32-prod-cuda-10.1
ImageAI-master- updated version to tf:2-3
```

## Files Description
1.data_augmentation_pipeline.ipynb - applies augmentation to the training data and saves it
2.dataset_creation_pipeline.ipynb - feeds from 3 datasets and creates train, validation and test sets
3.fire_train.py - trains model against the training images of fire extinguishers
4.fire_val.py - validation of the model against the validation images
5.fire_detection.py - application of the trained model to new test data

## Original Data directory 
https://drive.google.com/drive/folders/1EfidSd0ToVxPfxmiyUQ1ifSJMCTsOL-N?usp=sharing

### Original Data directory layout

    .
    ├── DATA_0                 # contains #100 360 degree images from Google dataset and that have been collected first hand, plus labels
    ├── DATA_1                 # contains #400 images with no deformation, swiped from internet, plus labels 
    ├── DATA_2                 # contains #400 images with high level deformation, obtained via data augmentation, plus labels      
    └── TEST

## Models and Datasets directory
https://drive.google.com/drive/folders/11_oMAxRm2sanwdQGY3jr1EG3t5AkJEF3?usp=sharing

### Dataset top-level directory layout

    .
    ├── scripts                 # contains a backup copy of the .py files living on the hpc server
    ├── model                   # contains zip files with different trained models and the datasets they have been trained on
    └── data                    # contains zip files with different datasets with their training and validation images
    
### Model.zip directory layout

    .
    ├── model_eval_High                 # contains results of trained model validated against images with high degree of deformation
    ├── model_eval_Low                  # contains results of trained model validated against images with high low of deformation
    └── model_eval_Non                  # contains results of trained model validated against images with high no deformation

### One dataset directory layout

    .
    ├──train                 
        ├── images              # contains images used for training
        └── annotations         # contains correspondent PascalVOC annotations in xml format
    └── validation           
        ├── images              # contains images used for validation     
        └── annotations         # contains correspondent PascalVOC annotations in xml format
    
```
