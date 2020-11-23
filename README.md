## 02830- advanced project in digital media engineering **
Majidi, Oldouz  \\ s163502  
Taormina, Arianna \\ s163671


**see TAB Projects for Tasks overview**


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
    
```
