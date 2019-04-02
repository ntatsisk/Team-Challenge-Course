# Team Challenge - Team3 - Automatic segmentation of the left ventricle - ADCD dataset

Repository of Team 3 for the course Team Challenge. Topic of this year is the segmentation of the left ventricle (LV) in 3D MRI images and the dataset comes from the ACDC challenge. Final goal is the accurate estimation of the Ejection Fraction of the heart for each patient.

## Getting Started

### Prerequisites
The following packages need to be installed:
- SimpleITK
- scipy
- gryds
- keras
- tensorflow
- skfuzzy
- etc.. please add any that need to be added

### Installing
Order of scripts:
1. center_detection.py

   a. LV_detection.py
   
2. train_model.py

   a. data_preprocessing.py
   
   b. data_augmentation.py
   
   c. unet_architecture.py
   
3. model_evaluation.ipynb

## Main Script
**train_model.py** is the main script that performs the training of a model and has a model and its weights as outputs. Before train_model.py can be run, center_detection.py needs to be run. train_model.py calls the data_preprocess function in data_preprocessing.py if there are not yet numpy arrays with image data present. After the data is split into training, validation and test, data augmentation is performed on the test set by calling the deformable_data_augmentation function from data_augmentation.py. Then the unet model from unet_architecture.py is called to train the model. The model and model weights are saved in model.json and model_weights.h5 respectively. These files can be used for the evaluation that is carried out in model_evaluation.ipynb.

## U-Net Architecture
**unet_architecture.py** contains the architecture of the model that works with the entire image as input. The model has down-sampling and up-sampling blocks. The down-sampling path had five convolutional blocks, each consisting of two convolutional layers with a filter size of 3x3. Layer four and five also contain dropout layers.  Each block ended with the use of 2x2 max pooling except for the last block. For the up-sampling path, each block started with a deconvolutional layer consisting of a filter with size 3x3 to increase the size of feature maps, following by a stride of 2x2 to decrease the number of feature maps. The last two layers were convolutional layers of a 3x3 filter and a 1x1 filter. A sigmoid function was used as the activation function which gave an output in the range of 0 and 1. While training the model the Adam optimization was used as the optimization strategy. The function returns the model.
--> here the link to the architecture needs to be written

## Authors
Symona Beyer

Madelon van den Dobbelsteen 

Dave van GruijtHuijsen

Anouk Marinus

Kostas Ntatsis

Vera van Hal
