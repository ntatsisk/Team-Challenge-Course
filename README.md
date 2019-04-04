# Team Challenge - Team3 - Automatic segmentation of the left ventricle - ADCD dataset

Repository of Team 3 for the course Team Challenge. Topic of this year is the segmentation of the left ventricle (LV) in 3D MRI images and the dataset comes from the ACDC challenge. Final goal is the accurate estimation of the Ejection Fraction of the heart for each patient.

## Getting Started

### Prerequisites
The following packages need to be installed:
- numpy
- SimpleITK
- scipy
- gryds
- keras
- tensorflow
- skfuzzy
- pandas
- skimage
- etc.. please add any that need to be added

### Code Layout
Order of scripts:
```
1. center_detection.py

   a. LV_detection.py
```
```
2. train_model.py

   a. data_preprocessing.py
   
   b. data_augmentation.py
   
   c. unet_architecture.py
```
```
3. model_evaluation.ipynb
```
## Script Instructions
**train_model.py** is the main script that performs the training of a model and has a model and its weights as outputs. Before train_model.py can be run, center_detection.py needs to be run. train_model.py calls the data_preprocess function in data_preprocessing.py if there are not yet numpy arrays with image data present. After the data is split into training, validation and test, data augmentation is performed on the test set by calling the deformable_data_augmentation function from data_augmentation.py. Then the unet model from unet_architecture.py is called to train the model. The model and model weights are saved in model.json and **model_weights.h5** respectively. These files can be used for the evaluation that is carried out in model_evaluation.ipynb.

**center_dectection.py** is a self-made helping to tool to determine the center of the left ventricle. Input for this tool are one end diastolic image of one patient (3D normalized numpy arrays) and the spacing. The output of the image are the found x and y center in the end diastolic phase. The images are preprocessed with a median filter to correct for field inhomogeneities. Some parameters thresholds are determined: area_rol (used to filter out the objects whose size is smaller than this value, with a value of 500/(spacing[1]*spacing[2]) in voxels), round_tol (used to filter out the objects whose roundness is smaller than this value, with a value of 0.45) and the spher_tol (used to filter out the objects whose sphericity is smaller than this value, with a value of 0.2). These parameters are used in the function LV_detection. The image and the number of clusters are also inputs for this function. Fuzzy means clustering takes place with three clusters, where the image is reshaped to a list and after the clustering reshaped back to the original size. It is made sure that the background cluster is always labeled as ‘0’, to prevent the cluster with the left ventricle from having label ‘0’ . The clustered regions are filtered based on the given parameters (area_tol, round_tol and spher_tol), when the values are lower than the given parameters, these regions will receive a value of 0 and will be filtered out. The detected image with one or more clustered regions is returned to center_detection.py. Here it checks whether one or more clusters remain. When there is only one clustered region left, this will be detected as the left ventricle. Otherwise, the region closest to the center of the image will be selected als left ventricle due to scanning protocol. An array with coordinates (xcenter and ycenter) of the left ventricles need to be saved manually as **centers_1mm.npy** so they can be used for the following step in the program.

In **data_preprocess.py** the left ventricle centers are loaded in, followed by loading in all files (images, ground truths and info file). Images are converted to numpy arrays, resized to an equal spacing of choice and cropped using the centers. The output of data_preprocess are three arrays of all images, all ground truths and patient info. patient_info.csv contains patient id, spacing, image size, stroke volume and ejection fraction.

**unet_architecture.py** contains the architecture of the model that works with the entire image as input. The model has down-sampling and up-sampling blocks. The down-sampling path had five convolutional blocks, each consisting of two convolutional layers with a filter size of 3x3. Layer four and five also contain dropout layers.  Each block ended with the use of 2x2 max pooling except for the last block. For the up-sampling path, each block started with a deconvolutional layer consisting of a filter with size 3x3 to increase the size of feature maps, following by a stride of 2x2 to decrease the number of feature maps. The last two layers were convolutional layers of a 3x3 filter and a 1x1 filter. A sigmoid function was used as the activation function which gave an output in the range of 0 and 1. While training the model the Adam optimization was used as the optimization strategy. The function returns the model.

**data_augmentation.py** takes 3D numpy arrays containing the images and ground truth in the training set as input. It first performs deformable transformations which are implemented as B-spline transformations on the training set. It is done by using “Gryds”, which is a Python package for geometric transformations for augmentations in deep learning developed by the TU/e. A for-loop is created over all images in the training dataset. The first step is to define an interpolator for the image. This is needed because it can happen that some points fall in between the grid after transformation. Afterwards, the displacement matrices “disp_i” and “disp_j” are generated for the displacements in the i-directions and j-directions respectively. For each image, two different displacement matrices are generated by creating two 3x3 matrices filled with random numbers between -0.05 and 0.05.  Then, the B-spline transformation is created and applied to both the image and the corresponding ground truth. 
The number of data augmentations that is performed per image in the training set is regulated by the parameter “nr_epochs”. If “nr_epochs” is set to 2, then two different data augmentations are applied to the image. All the newly generated images are stored in an empty list. This list is converted into numpy and an extra image dimension is added. This is done because the second part of the data augmentation requires a 4D numpy array. So, after performing the B-spline transformations, other kinds of transformations are applied to the images. In this stage, no new images are generated. The transformations that are applied to the deformed images are:
- a rotation between 0 and 5 degrees
- brightness change between 95 and 105 %
- translations in height between 0 and 5 pixels
- translations in width between 0 and 5 pixels

Also, the augmented images (not the ground truth!) are normalized samplewise. After augmentation, the newly generated images are stored in an empty list. This list is converted into a numpy array again. The original images are converted from a 3D numpy array to a 4D numpy array to be able to concatenate the original images with the data augmented images at the end of the function. The result are two 4D numpy arrays: one containing the original images and the data augmented images and one containing the original ground truth and the data augmented ground truth. 

In the last part, the model is evaluated with: **model_evaluation.ipynb**. First the model (**model.json**) and the weights of the model (**model_weights.h5**) need to be loaded. Then the cropped data from both the prediction and the ground truth are loaded in. For the ground truth only the labels for the left ventricles has to be saved, the others labels for the right ventricle and the myocardium are deleted. A unit variance normalization takes place on the images from the model. The predicted image and the ground truth are both turned into 4D data. Then all the patient information is loaded in. An array with a size of 1902 is created, where each row corresponds to the ID of every slice of the dataset. The data set consists of five different patients groups, with twenty samples per group. The patients of each group are shuffled and assigned to the training set (70%), validation set (15%) and test set (15%). The split identification numbers (id) are sorted and id masks are created. The images are splitted based on the masks. And the validation set consists of 1320 slices, the validation set on 264 slices and the test set on 318 slices for both the images from the model and the ground truth. For the validation set, prediction set and test set the dice score, maximum hausdorff distance and the ejection fraction are calculated by using **calculate_metrics.py**. For the dice score the true positive mask needs to be found, where the voxels for both the prediction and the ground truth have a value of one. So both the prediction and the ground truth image have detected the left ventricle. With the false positive mask, only the prediction image predicted the left ventricle and the ground truth image didn’t detect the left ventricle on this position. With the false negative mask, only the ground truth image predicted the left ventricle and the prediction image didn’t detect the left ventricle on this position. With these above mentioned values the dice score is calculated and printed per patient. The maximum, minimum, mean and standard deviation of the dice score are also printed. The hausdorff distance is calculated for each slice by using directed_hausdorff, this has to be done separately for each slice, because it is 2D instead of 3D. The mean hausdorff distance is calculated and printed per patient. The maximum, minimum, mean and standard deviation of the hausdorff distance are printed. The ejection fraction is calculated by dividing the stroke volume (end-diastolic volume subtracted by the end-systolic volume) by the end-diastolic volume and multiply with a factor of 100. This is done for both the prediction as the ground truth images. The maximum, minimum, mean and standard deviation of the ejection fraction are printed for all the different sets.


## Authors
Symona Beyer

Madelon van den Dobbelsteen 

Dave van GruijtHuijsen

Anouk Marinus

Konstantinos Ntatsis

Vera van Hal
