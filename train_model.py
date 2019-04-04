import numpy as np
import tensorflow as tf
import keras
from keras.models import model_from_json
from keras.callbacks import ModelCheckpoint
import time
from unet_architecture import *
from data_preprocess import *
from data_augmentation import *
import os
import ast
import copy

rescale_bool = True

# Check if the data numpy arrays already exist
if rescale_bool:
    exists1 = os.path.isfile("cropped_rescaled_images.npy")
    exists2 = os.path.isfile("cropped_rescaled_images_gt.npy")
else:
    exists1 = os.path.isfile("cropped_images.npy")
    exists2 = os.path.isfile("cropped_images_gt.npy")

# Call the data_preprocess function if there are no numpy arrays yet
if not exists1 or not exists2:
    print("Creating the files ... \n\n")
    if rescale_bool:
        data_preprocess(min_dim_x=144, min_dim_y=144, rescale_bool=rescale_bool,
                            new_scale_factor=1.0)
    else:
        data_preprocess()

# Load the data
print("Loading the data ...\n\n")
if rescale_bool:
    images = np.load("cropped_rescaled_images.npy")
    images_gt = np.load("cropped_rescaled_images_gt.npy")
else:
    images = np.load("cropped_images.npy")
    images_gt = np.load("cropped_images_gt.npy")

print("Image data has size: {}".format(images.shape))
print("Ground truth has size: {}".format(images_gt.shape))

# For now keep only the labels of the Left Venctricle (removes RV, myocardium)
images_gt[images_gt != 3] = 0
images_gt[images_gt == 3] = 1

# Zero-mean, unit-variance normalization
mean_per_slice = np.mean(images, axis=(1,2), keepdims=True)
std_per_slice =  np.std(images, axis=(1,2), keepdims=True)
images = (images - mean_per_slice) / std_per_slice

# Load patient information
# the converter is used to convert back the tuple
patient_info = pd.read_csv("patient_info.csv", converters={"spacing": ast.literal_eval,"image_pixels": ast.literal_eval })

id_list = patient_info["patient_id"].to_numpy()
image_sizes = patient_info["image_pixels"].to_numpy()
image_sizes = np.array([*image_sizes])
z_dim = image_sizes[:,2]

# Create an array of (1902,) where each row corresponds to the ID of every slice
# of the dataset
patient_id_array = np.array([])
for patient_id in id_list:
    patient_id_array = np.append(patient_id_array, np.full(shape=2*z_dim[patient_id-1], fill_value=patient_id))

print("The array of patient IDs has shape: ", patient_id_array.shape)
# Split dataset to train/valid/test based on patient ids (doesnt mix patient slices)
# and sample uniformly each of the 5 groups of patients
np.random.seed(0) # seed for reproducability
train_ids = np.array([])
valid_ids = np.array([])
test_ids = np.array([])

# There are 5 patient groups with 20 patients each
samples_per_group = 20
num_of_groups = 5
for group in range(num_of_groups):
    # Shuffle the patients of each group and split to train/valid/test
    group_id_list = copy.deepcopy(id_list[group*samples_per_group:(group+1)*samples_per_group])
    np.random.shuffle(group_id_list)
    train_ids = np.append(train_ids, group_id_list[:14]) # 70% training
    valid_ids = np.append(valid_ids, group_id_list[14:17]) # 15% validation
    test_ids = np.append(test_ids, group_id_list[17:]) # 15% test

# Create the id masks
train_msk = np.isin(patient_id_array, train_ids)
valid_msk = np.isin(patient_id_array, valid_ids)
test_msk = np.isin(patient_id_array, test_ids)

print("The train set consists of {} slices".format(np.count_nonzero(train_msk)))
print("The validation set consists of {} slices".format(np.count_nonzero(valid_msk)))
print("The test set consists of {} slices\n\n".format(np.count_nonzero(test_msk)))

# Now split the images based on the masks
train_set = images[train_msk]
train_set_gt = images_gt[train_msk]

valid_set = images[valid_msk]
valid_set_gt = images_gt[valid_msk]

test_set = images[test_msk]
test_set_gt = images_gt[test_msk]

augmented_train_set, augmented_train_set_gt=deformable_data_augmentation(train_set, train_set_gt)
print("Augmented training set has size: {}\n".format(augmented_train_set.shape))
print("Augmented ground truth in training set has size: {}\n".format(augmented_train_set_gt.shape))

# Compile and train U-net
model = unet(input_size=(144, 144, 1))
mc = ModelCheckpoint('weights{epoch:08d}.h5',
                                     save_weights_only=True, period=5)
model.fit(augmented_train_set, augmented_train_set_gt, batch_size=10, epochs=200)

# Serialize the model to json
print("Saving model and weights ...")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize the weights to HDF5
model.save_weights("model_weights.h5")
