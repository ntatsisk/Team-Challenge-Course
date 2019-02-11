import numpy as np
import tensorflow as tf
import keras
from keras.models import model_from_json
import time
from unet_architecture import *
from data_handle import *
import os

# Load the data
exists1 = os.path.isfile("cropped_images.npy")
exists2 = os.path.isfile("cropped_images_gt.npy")

# Check if data numpy arrays are not created
if not exists1 or not exists2:
    print("Creating the files ... \n\n")
    data_preprocess()

# Load the data
print("Loading the data ...\n\n")
images = np.load("cropped_images.npy")
images_gt = np.load("cropped_images_gt.npy")

print("Image data has size: {}".format(images.shape))
print("Ground truth has size: {}\n\n".format(images_gt  .shape))

# For now keep only the labels of the Left Venctricle
images_gt[images_gt != 3] = 0
images_gt[images_gt == 3] = 1

# Scale data and convert to 4D (requirement: for the conv2d training)
images = images / np.amax(images) # scale
images = np.reshape(images, newshape=(*images.shape, 1))
images_gt = np.reshape(images_gt, newshape=(*images_gt.shape, 1))

print("Reshaped image data has size: {}".format(images.shape))
print("Reshaped ground truth has size: {}\n\n".format(images_gt.shape))

# Split to training and validation set
# NOTE!!: both training and validation sets might contain frames from the same
# patient. So they are not actually independent!
msk = np.random.rand(images.shape[0]) < 0.8

train_set = images[msk, :, :, :]
train_set_gt = images_gt[msk, :, :, :]

valid_set = images[~msk, :, :, :]
valid_set_gt = images_gt[~msk, :, :, :]

# Convert the labels to one hot encoding
# https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
# (np.arange(a.max()) == a[...,None]-1).astype(int)
# train_set_gt =(np.arange(train_set_gt.max()) == train_set_gt[...,None]-1).astype(int)
# valid_set_gt =(np.arange(valid_set_gt.max()) == valid_set_gt[...,None]-1).astype(int)

# Compile and train U-net
model = unet(input_size=(128, 128, 1))
model.fit(train_set, train_set_gt, batch_size=64, epochs=20)
# score = model.evaluate(valid_set[:200, :, :, :], valid_set_gt[:200, :, :, :], batch_size=20)

# Serialize the model to json
print("Saving model and weights ...")
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# Serialize the weights to HDF5
model.save_weights("model_weights.h5")
