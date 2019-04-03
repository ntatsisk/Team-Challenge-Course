import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
import random
import scipy
import gryds

def data_augmentation(images, images_gt):
    '''
    Images and images_gt should be 3D numpy arrays in which the first index indicates the number of 2D images
    This function performs Bspline transformation, rotation, translation, brightness change and samplewise normalization of the augmented images
    The output is a 4D numpy array of 3D images with the original and augmented data
    ''' 
    
    #The nr_epochs states how many data augmentations are performed
    nr_epochs=2
    
    #Make an empty list for the deformed images and masks
    deformed_images=[]
    deformed_images_gt=[]
    
    #Define the augmentation apart from B-spline transformation for the images
    datagen=dict(rotation_range=5, 
        samplewise_center=True,
        samplewise_std_normalization=True,
        brightness_range=[0.95, 1.05],
        width_shift_range=5,
        height_shift_range=5)
    
    #Define the augmentation apart from B-spline transformation for the masks (these should not be normalized because they are binary)
    datagen_gt=dict(rotation_range=5, width_shift_range=5, height_shift_range=5)
    
    #Call the ImageDataGenerator
    train_datagen=ImageDataGenerator(**datagen)
    train_datagen_gt=ImageDataGenerator(**datagen_gt)
    
    print("Performing data augmentation... \n")
    for epoch in range(nr_epochs):
        print('Epoch', epoch)
        
        #Perform B-spline transformation with the use of "Gryds"
        for image in range(images.shape[0]):
            #By default the mode of the interpolator is constant which interpolates a value of 0 at the border.
            #Alternatively, you can choose a mode for other interpolation: nearest, mirror, wrap or reflect
            an_image_interpolator = gryds.Interpolator(images[image]) 
            an_image_gt_interpolator = gryds.Interpolator(images_gt[image])
            
            #Randomly generate a 2D displacement matrix of size (3,3) for the i and j direction
            disp_i=np.random.uniform(low=-0.05, high=0.05, size=(3,3))
            disp_j=np.random.uniform(low=-0.05, high=0.05, size=(3,3))
            
            #Calculate the transformation
            transformation = gryds.BSplineTransformation([disp_i, disp_j])
            
            #Apply the transformation to the image and corresponding mask
            deformed_image = an_image_interpolator.transform(transformation)
            deformed_image_gt = an_image_gt_interpolator.transform(transformation)
            
            #Save the deformed image and mask in a list
            deformed_images.append(deformed_image)
            deformed_images_gt.append(deformed_image_gt)
    
    #Convert list to numpy array        
    deformed_images=np.array(deformed_images)
    deformed_images_gt=np.array(deformed_images_gt)
    
    #Convert 3D to 4D numpy array
    deformed_images = np.reshape(deformed_images, newshape=(*deformed_images.shape, 1))
    deformed_images_gt = np.reshape(deformed_images_gt, newshape=(*deformed_images_gt.shape, 1)) 
    
    #Make an empty list for the resulting augmented images and masks
    augmented_images=[]
    augmented_images_gt=[]
    
    #Perform rotation, translation, brightness change and normalization to the deformed images
    batches = 0        
    for batch in train_datagen.flow(deformed_images, batch_size=1, seed=0):
        batches += 1
        augmented_images.append(batch[0,:,:,:])
        if batches >= len(deformed_images):
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
    
    #Perform rotation, translation and brightness change to the corresponding masks of the deformed images  
    #These augmentations are the same as for the deformed images, because of the seed          
    batches = 0
    for batch in train_datagen_gt.flow(deformed_images_gt, batch_size=1, seed=0):
        batches += 1
        augmented_images_gt.append(batch[0,:,:,:])
        if batches >= len(deformed_images_gt):
            # we need to break the loop by hand because
            # the generator loops indefinitely
            break
                
    #Convert list to numpy array
    augmented_images=np.array(augmented_images)
    augmented_images_gt=np.array(augmented_images_gt)
    
    #Convert original images and masks from 3D to 4D numpy array
    images = np.reshape(images, newshape=(*images.shape, 1))
    images_gt = np.reshape(images_gt, newshape=(*images_gt.shape, 1))
    
    #Concatenate numpy arrays of original images and masks with the augmented images and masks
    augmented_images=np.concatenate((images,augmented_images),axis=0)
    augmented_images_gt=np.concatenate((images_gt,augmented_images_gt), axis=0)
    
    #Save the resulting 4D numpy arrays
    np.save("augmented_images", augmented_images)
    np.save("augmented_images_gt", augmented_images_gt)
    
    return augmented_images, augmented_images_gt
