import os
import numpy as np
import SimpleITK as sitk

def data_preprocess(min_dim_x=128, min_dim_y=128, min_dim_z=1):
    """ Crops the images to min_dim_x, min_dim_y. Saves the modified
    images to two numpy arrays. Note: they should be power of 2. """
    root = './Data/'
    images = np.zeros(shape=(1, min_dim_y, min_dim_x))
    images_gt = np.zeros(shape=(1, min_dim_y, min_dim_x))
    for subdir in os.listdir(root):
        for filename in os.listdir(root + subdir):
            if 'frame' in filename:
                # Read image and convert to numpy array
                filepath = '/'.join([root, subdir, filename])
                im = sitk.ReadImage(filepath)
                im_array = sitk.GetArrayFromImage(im)

                # Calculate how much it needs to be cropped
                x_diff = im_array.shape[2] - min_dim_x
                y_diff = im_array.shape[1] - min_dim_y

                # Crop the array
                if (x_diff>0):
                    cropped_array = im_array[:, :, x_diff%2 + int(x_diff/2):-int(x_diff/2)]
                else:
                    cropped_array = im_array

                if (y_diff>0):
                    cropped_array = cropped_array[:, y_diff%2 + int(y_diff/2):-int(y_diff/2), :]

                # Store the cropped image array in new array along z axis (axis=0)
                if 'gt' in filename:
                    images_gt = np.concatenate((images_gt, cropped_array), axis=0)
                else:
                    images = np.concatenate((images, cropped_array), axis=0)

    # Remove first helper array for the concatenation
    images = images[1:, :, :]
    images_gt = images_gt[1:, :, :]

    # Save the data into numpy arrays
    np.save("cropped_images", images)
    np.save("cropped_images_gt", images_gt)
