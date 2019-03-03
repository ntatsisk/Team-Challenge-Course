import os
import numpy as np
import SimpleITK as sitk
import pandas as pd
from skimage.transform import rescale

def data_preprocess(min_dim_x=128, min_dim_y=128, min_dim_z=1, rescale_bool=False, new_scale_factor=1.0):
    """
    Crops the images to min_dim_x, min_dim_y. Saves the modified
    images to two numpy arrays. Creates csv file with patient images
    information

    new_scale_factor: Use lower than 1 when you want to "unzoom" in rescaling an
                      image to avoid cropping part of LV in some cases. It was
                      found that for a cropping goal of 128x128 pixels the minimum
                      new_scale_factor can be 0.585. ** Note that the new_spacing
                      of the images after rescaling will be 1 / new_scale_factor
    """
    if rescale_bool: print("Rescaling images is on ...")
    root = './Data/'
    images = np.zeros(shape=(1, min_dim_y, min_dim_x))
    images_gt = np.zeros(shape=(1, min_dim_y, min_dim_x))
    spacing = []
    image_sizes =[]
    patient_ids = []
    for subdir in os.listdir(root):
        patient_id = int(subdir[-3:]) # eg patient085 --> 85
        once = True # flag to write only once per patient in the csv file
        if (patient_id-1)%5==0: print("Processed {} out of 100 patients".format(patient_id-1))
        for filename in os.listdir(root + subdir):
            if 'frame' in filename:
                # Read image and convert to numpy array
                filepath = '/'.join([root, subdir, filename])
                im = sitk.ReadImage(filepath)
                spac = im.GetSpacing()
                im_array = sitk.GetArrayFromImage(im)
                im_size_unscaled = im_array.shape
                # Rescale using spacing, LVs with spacing<1 will become larger
                # while LVs with spacing>1 will become smaller (after cropping)
                if rescale_bool:
                    # anti_aliasing seems to cause a bug in some cases(downsampling), keep False unti further tested
                    im_array = rescale(im_array, scale=(1.0, spac[0]*new_scale_factor, spac[1]*new_scale_factor),
                                        preserve_range=True, multichannel=False, anti_aliasing=False)
                im_size = im_array.shape
                #print(spac, im_size)
                # Calculate how much it needs to be cropped
                x_diff = im_size[2] - min_dim_x
                y_diff = im_size[1] - min_dim_y
                z_dim = im_size[0]

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

                # For the csv file
                if once:
                    patient_ids.append(patient_id)
                    spacing.append(spac)
                    image_sizes.append(im_size_unscaled)
                    once = False

    # Remove first helper array for the concatenation
    images = images[1:, :, :]
    images_gt = images_gt[1:, :, :]
    images_gt = np.round(images_gt) # interpolation for rescaling creates
                                     # values between the 0,1,2 and 3

    # Save the data into numpy arrays
    if rescale_bool:
        np.save("cropped_rescaled_images", images)
        np.save("cropped_rescaled_images_gt", images_gt)
    else:
        np.save("cropped_images", images)
        np.save("cropped_images_gt", images_gt)

    # Create pandas dataframe with the inforamtion
    patient_info = pd.DataFrame([])
    patient_info["patient_id"] = patient_ids
    patient_info["spacing"] = spacing
    image_sizes_np = np.array(image_sizes)
    # reverse the order, final: (x,y,z)
    patient_info["image_pixels"] = tuple(image_sizes_np[:,[2,1,0]])

    # Write to .csv
    patient_info.to_csv("patient_info.csv", index=False)
    # !Use the following to read later:!
    # xx = pd.read_csv("patient_info.csv")
    # ids = xx["patient_id"].to_numpy() [example]