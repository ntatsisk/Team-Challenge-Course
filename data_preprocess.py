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
                      image to avoid cropping part of LV in some cases.
                      ** Note that the new_spacing of the images after rescaling
                      will be 1 / new_scale_factor.
    """
    if rescale_bool: print("Rescaling images is on ...\n")
    root = './Data/'
    images = np.zeros(shape=(1, min_dim_y, min_dim_x))
    images_gt = np.zeros(shape=(1, min_dim_y, min_dim_x))
    spacing = []
    image_sizes =[]
    patient_ids = []
    ED_volumes = []
    ES_volumes = []
    counter = -1;

    # Load centers of the detected LV
    detected_centers = np.load("centers_1mm.npy")

    for subdir in os.listdir(root):
        patient_id = int(subdir[-3:]) # eg patient085 --> 85
        once = True # flag to write only once per patient in the csv file
        if (patient_id-1)%5==0: print("Processed {} out of 100 patients".format(patient_id-1))
        ED_frame_flag = True
        for filename in os.listdir(root + subdir):
            if 'frame' in filename:
                # Keep count of number of images to match the right center with the image
                counter = counter+1
                # Read image and convert to numpy array
                filepath = '/'.join([root, subdir, filename])
                im = sitk.ReadImage(filepath)
                spac = im.GetSpacing()
                im_array = sitk.GetArrayFromImage(im)
                im_size_unscaled = im_array.shape
                # Calculate EF - before rescaling/cropping
                if 'gt' in filename:
                    voxelvolume = spac[0]*spac[1]*spac[2]
                    if ED_frame_flag: # the first frame of the folder is ED
                        ED_volume = np.sum(im_array==3) * voxelvolume
                        ED_volumes.append(ED_volume)
                        ED_frame_flag = False
                    else: # the second frame is ES
                        ES_volume = np.sum(im_array==3) * voxelvolume
                        ES_volumes.append(ES_volume)      

                # Convert the centers from 1mm spacing to current spacing
                x_center = int(np.floor(detected_centers[counter, 0] * new_scale_factor))
                y_center = int(np.floor(detected_centers[counter, 1] * new_scale_factor))
                
                # Rescale using spacing, LVs with spacing<1 will become larger
                # while LVs with spacing>1 will become smaller (after cropping)
                if rescale_bool:
                    # anti_aliasing seems to cause a bug in some cases(downsampling), keep False unti further tested
                    im_array = rescale(im_array, scale=(1.0, spac[0]*new_scale_factor, spac[1]*new_scale_factor),
                                        preserve_range=True, multichannel=False, anti_aliasing=False)

                # Crop the array
                cropped_array = im_array[:, :, x_center-int(min_dim_x/2):x_center+int(min_dim_x/2)]
                cropped_array = cropped_array[:, y_center-int(min_dim_y/2):y_center+int(min_dim_y/2), :]

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

    # Stroke volume and EF calculation
    ED_volumes = np.array(ED_volumes, dtype=np.float32)
    ES_volumes = np.array(ES_volumes, dtype=np.float32)
    strokevolumes = ED_volumes - ES_volumes
    patient_info["stroke_volume"] = strokevolumes*1e-3
    patient_info["ejection_fraction"] = strokevolumes / ED_volumes * 100
    # Write to .csv
    patient_info.to_csv("patient_info.csv", index=False)

