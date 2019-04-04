import numpy as np
from skimage.measure import label,regionprops
from scipy.ndimage.filters import median_filter
import matplotlib.pyplot as plt
from LV_detection import LV_detection

def center_detection(img, spacing):
    # input:
    #   - img: 3D normalized numpy array (end diastolic)
    #   - spacing: Numpy spacing format [ z,y,x ]
    # output:
    #   - x-center of left ventricle
    #   - y-center of left ventricle
    
    size_img = img.shape
    # precrop
    img = img[:,int(size_img[1]/2-0.9*size_img[1]/2):int(size_img[1]/2+0.9*size_img[1]/2),:]
    img = img[:,:,int(size_img[2]/2-0.9*size_img[2]/2):int(size_img[2]/2+0.9*size_img[2]/2)]
    
    # select slice
    slice = np.floor(img.shape[0]/2)-2
    frame=img[slice.astype('int'),:, :] #middle_slice.astype(dtype='int')
    
    # calculate center of the image
    size_frame = frame.shape
    center = [size_frame[0]/2,size_frame[1]/2]
    
    ### Parameters Threshold
    Area_tol = 500/(spacing[1]*spacing[2]) #voxels
    Round_tol = 0.45  #Roundness
    Spher_tol = 0.2   #Sphericity
   
    # preprocess image with median filter to correct for field inhomogeneities
    frame_eq=median_filter(frame,4)
        
    seg_img = LV_detection(frame_eq, Area_tol, Round_tol, Spher_tol,3)
    
    # for more than one region select closest to center (due to protocol)
    labeled_img = label(seg_img,connectivity=1,background=0)
    regions = regionprops(labeled_img)

    current_min = np.sqrt(np.square(center[0]-regions[0].centroid[0])+np.square(center[1]-regions[0].centroid[1]))
    current_region = 0
    region_min = 0
    if len(regions)>1:
        for region in regions:
            distance = np.sqrt(np.square(center[0]-region.centroid[0])+np.square(center[1]-region.centroid[1]))
            if (distance<current_min):
                region_min = current_region
                current_min = distance
            current_region = current_region+1
    
    # calculate the center of the left ventricle         
    xcenter = int(np.ceil(regions[region_min].centroid[1])) #caution: reverse dim [1] = x
    ycenter = int(np.ceil(regions[region_min].centroid[0])) #caution: reverse dim [0] = y
    
# uncomment for visualization
# =============================================================================
#     crop = img[:,int(ycenter-72):int(ycenter+72),:]
#     crop = crop[:,:,int(xcenter-72):int(xcenter+72)]
#     
#     plt.figure()
#     plt.imshow(crop[4,:,:],cmap='gray')
#     plt.show()
# =============================================================================

    return xcenter, ycenter
    
 

