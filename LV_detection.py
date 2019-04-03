import numpy as np
from skimage.measure import label,regionprops
import skfuzzy as fuzz
import matplotlib.pyplot as plt

def LV_detection(img, area_tol, round_tol, spher_tol,n):

    # Fuzzy Clustering Means with reshape to list and reshape back to orginal size
    dim=img.shape
    img = np.ndarray.transpose(img.reshape([dim[0]*dim[1],1]).astype('float64'))
    cluster_img=np.zeros([dim[0]*dim[1],1])
    cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(img,n,2,0.001,1000)
    for row in range(0,dim[0]*dim[1]-1):
        index = np.argmax(u[:,row])
        cluster_img[row]=index
    cluster_img = cluster_img.reshape([dim[0],dim[1]])
    
    # to make sure that background is always labeled as 'group 0'
    background = cluster_img[0,0]
    if background!=0:
        cluster_img = np.where(cluster_img==0, -1, cluster_img)
        cluster_img = np.where(cluster_img==background, 0, cluster_img)
        cluster_img = np.where(cluster_img==-1, background, cluster_img) 
        
# uncomment to visualize final result of Fuzzy Cluster Means Algorithm
# =============================================================================
#     plt.imshow(cluster_img, cmap='gray')
#     plt.show()
# =============================================================================
    
    # Selection of region based on thresholds for circularity, sphericity and area
    labeled_img = label(cluster_img,connectivity=1,background=0)
    regions = regionprops(labeled_img)
    detected_img=cluster_img
    for region in regions:
        if region.filled_area < area_tol :
            for coord in region.coords:
                detected_img[coord[0],coord[1]] = 0
        circularity = (region.perimeter*region.perimeter)/(4*3.1415*region.filled_area)
        sphericity = (2*np.sqrt(3.1415*region.filled_area))/region.perimeter
        if circularity<=round_tol:
            for coord in region.coords:
                detected_img[coord[0],coord[1]] = 0
        if sphericity <= spher_tol:
            for coord in region.coords:
                detected_img[coord[0],coord[1]]=0

# uncomment to visualize left over regions after all thresholds
# =============================================================================
#     plt.imshow(detected_img)
#     plt.show()
# =============================================================================
    return detected_img