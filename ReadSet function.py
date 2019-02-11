import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt # matplotlib is a nice library for plotting
from unet import * #the u-net model is in this file


def ReadSet(begin, end, datapath):
    ED=[]
    gt_ED=[]
    ES=[]
    gt_ES=[]
    nrslices=[]
    
    #generate the patient number as it is used in the filenames ("001" etc.)
    for patient in range(begin,end+1):
        if patient<10:
            patientnumber="00"+str(patient)
        elif patient<100:
            patientnumber="0"+str(patient)
        elif patient<1000:
            patientnumber=str(patient)
        #print(patientnumber)
        
        #open the patient info file
        info=open(datapath+"patient"+patientnumber+"/info.cfg")
        
        #from the patient info file, read the ED and ES frame numbers
        EDline=info.readline()
        ESline=info.readline()
        EDframe=EDline[-3:]
        ESframe=ESline[-3:]
        EDlist=list(EDframe)
        if EDlist[0]==' ':
            EDlist[0]="0"
        patientED=''.join(EDlist[0:2])
        ESlist=list(ESframe)
        if ESlist[0]==' ':
            ESlist[0]="0"
        patientES=''.join(ESlist[0:2])
        
        #generate the filenames of the ED and ES images and ground truths
        EDfilename=datapath+"patient"+patientnumber+"/patient"+patientnumber+"_frame"+patientED+".nii.gz"
        ESfilename=datapath+"patient"+patientnumber+"/patient"+patientnumber+"_frame"+patientES+".nii.gz"
        EDgt=datapath+"patient"+patientnumber+"/patient"+patientnumber+"_frame"+patientED+"_gt.nii.gz"
        ESgt=datapath+"patient"+patientnumber+"/patient"+patientnumber+"_frame"+patientES+"_gt.nii.gz"
        
        #read in these images and turn them into numpy
        im_ED=sitk.ReadImage(EDfilename)
        im_ED_np=sitk.GetArrayFromImage(im_ED)
        im_ES=sitk.ReadImage(ESfilename)
        im_ES_np=sitk.GetArrayFromImage(im_ES)
        im_EDgt=sitk.ReadImage(EDgt)
        im_EDgt_np=sitk.GetArrayFromImage(im_EDgt)
        im_ESgt=sitk.ReadImage(ESgt)
        im_ESgt_np=sitk.GetArrayFromImage(im_ESgt)
        
        #define the number of slices (might be used later on)
        nrslices.append(im_ED_np.shape[0])
        
        #binarize the ground truth data, only label 3 is of importance        
        im_EDgt_np[im_EDgt_np==1]=0
        im_EDgt_np[im_EDgt_np==2]=0
        im_EDgt_np[im_EDgt_np==3]=1
    
        im_ESgt_np[im_ESgt_np==1]=0
        im_ESgt_np[im_ESgt_np==2]=0
        im_ESgt_np[im_ESgt_np==3]=1
        
        #crop the images, so less data is used and the model will be faster
        #the images are cropped to a 160x160 images in the middle of the original image
        cropped_ED=im_ED_np[:,48:208,27:187]
        cropped_EDgt=im_EDgt_np[:,48:208,27:187]
        cropped_ES=im_ES_np[:,48:208,27:187]
        cropped_ESgt=im_ESgt_np[:,48:208,27:187]
        
        #add all patient's images and ground truths to the lists will all the data
        ED.append(cropped_ED)
        gt_ED.append(cropped_EDgt)
        ES.append(cropped_ES)
        gt_ES.append(cropped_ESgt)

                
    return ED, gt_ED, ES, gt_ES, nrslices

datapath="C:/Users/s140937/Documents/ME 1/Team Challenge/DEEL 2/TeamChallenge/Data/"
trainED,trainED_labels,trainES,trainES_labels,train_nrslices=ReadSet(1,5,datapath)

train = np.array(trainED,dtype='float32') #as mnist
train_labels = np.array(trainED_labels,dtype='float64') #as mnist

# np.save('train',train)
# np.save('train_labels',train_labels)


# plt.figure()
# plt.imshow(trainED[0][4,:,:], clim=(0,150), cmap='gray')
# plt.show()
