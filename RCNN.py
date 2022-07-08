import os
import torch
import torchvision
import numpy as np

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import SimpleITK as sitk
import glob

import torch.nn as nn
import nibabel as nib

import shutil  
import cv2



def RCNN(inp_flair, inp_t1, inp_t2):
    path_flair = inp_flair
    path_t1 = inp_t1
    path_t2 = inp_t2
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(2)
    
    

    model = torch.load('model_RCNN.pt', map_location={'cuda:1':'cuda:0'})
    
    #model = torch.load('model_RCNN.pt')
    model.to(device)
   
    
  
    
    im = sitk.ReadImage(path_t1)
       
    
    def read_image(path):
        img = sitk.ReadImage(path)
        img_as_numpy = sitk.GetArrayFromImage(img).astype('float32')
        img_as_tensor = torch.from_numpy(img_as_numpy)
        img_as_tensor = img_as_tensor.unsqueeze(1)
        return img_as_tensor
    
    
    def read_zscore(path):
        nib_img = nib.load(path)
        normal = zscore_normalize(nib_img)
        normal =  normal.get_fdata()
        normal = normal.astype(np.float32)
        img_as_tensor = torch.from_numpy(normal)
        img_as_tensor = img_as_tensor.permute(2,1,0)
        img_as_tensor = img_as_tensor.unsqueeze(1)
        return img_as_tensor
    
    def zscore_normalize(img, mask=None):

        """
        normalize a target image by subtracting the mean of the whole brain
        and dividing by the standard deviation
        Args:
            img (nibabel.nifti1.Nifti1Image): target MR brain image
            mask (nibabel.nifti1.Nifti1Image): brain mask for img
        Returns:
            normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
        """

        img_data = img.get_fdata()
        if mask is not None and not isinstance(mask, str):
            mask_data = mask.get_fdata()
        elif mask == 'nomask':
            mask_data = img_data == img_data
        else:
            mask_data = img_data > img_data.mean()
        logical_mask = mask_data > 0.  # force the mask to be logical type
        mean = img_data[logical_mask].mean()
        std = img_data[logical_mask].std()
        normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
        return normalized
    
    
    def getMinMax(inp_resp):
        response = inp_resp
        min_rx = []
        min_ry = []
        max_rx = []
        max_ry = []
        boxB = []
        for f in range(len(response)): 
            min_rx.append(response[f][0])
            min_ry.append(response[f][1])
            max_rx.append(response[f][2])
            max_ry.append(response[f][3])
        new_xmin = min(min_rx)
        new_ymin = min(min_ry)
        new_xmax = max(max_rx)
        new_ymax = max(max_ry)
        boxB = [[int(new_xmin), int(new_ymin), int(new_xmax), int(new_ymax)]]
        return boxB
    
    
    def get_multiBoxB(inp_img_slice):
        img_slice = inp_img_slice
        model.eval()
        with torch.no_grad():
            prediction = model([img_slice.to(device)])
            multiBoxB = prediction[0]['boxes']
        scores = prediction[0]['scores']
        scores_cut = []
        for f in range(len(scores)):
            if (scores[f] > 0.45):
                scores_cut.append(scores[f])
        cut = len(scores_cut)
        multiBoxB =  multiBoxB[:cut]
        return multiBoxB
    
    
    def get_4channel(inp_boxB):
        boxB = inp_boxB
        #temp = np.zeros((256,256))
        temp = np.zeros((512,512))
        j = 0
        channel4 = cv2.rectangle(temp,(  boxB[j][0], boxB[j][1] ),
                                    (  boxB[j][2], boxB[j][3] ), (150,255,255), -1)
        channel4 = torch.from_numpy(channel4)
        channel4 = channel4.unsqueeze(0)
        return channel4
    
    
    
    t1 = read_zscore(path_t1)
    t2 = read_zscore(path_t2)
    flair = read_zscore(path_flair)
    tensor = torch.cat((t1, t2, flair),1)

    rec_vol = torch.zeros(0,1,512,512)
    for f in range(len(tensor)):
        # ------ One Slice ---------
        img_slice = tensor[f,:,:,:]
        # get all predictions above 0.45
        multiBoxB = get_multiBoxB(img_slice)
        if len(multiBoxB) == 0:
            empty = torch.zeros(1,1,512,512)
            rec_vol = torch.cat((rec_vol, empty),0)
        elif len(multiBoxB) != 0:
            # fuse all predictions by taking min & max
            boxB = getMinMax(multiBoxB)
            # get the 4. channel by taking the BB
            full_slice = get_4channel(boxB)
            full_slice = full_slice.unsqueeze(0)
            # put slices together
            rec_vol = torch.cat((rec_vol, full_slice),0)
    
    
    


    roi = rec_vol[:,0,:,:]
    out_im = sitk.GetImageFromArray(roi)
    out_im.CopyInformation(im)
    sitk.WriteImage(out_im, 'results/rcnn_pred.nii.gz')