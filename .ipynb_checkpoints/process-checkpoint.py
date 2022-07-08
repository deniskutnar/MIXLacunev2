# -*- coding: utf-8 -*-


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

from pmap import *
from RCNN import *
from UNET import *




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(0)


test_data_path = glob.glob(f'input_data/**/')
for x in range(len(test_data_path)):

    flair_path = glob.glob(test_data_path[x]+'/*_FLAIR*')
    t1_path = glob.glob(test_data_path[x]+'/*T1*')
    t2_path = glob.glob(test_data_path[x]+'/*T2*')
    
    im = sitk.ReadImage(t1_path[0])
    

    sub_no = str(t1_path[0])
    sub_no = sub_no.rsplit('/', 1)[-1][0:7]
    print("Loading: T1, T2, Flair\n")


    #-------------------Prevalence map------------------------------
    print("Cretaing prevalence map")
    register(t1_path[0])
    
    #-------------------Prediction RCNN------------------------------
    print("Stage 1")
    RCNN(flair_path[0], t1_path[0], t2_path[0])
    
    #-------------------Prediction UNET------------------------------
    print("Stage 2")
    UNET(flair_path[0], t1_path[0], t2_path[0])
    
    #-------------------UNet pred - Map ------------------------------
    img_unet  = sitk.ReadImage('results/unet_pred.nii.gz')
    arr_unet = sitk.GetArrayFromImage(img_unet)

    img_map = sitk.ReadImage('results/prevalence_map.nii.gz')
    arr_map = sitk.GetArrayFromImage(img_map)

    out_arr = arr_unet + arr_map
    out_im = sitk.GetImageFromArray(out_arr)
    out_im.CopyInformation(im)

    Im = out_im
    BinThreshImFilt = sitk.BinaryThresholdImageFilter()
    BinThreshImFilt.SetLowerThreshold(2)
    BinThreshImFilt.SetUpperThreshold(5)
    BinThreshImFilt.SetOutsideValue(0)
    BinThreshImFilt.SetInsideValue(1)
    BinIm = BinThreshImFilt.Execute(Im)
    
    sitk.WriteImage(BinIm, 'results/unet-prev.nii.gz')
    
    #-------------------UNet pred - (Map) - RCNN ------------------------------
    img_up  = sitk.ReadImage('results/unet-prev.nii.gz')
    arr_up = sitk.GetArrayFromImage(img_up)
    
    img_rcnn = sitk.ReadImage('results/rcnn_pred.nii.gz')
    Im = img_rcnn
    BinThreshImFilt = sitk.BinaryThresholdImageFilter()
    BinThreshImFilt.SetLowerThreshold(0.1)
    BinThreshImFilt.SetUpperThreshold(250)
    BinThreshImFilt.SetOutsideValue(0)
    BinThreshImFilt.SetInsideValue(1)
    Bin_rcnn = BinThreshImFilt.Execute(Im)
    arr_rcnn = sitk.GetArrayFromImage(Bin_rcnn)
    
    out_arr = arr_up + arr_rcnn
    out_im = sitk.GetImageFromArray(out_arr)
    out_im.CopyInformation(im)
    
    Im = out_im
    BinThreshImFilt = sitk.BinaryThresholdImageFilter()
    BinThreshImFilt.SetLowerThreshold(2)
    BinThreshImFilt.SetUpperThreshold(5)
    BinThreshImFilt.SetOutsideValue(0)
    BinThreshImFilt.SetInsideValue(1)
    BinIm = BinThreshImFilt.Execute(Im)
    
    end = '/'+ sub_no + '_space-T1_binary_prediction.nii.gz'
    pred_path = os.path.join('output_data'  + end)
    sitk.WriteImage(BinIm, pred_path)
    
    #new_name = os.path.join('results-'  + sub_no)
    #os.rename("results", new_name)
    
    rem_path = ('results')
    shutil.rmtree(rem_path)
    print("Process done \n")
   

    
    
    
    
    
    
    
    
    

    
  
