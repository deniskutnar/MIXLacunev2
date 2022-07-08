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

from netgen import *


def UNET(inp_flair, inp_t1, inp_t2):
    path_flair = inp_flair
    path_t1 = inp_t1
    path_t2 = inp_t2
    

    model = UNet(in_channels=3,
                 out_channels=1,
                 n_blocks=4,
                 start_filters=32,
                 activation='leaky',
                 normalization='instance',
                 conv_mode='same',
                 dim=3)
    
    model.load_state_dict(torch.load('model_UNet.pt'))
    model.cuda()
   
    
       
    def read_image(path):
        img = sitk.ReadImage(path)
        img_as_numpy = sitk.GetArrayFromImage(img).astype('float32')
        img_as_tensor = torch.from_numpy(img_as_numpy)
        img_as_tensor = img_as_tensor.unsqueeze(0)
        return img_as_tensor

    def read_zscore(path):
        nib_img = nib.load(path)
        normal = zscore_normalize(nib_img)
        normal =  normal.get_fdata()
        normal = normal.astype(np.float32)
        img_as_tensor = torch.from_numpy(normal)
        img_as_tensor = img_as_tensor.permute(2,1,0)
        img_as_tensor = img_as_tensor.unsqueeze(0)
        #img_as_numpy = img_as_tensor.numpy()
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
    
    from torch.autograd import Variable
    def Unet_pred(inp_cube):
        model.eval()
        cube = inp_cube
        cube = cube.unsqueeze(0)
        cube = Variable(cube.cuda())
        y_pred = model(cube)
        y_pred = torch.sigmoid(y_pred)
        y_pred = y_pred.cpu().data.numpy() 
        return y_pred
    
    
    import torch.nn.functional as F
    def patchingXX(inp_x):
        x = inp_x 
        x = F.pad(x, (0, 0, 0, 0, 0, 0))  
        # Make pacthes 
        kc, kd, kh, kw = 4, 32, 32, 32  # kernel size
        dc, dd, dh, dw = 4, 32, 32, 32  # stride
        # Pad to multiples of 32
        x = F.pad(x, (x.size(3)%kw // 2, x.size(3)%kw // 2,
                      x.size(2)%kh // 2, x.size(2)%kh // 2,
                      x.size(1)%kd // 2, x.size(1)%kd // 2,
                      x.size(0)%kc // 2, x.size(0)%kc // 2))
        patches = x.unfold(1, kc, dc).unfold(2, kd, dd).unfold(3, kh, dh).unfold(4, kw, dw)
        unfold_shape = patches.size()
        patches = patches.contiguous().view(-1, kc, kd, kh, kw)

        # U-Net prediction on cubes
        rec_vol = torch.zeros(0,4,32,32,32)
        for x in range(len(patches)):
            one = patches[x,:, :, :, :]
            if one[3,:,:,:].max()==0:
                one = one.unsqueeze(0)
                rec_vol = torch.cat((rec_vol, one),0)
            elif one[3,:,:,:].max()!=0:
                data = one[:3,:,:,:]
                pred = Unet_pred(data)
                pred = torch.from_numpy(pred)
                pred = pred.squeeze(0)
                full = torch.cat((data,pred),0)
                full = full.unsqueeze(0)
                rec_vol = torch.cat((rec_vol, full),0)        
       
        # Reshape back
        patches_orig = rec_vol.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[5]
        output_d = unfold_shape[2] * unfold_shape[6]
        output_h = unfold_shape[3] * unfold_shape[7]
        output_w = unfold_shape[4] * unfold_shape[8]
        patches_orig = patches_orig.permute(0, 1, 5, 2, 6, 3, 7, 4, 8).contiguous()
        patches_orig = patches_orig.view(1, output_c, output_d, output_h, output_w)
        return patches_orig
    
    
    
    flair = read_zscore(path_flair)
    t1 = read_zscore(path_t1)
    t2 = read_zscore(path_t2)
    hot = read_image('results/rcnn_pred.nii.gz')
    
    tensor = torch.cat((flair,t1,t2, hot),0)
    tensor = tensor.unsqueeze(0)
    im = sitk.ReadImage(path_t1)
    
    
    pat = patchingXX(tensor)
    pmask = pat[0,3,:,:,:]
    out_im = sitk.GetImageFromArray(pmask)
    out_im.CopyInformation(im)

    Im = out_im
    BinThreshImFilt = sitk.BinaryThresholdImageFilter()
    BinThreshImFilt.SetLowerThreshold(0.1)
    #BinThreshImFilt.SetUpperThreshold(2)
    BinThreshImFilt.SetOutsideValue(0)
    BinThreshImFilt.SetInsideValue(1)
    BinIm = BinThreshImFilt.Execute(Im)
    
    
    sitk.WriteImage(BinIm, 'results/unet_pred.nii.gz')
    #sitk.WriteImage(out_im, 'results/unet_pred-binMAP.nii.gz')

