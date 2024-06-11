#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:50:00 2022

@author: TripleJ
"""
import numpy as np
import cv2

from skimage import morphology
import time
import os
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--input_dir', dest='InputPath', default='/home/ispl-public/Desktop/Dataset/LIME', help='directory for testing inputs')
parser.add_argument('--output_dir', dest='OutputPath', default='./', help='directory for testing outputs')
parser.add_argument('--gamma_max', dest='gamma_max', type=float, default=6.0, help='gamma_max value')
args = parser.parse_args()

''' pixel value: i2f: 0-255 to 0-1, f2i: 0-1 to 0-255 '''
def i2f(i_image):
    f_image = np.float32(i_image)/255.0
    return f_image

def f2i(f_image):
    i_image = np.uint8(f_image*255.0)
    return i_image

''' Compute 'A' as described by Tang et al. (CVPR 2014) '''
def Compute_A_Tang(im):
    erosion_window = 15
    n_bins = 200

    R = im[:, :, 2]
    G = im[:, :, 1]
    B = im[:, :, 0]

    # compute the dark channel
    dark = morphology.erosion(np.min(im, 2), morphology.square(erosion_window))

    [h, edges] = np.histogram(dark, n_bins, [0, 1])
    numpixel = im.shape[0]*im.shape[1]
    thr_frac = numpixel*0.99
    csum = np.cumsum(h)
    nz_idx = np.nonzero(csum > thr_frac)[0][0]
    dc_thr = edges[nz_idx]
    mask = dark >= dc_thr
    # similar to DCP till this step
    # next, median of these top 0.1% pixels
    # median of the RGB values of the pixels in the mask
    rs = R[mask]
    gs = G[mask]
    bs = B[mask]

    A = np.zeros((1,3))

    A[0, 2] = np.median(rs)
    A[0, 1] = np.median(gs)
    A[0, 0] = np.median(bs)

    return A

def GetIntensity(fi):
    return cv2.divide(fi[:, :, 0] + fi[:, :, 1] + fi[:, :, 2], 3)

def GetMax(fi):
    max_rgb = cv2.max(cv2.max(fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    return max_rgb

def GetMin(fi):
    min_rgb = cv2.max(cv2.min(fi[:, :, 0], fi[:, :, 1]), fi[:, :, 2])
    return min_rgb

''' Pixel Adaptive Gamma Correction '''
def PixelAdaptiveGamma(InputImg, NormImg, amax):
    GCImg = np.empty(InputImg.shape, InputImg.dtype)
    amin = 1
    xmax = 1
    xmin = 0
    Imax = GetMax(InputImg)

    a = (amax - amin) / (np.exp(-xmin) - np.exp(-xmax))
    b = amax - a * np.exp(-xmin)
    g2 = a * np.exp(-Imax) + b
    g1 = np.where(Imax < xmin, amax, g2)
    gamma = np.where(Imax > xmax, amin, g1)

    for ind in range(0, 3):
        GCImg[:, :, ind] = NormImg[:, :, ind] ** gamma

    return GCImg

''' Estimate Transmission Map '''
def EstimateTransmission(InputImg, NormImg, gamma_max=6):
    T_min = 0.1
    me = np.finfo(np.float32).eps
    hi = GetIntensity(NormImg)
    hmax = GetMax(NormImg)

    GCImg = PixelAdaptiveGamma(InputImg, NormImg, gamma_max)

    ji = GetIntensity(GCImg)
    jmax = GetMax(GCImg)

    tn = np.maximum(jmax * hi - hmax * ji, me)
    td = np.maximum((jmax - ji) * hi, me)

    Tmap = 1.0 - hi * (tn / td)
    
    return np.clip(Tmap, T_min, 1.0)

''' Recover dehazed image '''
def Recover(im, tmap, A):
    res = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - 1.0 + A[0, ind]) / tmap + 1.0 - A[0, ind]

    return np.clip(res, 0.0, 1.0)

''' Adjust image range '''
def Adjust(im, perh, perl):
    aim = np.empty(im.shape, im.dtype)
    im_h = np.percentile(im, perh)
    im_l = np.percentile(im, perl)
    for ind in range(0, 3):
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)

    return np.clip(aim, 0.0, 1.0)

''' Normalize image 0 between 1 '''
def Normalize(im):
    aim = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im_h = np.max(im[:, :, ind])
        im_l = np.min(im[:, :, ind])
        aim[:, :, ind] = (im[:, :, ind] - im_l) / (im_h - im_l)
        aim[:, :, ind] = np.clip(aim[:, :, ind], 0.0, 1.0)

    return aim

''' Main '''
def main(InputImg, gamma_max):
    start_time = time.time()
######################################################
    float_image = i2f(InputImg)
    NormImg = np.empty(float_image.shape, float_image.dtype)
    DenoiseImg = 1.0 - cv2.GaussianBlur(float_image, (7, 7), 0)

    A = Compute_A_Tang(DenoiseImg)
    for ind in range(0, 3):
        NormImg[:, :, ind] = DenoiseImg[:, :, ind] / A[0, ind]
    NormImg = Normalize(NormImg)
    
    Transmap = EstimateTransmission(float_image, NormImg, gamma_max)
    RecoverImg = Recover(float_image, Transmap, A)

    AdjustImg = Adjust(RecoverImg, 99.5, 0.5)

######################################################
    end_time = time.time()
    print("--- %s seconds ---" % (end_time - start_time))
    
    return AdjustImg


if __name__ == "__main__":
    FolderTemp = args.InputPath.split('/')
    OutputPath = args.OutputPath + '/Output/' + FolderTemp[-1]
    os.makedirs(OutputPath, exist_ok=True, mode=0o777)
    
    FileList = [file for file in os.listdir(args.InputPath) if
                    (file.endswith(".jpg") or file.endswith(".JPG") or file.endswith(".jpeg"))
                    or file.endswith(".webp") or file.endswith(".tiff") or file.endswith(".tif")
                    or file.endswith(".bmp") or file.endswith(".png")]

    for FileNum in range(0, len(FileList)):
        FilePathName = args.InputPath + '/' + FileList[FileNum]
        InputImg = cv2.imread(FilePathName, cv2.IMREAD_COLOR)
    
        OutputImg = main(InputImg, args.gamma_max)

        Name = os.path.splitext(FileList[FileNum])
        cv2.imwrite(OutputPath + '/' + Name[0] + '.png', f2i(OutputImg))
