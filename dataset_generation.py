
"""
Created on Mon Feb 24 13:30:33 2020

@author: cv504
"""

import os
from typing import Tuple
import numpy as np
from random import shuffle, randint
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
import scipy.io as sio
from keras.utils import to_categorical
import random
import scipy
from keras.layers import concatenate
from sklearn.decomposition import PCA, KernelPCA
from spectral import *
import spectral.io.aviris as aviris
import matplotlib.pyplot as plt
import math
import copy
import cv2

os.makedirs('grids', exist_ok=True)

dataname="PaviaU"
# dataname = "SalinasV"
# dataname="indianpines"
# paviaU
if dataname == "PaviaU":
    ordata = "PaviaU.mat"
    orindata = "paviaU"
    orlabel = 'PaviaU_gt.mat'
    orinlabel = 'paviaU_gt'
    data_name = "PaviaU"
    datapath = "./data/PaviaU.mat"
    ground_truth_path = "./data/PaviaU_gt.mat"
    window_size = 10
    batch_size = 8
    allclass_num = 10  # 9+1class_num = 9
    channels = 103
    numpca = 15
    numlabel = 3
    width = 340
    height = 610
    class_num = 9
    pixels = [0, 6631, 18649, 2099, 3064, 1345, 5029, 1330, 3682, 947]
    t_pixels = [0, 400, 400, 400, 400, 400, 400, 400, 400, 400]
    min_num = 2700
    max_num = 2800
    pixel_ratio = 0.075  # 0.077
    val_ratio = 0.020  # 0.019
    num_eachwindow = 25
    savedata = "predata\PU10-8\PU_1"
# indianpines
if dataname == "indianpines":
    ordata = "Indian_pines_corrected.mat"
    orindata = "indian_pines_corrected"
    orlabel = 'Indian_pines_gt.mat'
    orinlabel = 'indian_pines_gt'
    data_name = "Indian_pines"
    datapath = "./data/Indian_pines_corrected.mat"
    ground_truth_path = "./data/Indian_pines_gt.mat"
    window_size = 6
    batch_size = 4
    allclass_num = 17  # 16+1
    channels = 220
    numpca = 30
    numlabel = 3
    width = 145
    height = 145
    class_num = 16
    pixels = [0, 46, 1428, 830, 237, 483, 730, 28, 478, 20, 972, 2455, 593, 205, 1265, 386, 93]
    t_pixels = [0, 10, 150, 100, 20, 60, 100, 10, 50, 10, 100, 200, 60, 30, 120, 40, 20]
    min_num = 1100
    max_num = 1200
    pixel_ratio = 0.28  # 0.15
    val_ratio = 0.25  # 0.06
    num_eachwindow = 9
    savedata = "predata\IP6-4\IP_1"
# salinas valley
if dataname == "SalinasV":
    ordata = "Salinas_valley.mat"
    orindata = 'salinas'
    orlabel = 'Salinas_valley_gt.mat'
    orinlabel = 'salinas_gt'
    data_name = "Salinas_valley"
    datapath = "./data/Salinas_valley.mat"
    ground_truth_path = "./data/Salinas_valley_gt.mat"
    window_size = 10
    batch_size = 8
    allclass_num = 17  # 16+1
    channels = 204
    numpca = 15
    numlabel = 3
    width = 217
    height = 512
    class_num = 16
    pixels = [0, 2009, 3726, 1976, 1394, 2678, 3959, 3579, 11271, 6203, 3278, 1068, 1927, 916, 1070, 7268, 1807]
    t_pixels = [0, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180, 180]
    min_num = 2000
    max_num = 2100
    pixel_ratio = 0.045 # 0.043
    val_ratio = 0.0043 # 0.0052
    num_eachwindow = 16
    savedata = "predata\SV10-8\SV_1"
delt_w = int((window_size - 1) / 2)
gt_all_pixels = [[] for i in range(allclass_num)]
Center_pixel = [[] for i in range(allclass_num)]
Center_pixel_val = [[] for i in range(allclass_num)]
num_Train = [0 for i in range(allclass_num)]
num_Train_op = [0 for i in range(allclass_num)]
num_Train_val = [0 for i in range(allclass_num)]
num_Train_test = [0 for i in range(allclass_num)]
train_patches, train_patches_gt, op_train_patches, op_train_patches_gt, val_patches, val_patches_gt, test_patches, test_patches_gt = [], [], [], [], [], [], [], []


class WindowSize(object):
    def __init__(self, x: int, y: int):
        if not x > 0 or not y > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x, y))
        elif not isinstance(x, int) or not isinstance(y, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(type(x),
                                                                      type(y)))
        self.x = x
        self.y = y


class Stride(object):
    def __init__(self, x_stride: int, y_stride: int):
        if not x_stride > 0 or not y_stride > 0:
            raise ValueError(
                "x and y should be positive, were ({} {})".format(x_stride,
                                                                  y_stride))
        elif not isinstance(x_stride, int) or not isinstance(y_stride, int):
            raise TypeError(
                "x and y have to be integers, were: {} and {}".format(
                    type(x_stride),
                    type(y_stride)))
        self.x = x_stride
        self.y = y_stride


class Patch:
    def __init__(self,
                 index: int,
                 left_x: int,
                 right_x: int,
                 upper_y: int,
                 lower_y: int):
        self.index = index
        self.left_x = left_x
        self.right_x = right_x
        self.upper_y = upper_y
        self.lower_y = lower_y


#  apply PCA preprocessing for data sets
def applyPCA(X, numComponents=75):
    newX = np.reshape(X, (-1, X.shape[2]))
    pca = PCA(n_components=numComponents, whiten=True)
    newX = pca.fit_transform(newX)
    newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
    return newX, pca.explained_variance_ratio_


def Windowc(Center_pixel):
    patches = []
    index = 0
    for i in range(1, len(Center_pixel)):
        for j in range(len(Center_pixel[i])):
            left_border_x = Center_pixel[i][j][0] - (window_size - 1) / 2
            right_border_x = Center_pixel[i][j][0] + (window_size - 1) / 2 + 1
            upper_border_y = Center_pixel[i][j][1] - (window_size - 1) / 2
            lower_border_y = Center_pixel[i][j][1] + (window_size - 1) / 2 + 1
            patch = Patch(index, int(left_border_x), int(right_border_x), int(upper_border_y), int(lower_border_y))
            patches.append(patch)
            index = index + 1
    return patches


def allpixels(gt):
    for i in range(delt_w, gt.shape[0] - delt_w):
        for j in range(delt_w, gt.shape[1] - delt_w):
            if gt[i, j] != 0:
                gt_all_pixels[gt[i, j]].append([i, j])
    #                gt[i,j]=26
    return gt_all_pixels


def full(full, full_gt, allusepixels):
    newfull = []
    newfull_gt = []
    print(np.array(full).shape)
    for i in range(len(full_gt)):
        cp = list(set(full_gt[i].flatten()))
        if 0 in cp:
            for y in range(len(cp)):
                _full = copy.deepcopy(full)
                _full_gt = copy.deepcopy(full_gt)
                if cp[y] == 0:
                    continue
                for j in range(len(full_gt[i])):
                    for x in range(len(full_gt[i][j])):
                        if np.array(full_gt)[i, j, x] == 0:
                            _full[i][j][x] = random.choice(allusepixels[cp[y]])
                            _full_gt[i][j][x] = cp[y]
                newfull.append(_full[i])
                newfull_gt.append(_full_gt[i])
        else:
            newfull.append(full[i])
            newfull_gt.append(full_gt[i])
    return newfull, newfull_gt


def cut_pixels(window_gt):
    # half pixel
    window_gt_flatten = window_gt.flatten()

    op_window_gt_flatten = window_gt.flatten()
    nozero = np.nonzero(window_gt_flatten)
    nozero = nozero[0]
    if num_eachwindow < len(nozero):
        ls = list(range(0, len(nozero)))
        c_ls = random.sample(ls, math.floor(num_eachwindow))
        for i in c_ls:
            op_window_gt_flatten[nozero[i]] = 0
            ls.remove(i)
        for i in ls:
            window_gt_flatten[nozero[i]] = 0
    else:
        op_window_gt_flatten = np.zeros(len(op_window_gt_flatten))
    #    print("_____",window_gt_flatten.shape,op_window_gt_flatten.shape)
    return window_gt_flatten.reshape((window_size, window_size)), op_window_gt_flatten.reshape(
        (window_size, window_size))


def sliding_window(image: np.ndarray, window_size: WindowSize,
                   stride: Stride = 0):
    number_of_patches_in_x = int(((image.shape[0] - window_size) / stride) + 1)
    number_of_patches_in_y = int(((image.shape[1] - window_size) / stride) + 1)
    patches = []
    index = 0
    for x_dim_patch_number in range(0, number_of_patches_in_x):
        for y_dim_patch_number in range(0, number_of_patches_in_y):
            left_border_x = int(0 + stride * x_dim_patch_number)
            right_border_x = int(window_size + stride * x_dim_patch_number)
            upper_border_y = int(0 + stride * y_dim_patch_number)
            lower_border_y = int(window_size + stride * y_dim_patch_number)
            patch = Patch(index, left_border_x, right_border_x, upper_border_y, lower_border_y)
            patches.append(patch)
            index += 1
    return patches


def extract_grids(dataset_path: str, ground_truth_path: str, window_size: int):
    allusepixels = [[] for i in range(allclass_num)]
    allusepixels_val = [[] for i in range(allclass_num)]
    train_patches = []
    train_patches_gt = []
    op_train_patches = []
    op_train_patches_gt = []
    train_patches_gt_c = []
    val_patches = []
    val_patches_gt = []
    val_patches_gt_c = []
    test_patches = []
    test_patches_gt = []
    test_patches_gt_c = []
    data_path = os.path.join(os.getcwd(), 'data')
    print("__________Loading_________")
    print("__________Waiting_________")
    input_data = sio.loadmat(os.path.join(data_path, ordata))[orindata]
    gt_ori = sio.loadmat(os.path.join(data_path, orlabel))[orinlabel]
    #    ground_truth = imshow(data=input_data.astype(int), figsize=(6,6))
    #    gt=gt_ori.copy()
    #    gt_val=gt_ori.copy()
    # 扩充label维度
    #    gt0 = [gt_ori for K in range(numlabel)]
    #    print(np.array(gt0).shape)
    #    gt0 = np.reshape (gt0,(numlabel,height,width))
    #    gt0 = np.transpose(gt0,(1,2,0))

    gt_allpixels = allpixels(gt_ori)
    for i in range(1, len(gt_allpixels)):
        N_block = int(np.ceil(pixels[i] * pixel_ratio / num_eachwindow))
        N_block_val = int((np.ceil(pixels[i] * val_ratio / num_eachwindow)))
        #        N_space = len(gt_edge[i])/N_block
        for j in range(N_block):
            Center_pixel[i].append(gt_allpixels[i][np.random.randint(0, len(gt_allpixels[i]))])

        for x in range(N_block_val):
            Center_pixel_val[i].append(gt_allpixels[i][np.random.randint(0, len(gt_allpixels[i]))])
    # 画中心像素
    #    for i in range(gt0.shape[0]):
    #        for j in range(gt0.shape[1]):
    #            for x in range(1,len(Center_pixel)):
    #                for y in range(len(Center_pixel[x])):
    #                    if Center_pixel[x][y] == [i,j]:
    #                        gt0[i,j]=26

    patches = Windowc(Center_pixel)
    n = 0
    gt = np.zeros((height, width), int)
    gt_op = np.zeros((height, width), int)
    gt_val = np.zeros((height, width), int)
    x = 0
    for patch in patches:
        train_patch = input_data[patch.left_x:patch.right_x, patch.upper_y:patch.lower_y, :].copy()
        train_patch_gt = gt_ori[patch.left_x:patch.right_x, patch.upper_y:patch.lower_y].copy()
        if train_patch_gt.shape[0] == window_size and train_patch_gt.shape[1] == window_size:
            # remain half pixels
            train_patch_gt, op_train_patch_gt = cut_pixels(train_patch_gt)
            # back unuse pixels
            slipatches = sliding_window(train_patch, batch_size, 1)

            for slipatch in slipatches:
                train = train_patch[slipatch.left_x:slipatch.right_x, slipatch.upper_y:slipatch.lower_y, :].copy()
                train_gt = train_patch_gt[slipatch.left_x:slipatch.right_x, slipatch.upper_y:slipatch.lower_y].copy()
                op_train_gt = op_train_patch_gt[slipatch.left_x:slipatch.right_x,
                              slipatch.upper_y:slipatch.lower_y].copy()
                #                if x<10:
                #                    ground_truth = imshow(classes=train_gt.astype(int), figsize=(3,3))
                #                    ground_truth = imshow(classes=op_train_gt.astype(int), figsize=(3,3))
                #                    x= x+1
                #            train_gt_show = train_patch_gt[slipatch.left_x:slipatch.right_x,slipatch.upper_y:slipatch.lower_y].copy()
                # show纯色块
                #            nonzero = np.unique(train_gt)
                #            if len(nonzero) == 1 and train_gt[1,1] != 0 :
                ##                ground_truth = imshow(classes=train_gt.astype(int), figsize=(3,3))
                #                print(train_gt[1,1])
                #                n=n+1
                #                print(n)
                # 去除全0块
                nonzero = np.count_nonzero(train_gt)
                if nonzero == 0:
                    continue
                if n % 2 == 0:
                    train_patches.append(train)
                    train_patches_gt.append(train_gt)
                    train_patches, train_patches_gt = AugmentData(train, train_gt, train_patches, train_patches_gt)
                    if n % 1 == 0:
                        op_train_patches.append(train)
                        op_train_patches_gt.append(op_train_gt)
                    n = n + 1
                else:
                    #                val_patches.append(train)
                    #                val_patches_gt.append(train_gt)
                    n = n + 1
            #    for patch in patches:
            # 绘制方块
            #        gt[patch.left_x:patch.right_x,patch.upper_y:patch.lower_y] = 26

            gt[patch.left_x:patch.right_x, patch.upper_y:patch.lower_y] = train_patch_gt.copy()
            gt_op[patch.left_x:patch.right_x, patch.upper_y:patch.lower_y] = op_train_patch_gt.copy()

    # 计数train num
    for i in range(gt_ori.shape[0]):
        for j in range(gt_ori.shape[1]):
            if gt[i, j] != 0:
                num_Train[gt[i, j]] = num_Train[gt[i, j]] + 1
    for i in range(gt_ori.shape[0]):
        for j in range(gt_ori.shape[1]):
            if gt_op[i, j] != 0:
                num_Train_op[gt_op[i, j]] = num_Train_op[gt_op[i, j]] + 1
                # clear train gt
    for patch in patches:
        gt_ori[patch.left_x:patch.right_x, patch.upper_y:patch.lower_y] = 0  # op_train_patch_gt.copy()

    # 消除trainpatch的gt
    #    for patch in patches:
    #        gt_ori[patch.left_x:patch.right_x,patch.upper_y:patch.lower_y] = 0

    # *************************val***********************
    tn = 0
    patches_val = Windowc(Center_pixel_val)
    for patch_val in patches_val:
        val_patch = input_data[patch_val.left_x:patch_val.right_x, patch_val.upper_y:patch_val.lower_y, :].copy()
        val_patch_gt = gt_ori[patch_val.left_x:patch_val.right_x, patch_val.upper_y:patch_val.lower_y].copy()
        if val_patch_gt.shape[0] == window_size and val_patch_gt.shape[1] == window_size:
            #        val_patch_gt00,op_val_patch_gt = cut_pixels(val_patch_gt)
            gt_val[patch_val.left_x:patch_val.right_x, patch_val.upper_y:patch_val.lower_y] = val_patch_gt.copy()
            #        gt_ori[patch_val.left_x:patch_val.right_x,patch_val.upper_y:patch_val.lower_y]=op_val_patch_gt.copy()
            slipatches_val = sliding_window(val_patch, batch_size, 1)
            for slipatch_val in slipatches_val:
                val = val_patch[slipatch_val.left_x:slipatch_val.right_x, slipatch_val.upper_y:slipatch_val.lower_y,
                      :].copy()
                val_gt = val_patch_gt[slipatch_val.left_x:slipatch_val.right_x,
                         slipatch_val.upper_y:slipatch_val.lower_y].copy()
                nonzero = np.count_nonzero(val_gt)
                if nonzero == 0:
                    continue
                val_patches.append(val)
                val_patches_gt.append(val_gt)
    #                for i in range(val_gt.shape[0]):
    #                    for j in range(val_gt.shape[1]):
    #                        tn=tn+1
    #                        if val_gt[i,j] != 0:
    #                            x=x+1
    #        print(x,tn,x/tn)
    # 计数train num
    for i in range(gt_ori.shape[0]):
        for j in range(gt_ori.shape[1]):
            if gt_val[i, j] != 0:
                num_Train_val[gt_val[i, j]] = num_Train_val[gt_val[i, j]] + 1
                gt_ori[i, j] = 0
    for i in range(gt_ori.shape[0]):
        for j in range(gt_ori.shape[1]):
            if gt_ori[i, j] != 0:
                num_Train_test[gt_ori[i, j]] = num_Train_test[gt_ori[i, j]] + 1
            # ***************************test****************************

    all_test_patches = sliding_window(input_data, batch_size, batch_size)
    for _patch in all_test_patches:
        nonzero = np.count_nonzero(gt_ori[_patch.left_x:_patch.right_x, _patch.upper_y:_patch.lower_y])
        if nonzero == 0:
            continue
        test_patch = input_data[_patch.left_x:_patch.right_x, _patch.upper_y:_patch.lower_y, :].copy()
        test_patch_gt = gt_ori[_patch.left_x:_patch.right_x, _patch.upper_y:_patch.lower_y].copy()
        test_patches.append(test_patch)
        test_patches_gt.append(test_patch_gt)

    return train_patches, train_patches_gt, op_train_patches, op_train_patches_gt, val_patches, val_patches_gt, test_patches, test_patches_gt, gt, gt_op, gt_val, gt_ori


def savePreprocessedData(path, X_train, opX_train, X_val, X_test, y_train, opy_train, y_val, y_test, windowSize):
    data_path = os.path.join(os.getcwd(), path)

    with open(os.path.join(data_path, "x0trainwindowsize") + str(windowSize) + str(data_name) + ".npy",
              'bw') as outfile:
        np.save(outfile, X_train)
    with open(os.path.join(data_path, "x0valwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
        np.save(outfile, X_val)
    with open(os.path.join(data_path, "x0testwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
        np.save(outfile, X_test)
    with open(os.path.join(data_path, "opx0trainwindowsize") + str(windowSize) + str(data_name) + ".npy",
              'bw') as outfile:
        np.save(outfile, opX_train)
    with open(os.path.join(data_path, "y0trainwindowsize") + str(windowSize) + str(data_name) + ".npy",
              'bw') as outfile:
        np.save(outfile, y_train)
    with open(os.path.join(data_path, "opy0trainwindowsize") + str(windowSize) + str(data_name) + ".npy",
              'bw') as outfile:
        np.save(outfile, opy_train)
    with open(os.path.join(data_path, "y0valwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
        np.save(outfile, y_val)
    with open(os.path.join(data_path, "y0testwindowsize") + str(windowSize) + str(data_name) + ".npy", 'bw') as outfile:
        np.save(outfile, y_test)
    #


def AugmentData(X_train, y_train, extracted_patches, extracted_patches_gt):
    for k in range(0, 1):
        num = random.randint(0, 1)
        #        print(num)
        if (num == 0):
            flipped_patch = np.flipud(X_train)
            flipped_patch_gt = np.flipud(y_train)
        if (num == 1):
            flipped_patch = np.fliplr(X_train)
            flipped_patch_gt = np.fliplr(y_train)
        if (num == 2):
            no = random.randrange(-180, 180, 30)
            flipped_patch = scipy.ndimage.interpolation.rotate(X_train,
                                                               no, axes=(1, 0), reshape=False, output=None,
                                                               order=3, mode='constant', cval=0.0, prefilter=False)
            flipped_patch_gt = scipy.ndimage.interpolation.rotate(y_train,
                                                                  no, axes=(1, 0), reshape=False, output=None,
                                                                  order=3, mode='constant', cval=0.0, prefilter=False)
        extracted_patch = flipped_patch
        extracted_patch_gt = flipped_patch_gt
        extracted_patches.append(extracted_patch)
        extracted_patches_gt.append(extracted_patch_gt)
    return extracted_patches, extracted_patches_gt


def delete0(toc_gt):
    new_toc_gt = np.zeros((toc_gt.shape[0], toc_gt.shape[1], toc_gt.shape[2], allclass_num - 1))
    for i in range(0, toc_gt.shape[0]):
        for j in range(0, toc_gt.shape[1]):
            for p in range(0, toc_gt.shape[2]):
                new_toc_gt[i, j, p] = np.delete(toc_gt[i, j, p], [0])
    return new_toc_gt


for i in range(100):
    print(i)
    print("Train:", sum(num_Train))
    print("Val:", sum(num_Train_val))
    a = np.count_nonzero(num_Train) + np.count_nonzero(num_Train_val) + np.count_nonzero(num_Train_test)
    if sum(num_Train_val) > max_num or sum(num_Train_val) < min_num or sum(num_Train) > max_num or sum(
            num_Train) < min_num:  # or a != class_num *3:
        num_Train = [0 for i in range(allclass_num)]
        num_Train_val = [0 for i in range(allclass_num)]
        num_Train_op = [0 for i in range(allclass_num)]
        num_Train_test = [0 for i in range(allclass_num)]
        gt_edge = [[] for i in range(allclass_num)]
        Center_pixel = [[] for i in range(allclass_num)]
        Center_pixel_val = [[] for i in range(allclass_num)]
        train_patches, train_patches_gt, op_train_patches, op_train_patches_gt, val_patches, val_patches_gt, test_patches, test_patches_gt, gt, gt_op, gt_val, gt_ori = extract_grids(
            datapath, ground_truth_path, window_size)
    # val_patches=copy.deepcopy(test_patches)
    # val_patches_gt=copy.deepcopy(test_patches_gt)
    # train_patches,train_patches_gt=full(train_patches,train_patches_gt,allusepixels)
    # val_patches,val_patches_gt=full(val_patches,val_patches_gt,allusepixels_val)
    # test_patches, test_patches_gt=full(test_patches,test_patches_gt)

    # train_patches,train_patches_gt=cut_extand(train_patches,train_patches_gt,channels)
    # val_patches,val_patches_gt=cut_extand(val_patches,val_patches_gt,channels)
    # test_patches,test_patches_gt=cut_extand(test_patches,test_patches_gt,channels)

    # data_path = os.path.join(os.getcwd(), 'data')
    # input_data = sio.loadmat(os.path.join(data_path,ordata))[orindata]
    # gt_ori = sio.loadmat(os.path.join(data_path,orlabel))[orinlabel]
    #
    # cut_extand1(input_data,gt_ori,channels)
    else:
        ground_truth = imshow(classes=gt.astype(int), figsize=(6, 6))
        #        for x in range(len(train_patches_gt)):
        #         ground_truth = imshow(classes=train_patches_gt[x].astype(int), figsize=(2,2))
        #        ground_truth = imshow(classes=gt_op.astype(int), figsize=(6,6))
        #        ground_truth = imshow(classes=gt_val.astype(int), figsize=(6,6))
        #        ground_truth = imshow(classes=gt_ori.astype(int), figsize=(6,6))
        #        for i in range(gt.shape[0]):
        #            for j in range(gt.shape[1]):
        #                if gt[i,j] != 0:
        #                    gt_op[i,j] = gt[i,j]
        #                if gt_op[i,j] != 0 :
        #                    gt_ori[i,j] = gt_op[i,j]
        #                if gt_val[i,j] != 0:
        #                    gt_ori[i,j] = gt_val[i,j]
        # #        ground_truth = imshow(classes=gt_op.astype(int), figsize=(6,6))
        #        ground_truth = imshow(classes=gt_ori.astype(int), figsize=(6,6))

        train_patches_gt = to_categorical(train_patches_gt, allclass_num)
        op_train_patches_gt = to_categorical(op_train_patches_gt, allclass_num)
        val_patches_gt = to_categorical(val_patches_gt, allclass_num)
        test_patches_gt = to_categorical(test_patches_gt, allclass_num)

        train_patches_gt = delete0(train_patches_gt)
        val_patches_gt = delete0(val_patches_gt)
        test_patches_gt = delete0(test_patches_gt)
        op_train_patches_gt = delete0(op_train_patches_gt)
        print("train-----" + str(np.array(train_patches).shape))
        print("traingt-----" + str(np.array(train_patches_gt).shape))
        print("val-----" + str(np.array(val_patches).shape))
        print("valgt-----" + str(np.array(val_patches_gt).shape))
        print("test-----" + str(np.array(test_patches).shape))
        print("testgt-----" + str(np.array(test_patches_gt).shape))
        print("testgt-----" + str(np.array(test_patches_gt).shape))
        #        cv2. imwrite("1.png",ground_truth )
        savePreprocessedData(savedata, train_patches, op_train_patches, val_patches, test_patches, train_patches_gt,
                             op_train_patches_gt, val_patches_gt, test_patches_gt, batch_size)
        print(pixels)
        ratio = [0 for i in range(class_num)]
        for i in range(1, class_num):
            ratio[i] = num_Train[i] / pixels[i]
        print("Train :", num_Train, "#Total :", sum(num_Train), "#Total_Ratio:", sum(num_Train) / sum(pixels))
        print("Train_Ratio:", ratio)
        print("Train_op :", num_Train_op, "#Total:", sum(num_Train_op))
        print("Val :", num_Train_val, "#Total:", sum(num_Train_val))
        print("Test:", num_Train_test, "#Total:", sum(num_Train_test))
        csvlist = []
        num_Train[0] = "num_Train"
        csvlist.append(num_Train)
        num_Train_op[0] = "num_op"
        csvlist.append(num_Train_op)
        num_Train_val[0] = "num_val"
        csvlist.append(num_Train_val)
        csvlist.append(" ")
        import csv

        #
        with open('./' + savedata + "numpixels_" + data_name + '.csv', 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in csvlist:
                writer.writerow(row)

        break

