
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 29 08:40:49 2018

@author: user
"""

import numpy as np
from matplotlib import pyplot as plt
from skimage.filters import threshold_mean
import network
import os
import torch
from PIL import Image
from skimage.transform import resize
import jiwer
import pandas as pd
import tensorflow as tf

    
# Utils
def reshape(data):
    dim = int(np.sqrt(len(data)))
    data = np.reshape(data, (dim, dim))
    return data

def plot(data, test, predicted, figsize=(3, 3)):
    data = [reshape(d) for d in data[1:4]]
    test = [reshape(d) for d in test[1:4]]
    predicted = [reshape(d) for d in predicted[1:4]]
    
    fig, axarr = plt.subplots(len(data), 3, figsize=figsize)
    for i in range(len(data)):
        if i==0:
            axarr[i, 0].set_title('Train data')
            axarr[i, 1].set_title("Input data")
            axarr[i, 2].set_title('Output data')
            
        axarr[i, 0].imshow(data[i])
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(test[i])
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(predicted[i])
        axarr[i, 2].axis('off')
            
    plt.tight_layout()
    plt.savefig("result_arabic.png")
    plt.show()

def preprocessing(img, w=1000, h=400):
    w, h = img.size
    img = img.resize((w,h))
    # Thresholding
    thresh = threshold_mean(img)
    binary = img > thresh
    shift = 2*(binary*1)-1 # Boolian to int
    
    # Reshape
    flatten = np.reshape(shift, (w*h))
    return flatten

def main():
    # Load data
    train_dir = 'data/train_data'
    test_dir =  'data/test_data'

    data = []
    for images in os.listdir(train_dir):
        data.append(Image.open(train_dir +'/'+ images))
        
    test = []
    for images in os.listdir(test_dir):
         test.append(Image.open(test_dir + '/' + images))
    # Preprocessing
    print("Start data preprocessing...")
    data = [preprocessing(d) for d in data]
    
    # Create Hopfield Network Model
    model = network.HopfieldNetwork()
    model.train_weights(data)
    
    # test dataset
    test = [preprocessing(d) for d in test]
    
    predicted = model.predict(test, threshold=50, asyn=True)
    print("Sample of prediction results...")
    plot(data, test, predicted, figsize=(5, 5))
    print("Network weights matrix...")
    model.plot_weights()
    
    ground_truth_dir = 'data/Groundtruth-Unicode.xlsx'
    ground_truth = pd.read_excel(ground_truth_dir)
    ground_truth = str(ground_truth[:])

    wer = jiwer.wer(ground_truth, str(predicted))
    print("word error rate (WER): %f", wer)

    
if __name__ == '__main__':
    """Device Selection"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    main().to(device)
