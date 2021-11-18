# -*- coding: utf-8 -*-
"""
Created on Thu May  6 13:18:13 2021

@author: rgerr
"""

import torch
import os
from PIL import Image
from torchvision import transforms



trainpath = 'E:/CheXpert-v1.0/train/'


def findimages(path):
    list_of_files = {}
    for (dirpath, dirnames, filenames) in os.walk(path):
        for filename in filenames:
            if filename.endswith('.jpg'): 
                list_of_files[filename] = os.sep.join([dirpath, filename])

def addimage(filename):
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
       input_batch = input_batch.to('cuda')
       model.to('cuda')

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        print(output[0])
        # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        print(probabilities)