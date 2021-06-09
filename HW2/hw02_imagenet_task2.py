# -*- coding: utf-8 -*-
"""
Created on Mon Feb 15 00:30:55 2021

@author: 11327
"""

import argparse
# code from handout
parser = argparse.ArgumentParser ( description = 'HW02 Task2')
parser.add_argument('--imagenet_root', nargs ='*', type =str , required = True )
parser.add_argument('--class_list', nargs ='*', type =str , required = True )
args,args_other = parser.parse_known_args()

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
import scipy.misc
import torchvision.transforms as tvt
import os
import numpy as np
import random


class your_dataset_class (Dataset):
    def __init__ (self, root, class_list, transform) : # three inputs: Image position, class_list, tvt.compose()
        '''
        Make use of the arguments from argparse
        initialize your program - defined variables
        e.g. image path lists for cat and dog classes
        you could also maintain label_array
        0 -- cat
        1 -- dog
        Initialize the required transform
        '''
        i = 0
        self.imagepath = []
        self.class_list = class_list
        self.root = root
        self.folderpath = ''
        self.im = list()
        self.label_array = []
        for c_l in self.class_list:
            # construct the path
            if c_l == self.class_list[0]:
                self.folderpath = os.path.join(self.root+'/','cat/*.jpg')
            if c_l == self.class_list[1]:
                self.folderpath = os.path.join(self.root+'/','dog/*.jpg')
            self.folderpath = self.folderpath.replace('\\','/')
            self.imagepath = glob.glob(str(self.folderpath))

            for p in self.imagepath:
                self.im.append(np.array(Image.open(p),dtype=np.float)) # open and initialize the images
                # obtain label_array
                if c_l == self.class_list[0]: # the first class is cat
                    self.label_array.append(0.)
                elif c_l == self.class_list[1]: # the second class is dog
                    self.label_array.append(1.)
                i = i+1
        self.transform = transform # initialize the required transform
        self.number = i # num of images
        self.one_hot = [] # one hot encoding
        self.tensor = []
        
    def __len__ (self) :
        '''
        return the total number of images
        refer pytorch documentation for more details
        
        '''
        return self.number
        
    def __getitem__ (self, idx) :
        '''
        Load color image (s), apply necessary data conversion and transformation
        e.g. if an image is loaded in HxWXC ( Height X Width X Channels ) format
        rearrange it in CxHxW format , normalize values from 0
        -255 to 0-1
        and apply the necessary transformation .
        
        Convert the corresponding label in 1-hot encoding .
        Return the processed images
        and labels in 1-hot encoded format
        '''
        self.tensor = self.transform((self.im[idx]).T) # apply transform
        # 1-hot coding
        self.one_hot = self.label_array[idx]
        # assign the 1 hot encoding
        if self.label_array[idx] == 0:
            self.one_hot = np.array([1.,0.])
        elif self.label_array[idx] == 1:
            self.one_hot = np.array([0.,1.])
   
        return self.tensor, self.one_hot

if __name__ == '__main__':
        
    batchsize = 10
    
    random.seed(0)
    # code from handout
    transform = tvt.Compose([tvt.ToTensor(), tvt.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = your_dataset_class(args.imagenet_root[0], args.class_list, transform)
    
    train_data_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    batch_size = batchsize,
    shuffle =True ,
    num_workers =0)
    
    val_dataset = your_dataset_class (args.imagenet_root[1], args.class_list, transform)
    val_data_loader = torch.utils.data.DataLoader(dataset = val_dataset,
    batch_size = batchsize,
    shuffle =True,
    num_workers =0)
            
    dtype = torch.float64
    
    device = torch.device ("cuda:0" if torch.cuda.is_available() else " cpu")
    
    epochs = 10 # feel free to adjust this parameter
    D_in , H1 , H2 , D_out = 3*64*64 , 1000 , 256 , 2
    w1 = torch.randn ( D_in , H1 , device =device , dtype = dtype ) / 10**10 # Divided by 10^10 to avoid the w becoming infinity
    w2 = torch.randn ( H1 , H2 , device =device , dtype = dtype ) / 10**10
    w3 = torch.randn ( H2 , D_out , device =device , dtype = dtype ) / 10**10
    learning_rate = 1e-9
    
    # test = 0
    outputtxt = open('./output.txt', mode = 'w') # open the output.txt
    
    for t in range(epochs):
        
        loss = 0;
        
        for i , data in enumerate ( train_data_loader ):

            inputs , labels = data

            inputs = inputs.to ( device )

            labels = labels.to ( device )
            x = inputs.view ( inputs.size(0) , -1 )
            h1 = x.mm ( w1 ) ## In numpy , you would say h1 = x.dot(w1)
            h1_relu = h1.clamp (min =0 )
            h2 = h1_relu.mm ( w2 )
            h2_relu = h2.clamp (min =0 )
            y_pred = h2_relu.mm ( w3 )
            # Compute and print loss
            y = labels
            loss = ( y_pred - y ).pow(2).sum().item() + loss
            y_error = y_pred - y
    
            # TODO : Accumulate loss for printing per epoch
            grad_w3 = h2_relu. t().mm ( 2 * y_error ) # <<<<<< Gradient of Loss w.r.t w3
            h2_error = 2.0 * y_error.mm ( w3.t() ) # backpropagated error to the h2 hidden layer
            h2_error [h2 < 0] = 0 # We set those elements of the backpropagated error
            grad_w2 = h1_relu.t().mm ( 2 * h2_error ) # <<<<<< Gradient of Loss w.r.t w2
            h1_error = 2.0 * h2_error.mm ( w2.t() ) # backpropagated error to the h1 hidden layer
            h1_error [h1 < 0] = 0 # We set those elements of the backpropagated error
            grad_w1 = x.t().mm ( 2 * h1_error ) # <<<<<< Gradient of Loss w.r.t w2
            # Update weights using gradient descent
            w1 -= learning_rate * grad_w1
            w2 -= learning_rate * grad_w2
            w3 -= learning_rate * grad_w3
    
        # print loss per epoch
        epoch_loss = loss
        # print(test)
        print ('Epoch  %d:\t %0.4f'%(t , epoch_loss), file=outputtxt)
    # Store layer weights in pickle file format
    torch . save ({'w1':w1 ,'w2':w2 ,'w3':w3},'./ wts.pkl')     
    # Validation Part
    correct = 0
    loss = 0;
    
    for j, data_v in enumerate(val_data_loader):
        inputs , labels = data_v
        inputs = inputs.to ( device )

        labels = labels.to ( device )
        x = inputs.view ( inputs.size(0) , -1 )
        h1 = x.mm ( w1 ) ## In numpy , you would say h1 = x.dot(w1)
        h1_relu = h1.clamp (min =0 )
        h2 = h1_relu.mm ( w2 )
        h2_relu = h2.clamp (min =0 )
        y_pred = h2_relu.mm ( w3 )
        # Compute and print loss
        y = labels
        for i in range(batchsize):
            if (y_pred[i,0]>y_pred[i,1]) & (y[i,0]==1):
                correct = correct + 1
            if (y_pred[i,0]<y_pred[i,1]) & (y[i,1]==1):
                correct = correct + 1
        loss = ( y_pred - y ).pow(2).sum().item()+loss
        y_error = y_pred - y
        
    Val_Acc = correct / val_dataset.number    
        
    print('\n', file=outputtxt)
    print ('Val Loss:\t %0.4f'%(loss), file=outputtxt)
    print ('Val Accuracy: %f %%' %(Val_Acc*100), file=outputtxt)
    outputtxt.close()
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        