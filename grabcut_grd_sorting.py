# -*- coding: utf-8 -*-
"""
Created on Sat Sep  1 23:48:59 2018

@author: Fakrul-IslamTUSHAR
"""

import shutil
import os
import pandas

###This is the Directory Where you have the images you want to get the name and compare with the image Directory
Source_dir1=os.path.join('D:/Preprocessint_Atlectesis_data_sep4/Atlectethesis/Duke_Complete_hu_img')
Source_dir1_list = os.listdir(Source_dir1)

####This the directory from where you want to move the same_name Images to Destination Directory
img_dir=os.path.join('D:/Sept4_Atelectasis Slices')
img_list_dir = os.listdir(img_dir)

###This is the Destination directory
destination= os.path.join('D:/destination')


#for sub_dir in Source_dir1_list: #This will run a for-loop to the length of the Sorce_Directory_1
#    if str(sub_dir in img_list_dir:  #tHis comoares the name of the images in both directory
#        dir_to_move = os.path.join(img_dir, sub_dir) #which one to move
#        shutil.copy(dir_to_move, destination)

#For-loop for move the files
for sub_dir in range(0, len(Source_dir1_list)): #This will run a for-loop to the length of the Sorce_Directory_1
    for sub_dir2 in range(0,len(img_list_dir)):
        seg= str(Source_dir1_list[sub_dir])
        grd=str(img_list_dir[sub_dir2])
        if (seg[0:12]==grd[0:12]):
            print(seg)
            print(grd)
        #tHis comoares the name of the images in both directory
            dir_to_move = os.path.join(img_dir,grd) #which one to move
            shutil.move(dir_to_move, destination) #where to move= move, copy=where to copy