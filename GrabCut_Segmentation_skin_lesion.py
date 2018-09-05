
"""
Created on Thu May 24 01:15:31 2018

@author: Fakrul-IslamTUSHAR
"""
# =============================================================================
# Instruction
# =============================================================================
"""
*Put the code to the folder of images
*Give the destination folder path where you want to save the segemntaed images.
"""

# =============================================================================
# Import Libraries
# =============================================================================

import cv2
import numpy as np
from glob import glob
import os
from matplotlib import pyplot as plt

# =============================================================================
# Get the image
# =============================================================================

#Getting all the images in the folder
for im in glob('*.jpg'):
     img = cv2.imread(im,-1)
     filename_w_ext = os.path.basename(im)
     filename, file_extension = os.path.splitext(filename_w_ext)
     cv2.namedWindow(filename, cv2.WINDOW_NORMAL)
     #cv2.imshow(filename,img)
     print filename
     #cv2.waitKey(0)
     median = cv2.medianBlur(img,5) # Apply Median filter
# =============================================================================
#      img = cv2.imread(org_imgName,-1)

     Z = median.reshape((-1,3))

# convert to np.float32
     Z = np.float32(Z)

# define criteria, number of clusters(K) and apply kmeans()
     criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
     K = 8
     ret,label,center=cv2.kmeans(Z,K,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

# Now convert back into uint8, and make original image
     center = np.uint8(center)
     res = center[label.flatten()]
     kmeans_img = res.reshape((img.shape))

     cv2.namedWindow("Kmean_img", cv2.WINDOW_NORMAL)
     cv2.imshow('Kmean_img',kmeans_img)
     #cv2.waitKey(0)
     output_path = 'C:\SKINLESIONSEGMENTATION\kmeans' #Give path where you want to save the kmeans images
     cv2.imwrite(os.path.join(output_path, filename+'_kmeans' +'.png'),kmeans_img)
     cv2.destroyAllWindows()
# =============================================================================
# =============================================================================
#    Adaptive histogram equalization  
# =============================================================================
     clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8,8))

     hsv = cv2.cvtColor(kmeans_img, cv2.COLOR_BGR2HSV)# convert from BGR to HSV color space
     output_path = 'C:\SKINLESIONSEGMENTATION\hsv_img' # Give the path where you want to save HSV images
     cv2.imwrite(os.path.join(output_path, filename+'.png'),hsv)
     cv2.destroyAllWindows()
     
     h, s, v = cv2.split(hsv)  # split on 3 different channels
     #apply CLAHE to the L-channel
     h1 = clahe.apply(h)
     s1 = clahe.apply(s)
     v1 = clahe.apply(v)

     lab = cv2.merge((h1,s1,v1))  # merge channels
     
     output_path = 'C:\SKINLESIONSEGMENTATION\enhanced_hsv_img' # Destination folder for save Enhances HSV images
     cv2.imwrite(os.path.join(output_path, filename+'_AHE' +'.png'),lab)
     cv2.destroyAllWindows()
     
     Enhance_img= cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR

     output_path = 'C:\SKINLESIONSEGMENTATION\enhanced_bgr_image' # Destination folder for save Enhances BGR images
     cv2.imwrite(os.path.join(output_path, filename+'_AHE' +'.png'),Enhance_img)
     cv2.destroyAllWindows()
     
# =============================================================================
#    making the mask for grabcut
# =============================================================================
     hsv = cv2.cvtColor(Enhance_img, cv2.COLOR_BGR2HSV)    
     lower_green = np.array([50,100,100])
     upper_green = np.array([100,255,255])
     mask_g = cv2.inRange(hsv, lower_green, upper_green)
     output_path = 'C:\SKINLESIONSEGMENTATION\Green_mask'
     cv2.imwrite(os.path.join(output_path, filename+'.png'),mask_g)
     
     ret,inv_mask = cv2.threshold(mask_g,127,255,cv2.THRESH_BINARY_INV)
     output_path = 'C:\SKINLESIONSEGMENTATION\Inverse_Green_mask'
     cv2.imwrite(os.path.join(output_path, filename+'.png'),inv_mask)
     
     res = cv2.bitwise_and(img,img, mask= mask_g)
     output_path = 'C:\SKINLESIONSEGMENTATION\rio'
     cv2.imwrite(os.path.join(output_path, filename+'.png'),res)
     
     mask = np.zeros(img.shape[:2],np.uint8)
     bgdModel = np.zeros((1,65),np.float64)
     fgdModel = np.zeros((1,65),np.float64)
     
     if (np.sum(inv_mask[:])<80039400):
        newmask = inv_mask
        

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
        mask[newmask == 0] = 0
        mask[newmask == 255] = 1
        dim= cv2.grabCut(img,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        GrabCut_img = img*mask2[:,:,np.newaxis]
        output_path = 'C:\SKINLESIONSEGMENTATION\grabcut_hsv'
        cv2.imwrite(os.path.join(output_path, filename+'_GrabCut' +'.png'),GrabCut_img)
        cv2.destroyAllWindows()
     else:
        
     
# =============================================================================
#      GrabCut
# =============================================================================
        
# =============================================================================
# =============================================================================
        #initializing the Ractangle based on the image dimention
        s = (img.shape[0] / 10, img.shape[1] / 10)
        rect = (s[0], s[1], img.shape[0] - (3/10) * s[0], img.shape[1] - s[1])
        cv2.grabCut(Enhance_img,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        GrabCut_img= img*mask2[:,:,np.newaxis]
     
        plt.imshow(GrabCut_img)
        plt.colorbar()
        plt.show()
     
        output_path = 'C:\SKINLESIONSEGMENTATION\grabcut_hsv'
        cv2.imwrite(os.path.join(output_path, filename+'_GrabCut' +'.png'),GrabCut_img)
        cv2.destroyAllWindows()
     

# =============================================================================
# Binarization
# =============================================================================
     imgmask = cv2.medianBlur(GrabCut_img,5)
     ret,Segmented_mask = cv2.threshold(imgmask,0,255,cv2.THRESH_BINARY)
     output_path = 'C:\SKINLESIONSEGMENTATION\grabcut_hsv_bw'
     cv2.imwrite(os.path.join(output_path, filename+'.png'),Segmented_mask)
     cv2.destroyAllWindows()
       
# =============================================================================
#      2nd GRABCUT
# =============================================================================
     if (np.sum(inv_mask[:])<80039400):
        newmask = inv_mask

# wherever it is marked white (sure foreground), change mask=1
# wherever it is marked black (sure background), change mask=0
        mask[newmask == 0] = 0
        mask[newmask == 255] = 1
        dim2= cv2.grabCut(lab,mask,None,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_MASK)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        GrabCut_img2 = img*mask2[:,:,np.newaxis]
        output_path = 'C:\SKINLESIONSEGMENTATION\grabcut_bgr'
        cv2.imwrite(os.path.join(output_path, filename+'_GrabCut' +'.png'),GrabCut_img2)
        cv2.destroyAllWindows()
     else:
     
        cv2.grabCut(lab,mask,rect,bgdModel,fgdModel,10,cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
        GrabCut_img2= img*mask2[:,:,np.newaxis]
        output_path = 'C:\SKINLESIONSEGMENTATION\grabcut_bgr'
        cv2.imwrite(os.path.join(output_path, filename+'_GrabCut' +'.png'),GrabCut_img2)
        cv2.destroyAllWindows()
     

# =============================================================================
# Binarization
# =============================================================================
     imgmask2 = cv2.medianBlur(GrabCut_img2,5)
     ret,Segmented_mask2 = cv2.threshold(imgmask2,0,255,cv2.THRESH_BINARY)
     output_path = 'C:\SKINLESIONSEGMENTATION\grabcut_bgr_bw'
     cv2.imwrite(os.path.join(output_path, filename+'.png'),Segmented_mask2)
     cv2.destroyAllWindows()
     
     plt.imshow(GrabCut_img2)
     plt.colorbar()
     plt.show()
     
     
# =============================================================================
