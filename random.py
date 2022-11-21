# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 01:00:10 2022

@author: Ivan
"""
import os
import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio
import skimage.io
import random

path = "images/histology"

ReadImages=os.listdir(path)
#get the random image from histology folder
def randomImg():
    images=[]
    for img in ReadImages:
        image=skimage.io.imread(f'{path}/{img}')
        image=skimage.img_as_float(image)
        images.append(image)
        
    n = random.randint(-1,len(images)-1)
    img = images[n]
    #shape = img.shape
    return img

image=randomImg()

#defining noise functions 
noises=['gaussian', 's&p','salt','pepper','speckle']
noise = noises[random.randint(-1,len(noises)-1)]
#adding noise function
def addNoise(img, noise_mode=noise):
    sigma=0.6
    if (noise_mode == 'gaussian' or noise_mode == 'speckle'):
        noise_img=random_noise(img, mode=noise_mode, seed=None, clip=True, var= sigma**2)
    else:
        noise_img = random_noise(img, mode=noise_mode, amount=0.3)
    return noise_img

noiseImg=addNoise(image)


#defining wavelet functions
wavelets=['haar', 'db2', 'sym2', 'coif2']
wavelet = wavelets[random.randint(-1,len(wavelets)-1)]

#Estimate value of standard deviation value
sigma_est=estimate_sigma(noiseImg,multichannel=True, average_sigmas=True)
#applying Bayes shrinking
denoiseBayes= denoise_wavelet(noiseImg, method='BayesShrink', mode='soft', wavelet_levels=3, wavelet=wavelet,
                              multichannel=True, convert2ycbcr=True, rescale_sigma= True)
#applying Visus shrinking
denoiseVisus= denoise_wavelet(noiseImg, method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=5, wavelet=wavelet,
                              multichannel=True, convert2ycbcr=True, rescale_sigma= True)

#ploting images
fig, axes = plt.subplots(2, 2, figsize=[15, 15])

axes[0,0].imshow(image)
axes[0,0].set_title('Original Image')
axes[0,0].set_axis_off()

axes[0,1].imshow(noiseImg)
axes[0,1].set_title(f'{noise} noise Image')
axes[0,1].set_axis_off()

axes[1,0].imshow(denoiseBayes)
axes[1,0].set_title(f'Bayes denoise/{wavelet}')
axes[1,0].set_axis_off()

axes[1,1].imshow(denoiseVisus)
axes[1,1].set_title(f'Visus denoise/ {wavelet}')
axes[1,1].set_axis_off()

#calculate PSNR
noisy_p=peak_signal_noise_ratio(image, noiseImg)
noisy_bayes=peak_signal_noise_ratio(image, denoiseBayes)
noisy_visu=peak_signal_noise_ratio(image, denoiseVisus)
     
print('Original image vs. Noisy image',noisy_p)     
print('Original image vs. Denoised Bayes',noisy_bayes)  
print('Original image vs. Denoised Visus',noisy_visu)  
      
      

