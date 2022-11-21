# -*- coding: utf-8 -*-
"""
Created on Sun Nov 20 17:08:27 2022

@author: Ivan
"""

import matplotlib.pyplot as plt
from skimage.util import random_noise
from skimage.restoration import (denoise_wavelet, estimate_sigma)
from skimage.metrics import peak_signal_noise_ratio
import skimage.io
import pandas as pd

path = "images/histology/h_03.png"

image=skimage.io.imread(path)
image=skimage.img_as_float(image)


sigma=0.25
noise_img=random_noise(image, var= sigma**2)
sigma_est=estimate_sigma(noise_img,multichannel=True, average_sigmas=True)

fig, axes = plt.subplots(1,2 , figsize=[15, 15])

axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].set_axis_off()

axes[1].imshow(noise_img)
axes[1].set_title( 'noise Image')
axes[1].set_axis_off()

df=pd.DataFrame()

wavelets=['haar', 'db2', 'sym2', 'coif2']

for i in range(len(wavelets)):
    for j in range(4):
        denoiseBayes= denoise_wavelet(noise_img, method='BayesShrink', mode='soft', wavelet_levels=j+1, wavelet=wavelets[i],
                                      multichannel=True, convert2ycbcr=True, rescale_sigma= True)
        noisy_bayes=peak_signal_noise_ratio(image, denoiseBayes)
        new_row = {'Denoise method':'Bayes Shrink', 'mode':'soft', 'wavelet':wavelets[i], 'Wavelet levels':j+1, 'PSN':noisy_bayes}
        df = df.append(new_row, ignore_index=True)
        
        denoiseBayes= denoise_wavelet(noise_img, method='BayesShrink', mode='hard', wavelet_levels=j+1, wavelet=wavelets[i],
                                      multichannel=True, convert2ycbcr=True, rescale_sigma= True)
        noisy_bayes=peak_signal_noise_ratio(image, denoiseBayes)
        new_row = {'Denoise method':'Bayes Shrink', 'mode':'hard', 'wavelet':wavelets[i], 'Wavelet levels':j+1, 'PSN':noisy_bayes}
        df = df.append(new_row, ignore_index=True)
        
        
        denoiseVisus= denoise_wavelet(noise_img, method='VisuShrink', mode='soft', sigma=sigma_est/3, wavelet_levels=j+1, wavelet=wavelets[i],
                                      multichannel=True, convert2ycbcr=True, rescale_sigma= True)
        noisy_bayes=peak_signal_noise_ratio(image, denoiseVisus)
        new_row = {'Denoise method':'Visus Shrink', 'mode':'soft', 'wavelet':wavelets[i], 'Wavelet levels':j+1, 'PSN':noisy_bayes}
        df = df.append(new_row, ignore_index=True)
        
        denoiseVisus= denoise_wavelet(noise_img, method='VisuShrink', mode='hard', sigma=sigma_est/3, wavelet_levels=j+1, wavelet=wavelets[i],
                                      multichannel=True, convert2ycbcr=True, rescale_sigma= True)
        noisy_bayes=peak_signal_noise_ratio(image, denoiseVisus)
        new_row = {'Denoise method':'Visus Shrink', 'mode':'hard', 'wavelet':wavelets[i], 'Wavelet levels':j+1, 'PSN':noisy_bayes}
        df = df.append(new_row, ignore_index=True)
        
   
        
      
df=df.sort_values(['PSN'], ascending =False)
print(df.head(10))








      

