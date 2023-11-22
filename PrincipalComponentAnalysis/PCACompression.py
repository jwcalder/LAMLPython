# %%
"""
#PCA-based image compression

We consider here an application of PCA to image compression. We first load an image and show some of its trucated SVD's (for row-wise compression of the image matrix).
"""

# %%
#pip install -q graphlearning

# %%
import graphlearning as gl
import matplotlib.pyplot as plt
from scipy import sparse
import numpy as np
plt.ion()

img = gl.datasets.load_image('chairtoy')

X = np.hstack((img[:,:,0],img[:,:,1],img[:,:,2]))

vals,Q = sparse.linalg.eigsh(X.T@X,k=500)
Q = Q[:,::-1]

for k in [1,5,25,50,100,200]:
    Xk = np.clip(X@Q[:,:k]@Q[:,:k].T,0,1)
    imgk = np.stack((Xk[:,:512],Xk[:,512:1024],Xk[:,1024:]),axis=2)
    plt.figure()
    plt.imshow(imgk)

# %%
"""
Let's load the cameraman image.
"""

## %%
#import matplotlib.pyplot as plt
#import graphlearning as gl
#
##Load and display image
#I = gl.datasets.load_image('cameraman')
#plt.figure(figsize=(10,10))
#plt.imshow(I,cmap='gray')
#
##Check data type of image
#print('Data type: '+str(I.dtype))
#print('Pixel intensity range: (%d,%d)'%(I.min(),I.max()))
#
##Print image shape
#print(I.shape)
#
## %%
#"""
#Let's now convert the image into 8x8 patches. The image is 512x512 so this gives 4096 patches, each with 8x8=64 pixels.
#"""
#
## %%
#X = image_to_patches(I,patch_size=(8,8))
#
#print(X.shape)
#num_patches = (512/8)**2
#print(num_patches)
#
## %%
#"""
#To compress the image, we run PCA on the patches, and project the image to the best linear subspace obtained by PCA.
#"""
#
## %%
#from scipy.sparse import linalg
#import numpy as np
#
##Number of principal components to use
#num_comps = 5
#
##Compute the principal components
#Vals, P = linalg.eigsh(X.T@X,k=num_comps,which='LM')
#
##Compress the image by projecting to the linear subspace spanned by P
#X_compressed = X@P
#print(X_compressed.shape)
#
##Compute size of compressed image and compression ratio
#compressed_size = X_compressed.size + P.size
#comp_ratio = I.size/compressed_size
#print('Compression ratio: %.1f:1'%comp_ratio)
#
## %%
#"""
#Let's now decompress the image by changing coordinates back to the standard ones. We'll also show the reconstructed image and the error between the original and reconstruction.
#
#The reconstruction quality in image compression is measured by the peak signal to noise ratio (PSNR) in dB. PSNR values between 30dB and 50dB are acceptable in image compression.
#"""
#
## %%
#import matplotlib.pyplot as plt
#import numpy as np
#
##Decompress image
#X_decompressed = X_compressed@P.T
#print(X_decompressed.shape)
#I_decompressed = patches_to_image(X_decompressed,I.shape,patch_size=(8,8))
#print(I_decompressed.shape)
#
##Decompress and clip image to [0,1]
#I_decompressed = np.clip(I_decompressed,0,1)
#
##Plot decompressed (reconstructed image) and difference image
#plt.figure(figsize=(30,10))
#plt.imshow(np.hstack((I,I_decompressed, I-I_decompressed+0.5)), cmap='gray', vmin=0, vmax=1)
#
##Compute Peak Signal to Noise Ratio (PSNR)
#MSE = np.sum((I-I_decompressed)**2)/I.size
#PSNR = 10*np.log10(np.max(I)**2/MSE)
#print('PSNR: %.2f dB'%PSNR)
#plt.show()
#
## %%
#"""
###Exercises
#1. Write Python code to perform row-wise compression of the cameraman image. Essentially you can just set X=I and avoid the image_to_patches and patches_to_image functions. Compare the PSNR at similar compression ratios to the patch/block-wise compression used in this notebook.
#"""


