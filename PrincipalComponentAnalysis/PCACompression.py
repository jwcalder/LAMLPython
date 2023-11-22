# %%
"""
#PCA-based image compression

We consider here an application of PCA to image compression. We first load an image and show some of its trucated SVD's (for row-wise compression of the image matrix).
"""

# %%
#pip install -q graphlearning

# %%
"""
Let's load an image and display it.
"""

# %%
import graphlearning as gl
import matplotlib.pyplot as plt
plt.ion()

img = gl.datasets.load_image('chairtoy')
plt.figure()
plt.imshow(img)

#Check data range and shape
print('Pixel intensity range: (%d,%d)'%(img.min(),img.max()))
print(img.shape)

# %%
"""
Now let's display the truncated SVD. 
"""

# %%
from scipy import sparse
import numpy as np


X = np.hstack((img[:,:,0],img[:,:,1],img[:,:,2]))

vals,Q = sparse.linalg.eigsh(X.T@X,k=500)
Q = Q[:,::-1]

for k in [1,5,25,50,100,200]:
    Xk = np.clip(X@Q[:,:k]@Q[:,:k].T,0,1)
    imgk = np.stack((Xk[:,:512],Xk[:,512:1024],Xk[:,1024:]),axis=2)
    plt.figure()
    plt.imshow(imgk)
    plt.title('%d Singular Vectors'%k)


# %%
"""
For block-based compression, we'll convet the image to 8x8 patches. The image is 512x512 so this gives 4096 patches, each with 8x8x3=192 numbers (RGB for each pixel).
"""

# %%
m = 8
X = gl.utils.image_to_patches(img,patch_size=(m,m))
print(X.shape)
grid = gl.utils.color_image_grid(X,n_rows=5,n_cols=10)

# %%
"""
Now let's do an SVD (or PCA) on the blocks. We show to top 50 singular vectors, which start out as slowly varying low frequencies, with the later singular vectors capturing fine scale details and texture in the image blocks. 
"""

# %%
from scipy import sparse

cov_matrix = X.T@X
Vals, V = sparse.linalg.eigsh(cov_matrix,k=3*m*m-1,which='LM')
Vals = Vals[::-1]
V_all = V[:,::-1]

P = V_all.T.copy()
P = P - P.min()
P = P/P.max()
gl.utils.color_image_grid(P,n_rows=5,n_cols=10)

# %%
"""
To compress the image, we project the image blocks onto the top k singular vectors and the reconstruc the image from its blocks.
"""

# %%
for num_comps in [1,10,20,40,80]:

    #Get top singular vectors
    V = V_all[:,:num_comps]

    #Compress and decompress the image
    comp_X = X@V
    decomp_X = comp_X@V.T

    #Compute compression ration
    comp_size = V.shape[0]*V.shape[1] + comp_X.shape[0]*comp_X.shape[1]
    comp_ratio = X.shape[0]*X.shape[1]/comp_size

    #Recontruct image from patches for visualization
    img_comp = gl.utils.patches_to_image(decomp_X, (img.shape[0],img.shape[1]), patch_size=(m,m))
    img_comp = np.clip(img_comp,0,1)

    #Print PSNR
    MSE = np.sum((X-decomp_X)**2)/X.shape[0]/X.shape[1]
    PSNR = 10*np.log10(np.max(X)**2/MSE)
    print('%f,%f'%(comp_ratio,PSNR))

    #Plot compressed and difference image
    img_diff = np.clip(img_comp - img + 0.5,0,1)
    plt.figure()
    plt.imshow(np.hstack((img_comp,img_diff)))
    plt.title('Compression Raio: %.1f:1'%comp_ratio)


# %%
"""
## Exercises
1. Play around with different block sizes. What is the best for compression?
2. Compare the PSNR for the original row-based compression with block-based compression. How does the PSNR compare at the same label rates. 
3. Generate a plot of the singular values for the image and the blocks. Which one decays faster? 
4. Project the blocks into two dimensions to visualize the block space. 
5. [Challenging] Write a compression algorithm that chooses the best singular vectors to use for each block, instead of the top $k$. To do this, choose a threshold $\mu>0$, project the image blocks onto all of the singular vectors, and then discard (i.e., set to zero) any coefficient (i.e., PCA coordinate) that is smaller than $\mu$. Reconstruct the image from the truncated blocks, and compute the compression ratio assuming you do not have to store the zeros (the coefficients that were threshold ed to zero), and assume you don't need to store the singular vectors (the setting is that you learn good singular vectors, and then share them between the encoder and decoder, so only the coefficients must be transmitted/stored). How does this compare with the block-based method in this notebook? 
#"""


