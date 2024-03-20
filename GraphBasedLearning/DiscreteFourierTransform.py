# %%
"""
# The Discrete Fourier Transform

In this notebook we explore the Fast Fourier Transform (FFT) and and applications to signal denoising.
"""

# %%
"""
## The Fast Fourier Transform (FFT)

We start with the FFT. Below, we define our basic DFT and inverse dft, which take $O(n^2)$ operations for a signal of length $n$.

"""

# %%
import numpy as np

def dft(f):
    """Computes the Discrete Fourier Transform (DFT) of f

    Args:
        f: Function to compute DFT of (real or complex vector)

    Returns:
        Df: DFT of f
    """
    n = len(f)
    k = np.arange(n)
    Df = np.zeros(n,dtype='complex')
    for l in range(n):
        Df[l] =  np.sum(f*np.exp(-2*np.pi*1j*k*l/n))
    return Df

def idft(Df):
    """Computes the inverse Discrete Fourier Transform (DFT) of Df

    Args:
        Df: Function to compute iDFT of (real or complex vector)

    Returns:
        f: Inverse DFT of Df
    """
    n = len(Df)
    k = np.arange(n)
    f = np.zeros(n,dtype='complex')
    for l in range(n):
        f[l] =  np.sum(Df*np.exp(2*np.pi*1j*k*l/n))/n
    return f

# %%
"""
Let's now define the FFT and inverse FFT, using recursive programming.
"""

# %%
def my_fft(f):
    """Computes the Discrete Fourier Transform (DFT) of f
    using the Fast Fourier Transform.

    Args:
        f: Function to compute DFT of (real or complex vector)

    Returns:
        Df: DFT of f
    """

    n = len(f)
    k = np.arange(n)
    if n == 1:
        return f
    else:
        #DFT of even and odd parts, recursively
        Dfe = my_fft(f[::2])
        Dfo = my_fft(f[1::2])

        #Periodically extend to length of f
        Dfe = np.hstack((Dfe,Dfe))
        Dfo = np.hstack((Dfo,Dfo))

        #Combine Dfe and Dfo to get Df
        return Dfe + np.exp(-2*np.pi*1j*k/n)*Dfo

def my_ifft(Df):
    """Computes the inverse Discrete Fourier Transform (DFT) of Df
        using the Fast Fourier Transform.

    Args:
        Df: Function to compute iDFT of (real or complex vector)

    Returns:
        f: Inverse DFT of Df
    """

    return np.conjugate(my_fft(np.conjugate(f)))/len(f)

# %%
"""
Let's compare our FFT with the DFT we programmed last time, to make sure they work.
"""

# %%
n = 1024
f = np.random.randn(n)
print(np.max(np.absolute(my_fft(f) - dft(f))))
print(np.max(np.absolute(my_ifft(f) - idft(f))))

# %%
"""
Now let's compare the CPU time for the naive DFT and our FFT implementation.
"""

# %%
import time

n = int(2**15)   #Approximately n=32,000
f = np.random.randn(n)

start_time = time.time()
Df = dft(f)
print("Naive DFT: %s s" % (time.time() - start_time))

start_time = time.time()
Df = my_fft(f)
print("Our FFT: %s s" % (time.time() - start_time))

# %%
"""
SciPy has a very efficient implementation of the FFT. Let's compare our version to Scipy's.
"""

# %%
from scipy.fft import fft
from numpy.fft import fft as numpy_fft

n = int(2**20)   #Approximately n=1 million
f = np.random.randn(n)

start_time = time.time()
Df = fft(f)
print("SciPy FFT: %s s" % (time.time() - start_time))

start_time = time.time()
Df = numpy_fft(f)
print("NumPy FFT: %s s" % (time.time() - start_time))

start_time = time.time()
Df = my_fft(f)
print("Our FFT: %s s" % (time.time() - start_time))

# %%
"""
## Signal Denoising

We now turn to signal denoising. First we define a function to generate random signals, and the Tikhonov denoising function.
"""

# %%
import numpy as np
from scipy.fft import fft
from scipy.fft import ifft

def random_signal(n,m,seed=None):
    """Returns a random signal (random trig polynomial)

    Args:
        n: Number of samples (length of signal)
        m: Degree for trig polynomial
        seed: Seed for random number generator, for reproducible results (defualt=None)

    Returns:
        x,f: Length n numpy arrays containing x-coordiantes and the random signal
    """

    if seed is not None:
        np.random.seed(seed=seed)
    x = np.arange(n)/n
    f = np.zeros(n)
    for k in range(1,m):
        f += (np.random.rand(1)-0.5)*np.sin(np.pi*k*x)/k
        f += (np.random.rand(1)-0.5)*np.cos(np.pi*k*x)/k

    return x,f

def tikhonov_denoise(f,lam):
    """Tikhonov regularized denoising

    Args:
        f: Noisy signal (numpy array)
        lam: Regularization parameter

    Returns:
        Denoised signal
    """

    n = len(f)
    k = np.arange(n)
    G = 1/(1 + lam - lam*np.cos(2*np.pi*k/n))
    return ifft(G*fft(f)).real

# %%
"""
Let's generate and plot a random signal with noise.
"""

# %%
import matplotlib.pyplot as plt
plt.ion()

#Signal of length 256
n = 256
x,f = random_signal(n,30,seed=123)

#Add some Gaussian noise
sigma = 0.5*np.std(f)
f_noisy = f + sigma*np.random.randn(n)

#Plot signal and noisy signal
plt.figure()
plt.plot(x,f_noisy,linewidth=0.5)
plt.plot(x,f,linewidth=2)

# %%
"""
We'll now run Tikhonov denoising for different $\lambda$ and plot the result. Play around with lam (lambda) and see what you get.
"""

# %%
lam = 10
f_denoised = tikhonov_denoise(f_noisy,lam)

plt.figure()
plt.plot(x,f,label='Original')
plt.plot(x,f_noisy,linewidth=0.5,label='Noisy')
plt.plot(x,f_denoised,label='$\lambda=%d$'%lam)
plt.legend(fontsize=20)

# %%
"""
The denoising works quite well. But we see some peculiar boundary effects. This is due to the DFT periodicizing the signal, and the starting and ending values do not match up. As far as the DFT is concerned, the signal looks like the periodic version below. Recall the filtering is just locally averaging the signal, and the average is **periodic** so at the boundary it sees the other end of the signal.
"""

# %%
plt.figure()
fp = np.hstack((f[1:],f[0]))
plt.plot(x,fp)
plt.plot(x+1,fp)
plt.plot(x-1,fp)
plt.title('Periodic extension of the signal')

# %%
"""
##Exercise

Take the even extension of $f$ before applying Tikhonov denoising. After denoising, restrict the output to the first $n$ samples to remove the extra extension data.

The even extension of a signal with $n=7$ samples $[a,b,c,d,e,f,g]$ is the length $2n-2=12$ signal $[a,b,c,d,e,f,g,f,e,d,c,b]$. A general formula for the even extension is

$$f_e(k) =
\begin{cases}
f(k),&\text{if } 0 \leq k \leq n-1,\\
f(2(n-1)-k),&\text{if }n \leq k \leq 2(n-1)-1.
\end{cases}$$

"""
