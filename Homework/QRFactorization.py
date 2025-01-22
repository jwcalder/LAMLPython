# %%
"""
# QR Factorization

The QR factorization of a square matrix $A$ produces an orthogonal matrix $Q$ and an upper triangular matrix $R$ such that
$$A = QR.$$
The QR factorization is useful for solving the linear equation $A\mathbf{x}=\mathbf{b}$ by solving the equivalent equation $QR\mathbf{x}=\mathbf{b}$, or rather
$$R\mathbf{x} = Q^T \mathbf{b},$$
where we recall that $Q^T$ is the inverse of $Q$, since $Q$ is orthonormal. As $R$ is upper triangular, we can solve $R\mathbf{x}=Q^T\mathbf{b}$ by back substitution. While we focus on square matrices in this notebook, it is important to note that QR factorization can also be applied to non-square matrices, in which case there are interesting connections to least squares solutions to linear systems.

## QR via Gram-Schmidt

The simplest way to compute a QR factorization is with the Gram-Schmidt algorithm. We assume $A$ is a square $n\times n$ non-singular matrix (i.e., its columns are linearly independent). The main steps of the Gram-Schmidt algorithm are presented in pseudocode below.

Let $\mathbf{v}_1,\mathbf{v}_2,\dots,\mathbf{v}_n$ denote the columns of $A$. We initialize $\mathbf{u}_1=\mathbf{v}_1/r_{11}$, where $r_{11}=\|\mathbf{v}_1\|$ and repeat the steps below for $k=2$ through $k=n$.

1. Compute the coefficients of the $k$th column of $R$, that is $r_{jk} = \mathbf{u}_j\cdot \mathbf{v}_k$ for $j \leq k-1$, $r_{kk} = \sqrt{\|\mathbf{v}_k\|^2 - \sum_{j=1}^{k-1} r_{jk}^2}$, and $r_{jk}=0$ for $j>k$.
2. Compute the $k$th column of $Q$, given by $\mathbf{u}_k = \frac{1}{r_{kk}}\left( \mathbf{v}_k - \sum_{j=1}^{k-1} r_{jk}\mathbf{u}_j \right)$.

The vectors $\mathbf{u}_k$ form the columns of $Q$, and by the definition of $\mathbf{u}_k$ we have
$$\mathbf{v}_k = \sum_{j=1}^k r_{jk} \mathbf{u}_j,$$
for all $k$, which is equivalent to the QR factorization statement
$$A = QR.$$



"""

# %%
"""
### Exercise

Write Python code for QR factorization using Gram-Schmidt. You can use the template in the code below; fill in the gaps. Try your algorithm on some toy matrices.
"""

# %%
import numpy as np

def QR_GS(A):
    """QR via Gram-Schmidt
    ======
    Produces a QR factorization via standard Gram-Schmidt.
    The algorithm is numerically unstable and may not return orthonormal Q.

    Parameters
    ----------
    A : numpy array, float
        Non-singular matrix to perform QR on. A should be mxn with m >= n
        and all columns linearly independent.

    Returns
    -------
    Q : numpy array, float
        Orthogonal basis.
    R : numpy array, float
        Upper triangular matrix R so that A=QR
    """

    #Get shapes of matrices and initialize Q,R
    m,n = A.shape
    Q = np.zeros((m,n))
    R = np.zeros((n,n))

    #First step (note v_1 = A[:,0], v_2 = A[:,1], etc.)
    R[0,0] =
    Q[:,0] =

    for k in range(1,m):

        r = 0
        for j in range(k): #j=0,...,k-1
            R[j,k] = #u_j dot v_k (v_k = A[:,k], u_j = Q[:,j])
            r = r + R[j,k]**2
        R[k,k] = #sqrt(norm(v_k)^2 - r)
        Q[:,k] =

    return Q,R

# %%
"""
As an example, we define the matrix
$$A = \begin{pmatrix}
1 & 1\\
0 & 1
\end{pmatrix}.$$
The QR factorization is $A = I A$, where $I$ is the identity matrix (in fact, we always have $Q=I$ for any $A$ that is already upper triangular). We can verify this in Python code below.
"""

# %%
A = np.array([[1,0],[1,1]])
print("A=\n",A.T)

Q,R = QR_GS(A.T)
print("Q=\n",Q)
print("R=\n",R)

# %%
"""
In the code above, replace $A$ with $A^T$. How does the QR factorization change? Verify the QR factorization for $A^T$ by hand.
"""

# %%
"""
Let us check below that the QR factorization works for randomly generated matrices.
"""

# %%
A = np.random.rand(10,10)
Q,R = QR_GS(A)
print(np.linalg.norm(A - Q@R))

# %%
"""
We see above that the norm $\|A-QR\|$ is close to machine precision, around $10^{-15}$ in this case. We can also check the orthogonality of $Q$ with the code below, which computes the norm $\|I - Q^TQ\|$. Clearly $Q$ is very close to orthogonal, up to machine precision.
"""

# %%
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
### Loss of orthogonality and numerical instabilities

In exact arithmetic, the Gram-Schmidt procedure works to produce a valid QR factorization. However, in the inexact world of floating point arithmetic, the method is numerically unstable and floating point roundoff errors accumulate, leading to a loss of orthogonality in $Q$. When $Q$ is not orthogonal, the definition of $R$ is incorrect and does not lead to a correct factorization.

The instabilities in Gram-Schmidt can be observed for very large matrices, or for ill-conditioned matrices. Our first example is for large matrices.
"""

# %%
A = np.random.rand(1000,1000)
Q,R = QR_GS(A)
print(np.linalg.norm(A - Q@R))
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
Note that both $\|A-QR\|$ and $\|I - Q^TQ\|$ are much larger than the machine precision of around $10^{-15}$ we saw earlier, but they are still quite small.


"""

# %%
"""
Random matrices are in fact quite easy to compute with as they are often well-conditioned. The situation can be substantially worse for other matrices that are poorly conditioned. An example of an ill-conditioned matrix is the Hilbert matrix
$$A = \begin{pmatrix}
1 & \frac12 & \frac13 & \cdots & \frac1n\\
\frac12 & \frac13 & \frac14 & \cdots & \frac{1}{n+1}\\
\frac13& \frac14 & \frac15 & \cdots & \frac{1}{n+2}\\
\vdots & \vdots & \vdots & \ddots & \vdots\\
\frac1n & \frac{1}{n+1} & \frac{1}{n+2} & \cdots & \frac{1}{2n-1}
\end{pmatrix}.$$

"""

# %%
def hilbert_matrix(n):
    """Hilbert Matrix
    ======
    Returns the nxn Hilbert matrix with (i,j) entry 1/(i+j+1), i,j=0,...,n-1

    Parameters
    ----------
    n : int
        Size of matrix (nxn)

    Returns
    -------
    A : numpy array, float
        Hilbert Matrix
    """

    x = np.arange(n)[:,None]
    A = 1/(x + x.T + 1)
    return A

# %%
"""
Check the performance of your QR algorithm on the Hilbert matrix.
"""

# %%
A = hilbert_matrix(20)
Q,R = QR_GS(A)
print(np.linalg.norm(A - Q@R,ord=np.inf))
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
Your code likely produced `nan` (not a number) and did not work. Can you figure out why? The Hilbert matrix is non-singular, but very poorly conditioned.


"""

# %%
"""
### Exercise

To fix this, modify your QR code to do the computation in the following way.

1. Compute some of the coefficients of the $k$th column of $R$, that is $r_{jk} = \mathbf{u}_j\cdot \mathbf{v}_k$ for $j \leq k-1$ and $r_{jk}=0$ for $j>k$.
2. Compute $\mathbf{x}_k = \mathbf{v}_k - \sum_{j-1}^{k-1} r_{jk}\mathbf{u}_j$ and $r_{kk} = \|\mathbf{x}_k\|$.
3. Compute the $k$th column of $Q$,  $\mathbf{u}_k = \frac{\mathbf{x}_k}{r_{kk}}$.

After making your modifications, try the Hilbert matrix again.
"""

# %%
A = hilbert_matrix(20)
Q,R = QR_GS(A)
print(np.linalg.norm(A - Q@R,ord=np.inf))
print(np.linalg.norm(Q.T@Q - np.eye(Q.shape[0])))

# %%
"""
Your code should run this time and produce real numbers. You should see that $A=QR$ up to machine precision, but that $Q$ is very var from being orthogonal. This is the result of the accumulation of floating point roundoff errors, which is most evident in ill-conditioned matrices like the Hilbert matrix.

If this is your first experience thinking about floating point numbers and roundoff errors, try running the code below, which should return 1 (it does not, due to accumulation of floating point roundoff errors, which is particularly bad when operating with very small and very large numbers at the same time). The exercise above also illustrates how the way a computation is performed can matter, even if it is mathematically equivalent!
"""

# %%
print(1+1e-16)
print(0.6 == 0.6)
print(0.1 + 0.2 + 0.3 == 0.6)
print(0.1 + 0.2 + 0.3)

# %%
n = int(1e6)
b = 1e6
c = b

#In exact arithmetic, the loop below just adds 1 to c, and is the same as c=c+1
#In floating point arithmetic, the errors in c += 1/n accumulate, expecially
#since 1/n is far smaller than c.
for i in range(n):
    c += 1/n

#Since we start at c=b and the loop above should just perform c=c+1
#we should have c=b+1 and so c-b=1. This is not the case in floating point
#arithmetic.
print(c-b)

# %%
"""
### Additional exercises





"""

# %%
"""
1. QR algorithms in Python packages.
  * Find an implementation of QR factorization in Numpy or Scipy (or any other Python package).
  * Compare the implementation of QR that you found to Gram-Schmidt and Modified Gram-Schmidt. Compute both $\|A-QR\|$ and $\|I - Q^TQ\|$ for all three methods. Try random matrices, and ill-conditioned ones.
  * Can you find out what algorithm is used to compute QR for the implementation you found?
  * Compare the run-times of all three algorithms for large matrices, say around $1000\times 1000$.
"""

# %%
import time

#Recall this is how we can time the execution of code in python
start_time = time.time()
#Write code
print("Time taken: %s seconds." % (time.time() - start_time))

# %%
"""
2. Write Python code to solve the linear system $A\mathbf{x}=\mathbf{b}$ with $QR$ factorization. Test your method on some large random matrices. You can use `np.linalg.inv` to compute the inverse of the upper triangular matrix $R$.
"""

# %%
