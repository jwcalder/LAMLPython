# %%
"""
# Solving linear equations in Numpy
This notebook overviews some basic approaches to solving linear systems and eigenvalue problems in [NumPy](https://NumPy.org/).
"""

# %%
"""
## Solving small linear systems
We can use the `numpy.linalg.solve` command to exactly solve small or moderate size linear systems of full rank. We can check if a system is full rank using `numpy.linalg.matrix_rank`. Below we show how to solve a linear system with matrix
$$A = \begin{pmatrix} 
4 & -1 & 0 & 0\\
-1 & 4 & -1 & 0\\  
0 & -1 & 4 & -1 \\  
0 & 0 & -1 & 4 
\end{pmatrix},
$$
which appears in the linear solvers for certain partial differential equations involving the Laplacian. 
"""

# %%
import numpy as np

A = np.array([[4,-1,0,0],
              [-1,4,-1,0],
              [0,-1,4,-1],
              [0,0,-1,4]])

k = np.linalg.matrix_rank(A)
print('Rank = ',k)

b = np.ones(4)
x = np.linalg.solve(A,b)
print('Solution = ',x)
print('A*x=',A@x,'b=',b)

# %%
"""
## Solving large sparse linear systems
Direct solvers for linear systems typically struggle with large systems of linear equations. When the matrix $A$ is has some structure, such as symmetric positive definitene and sparae, there are far more efficient solvers based on indirect iterative methods. Below we use the conjugate gradient method (covered in chapters 6 and 11 of the textbook) to solve a linear system with a matrix $A$ of the same form as above, but much larger. In this case, we use the `scipy` package, which has support for sparse matrices. The conjugage gradient method is 
"""

# %%
from scipy import sparse

n = 1000
o = np.ones(n)
A = sparse.diags(4*o) - sparse.diags(o[1:],1) - sparse.diags(o[1:],-1)
print(A)

x0 = np.random.randn(n) #Initialization
b = np.ones(n)
x,_ = sparse.linalg.cg(A,b,x0=x0)

print('Solution = ',x)
print('A*x=',A*x,'b=',b)




