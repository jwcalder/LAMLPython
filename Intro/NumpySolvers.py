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
0 & 0 & -1 & 4 \\  
$$
which appears in the linear solvers for certain partial differential equations involving the Laplacian. 
"""

# %%
import numpy as np

A = np.random.rand(5,3)


