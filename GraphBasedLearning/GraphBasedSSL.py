# %%
"""
#Graph-based semisupervised learning

This is a brief demo of graph-based semi-supervised learning using the [Graph Learning](https://github.com/jwcalder/GraphLearning) package.
"""

# %%
#pip install -q graphlearning annoy

# %%
"""
We first consider the two-moons data set. The red stars are the locations of the labeled nodes. See how well you can do with one label per moon.
"""

# %%
import numpy as np
import graphlearning as gl
import matplotlib.pyplot as plt
import sklearn.datasets as datasets
plt.ion()

#Draw data randomly and build a k-nearest neighbor graph with k=10 neighbors
X,labels = datasets.make_moons(n_samples=500,noise=0.1)
W = gl.weightmatrix.knn(X,10)

#Generate training data
train_ind = gl.trainsets.generate(labels, rate=3)
train_labels = labels[train_ind]

#Semi-supervsied learning
model = gl.ssl.laplace(W)
pred_labels = model.fit_predict(train_ind, train_labels)

#Compute accuracy
accuracy = gl.ssl.ssl_accuracy(pred_labels, labels, train_ind)
print("Accuracy: %.2f%%"%accuracy)

#Make plots
plt.figure()
plt.scatter(X[:,0],X[:,1], c=pred_labels)
plt.scatter(X[train_ind,0],X[train_ind,1], c='r', marker='*', s=100)
plt.show()

# %%
"""
We can now run an experiment classifying MNIST digits. We first load the dataset and display some images.
"""

# %%
import graphlearning as gl

#Load MNIST data
data,labels = gl.datasets.load('mnist')

#Display images
gl.utils.image_grid(data,n_rows=16,n_cols=16)

# %%
"""
Now let's try some semi-supervised learning on MNIST. We'll show the results of Laplace learning and graph nearest neighbors. The methods available in the package are listed in the documentation here: https://jwcalder.github.io/GraphLearning/ssl.html
"""

# %%
import graphlearning as gl

W = gl.weightmatrix.knn('mnist', 10)
D = gl.weightmatrix.knn('mnist', 10, kernel='distance')

num_train_per_class = 100
train_ind = gl.trainsets.generate(labels, rate=num_train_per_class)
train_labels = labels[train_ind]

models = [gl.ssl.graph_nearest_neighbor(D), gl.ssl.laplace(W)]

for model in models:
    pred_labels = model.fit_predict(train_ind,train_labels)
    accuracy = gl.ssl.ssl_accuracy(labels,pred_labels,train_ind)
    print(model.name + ': %.2f%%'%accuracy)

#Plot some of the misclassified images and their labels
ind_incorrect = labels != pred_labels
gl.utils.image_grid(data[ind_incorrect,:],title='Misclassified')
print(pred_labels[ind_incorrect][:10])
print(labels[ind_incorrect][:10])

# %%
"""
##Exercise

1. Try playing around with the label rate above. How do things work for 1 label per class?
2. Choose another graph from [GraphLearning](https://jwcalder.github.io/GraphLearning/datasets.html#graphlearning.datasets.load_graph) to try Laplace learning on. For example, try PubMed.
3. Write a function label propagation (i.e., gradient descent) to solve the Laplace learning equation, as described in the book. Can you achieve higher accuracy by stopping early?
"""
