# JupyterCon 2018
Notebooks and code of my 2018 JupyterCon [talk](https://conferences.oreilly.com/jupyter/jup-ny/public/schedule/detail/71406)


## Dependencies
The notebooks and code depend on the following python packages:

- `numpy` (Version 1.14.0)
- `scipy` (Version 1.0.0)
- `pandas` (Version 0.22.0)
- `sklearn` (Version 0.19.1)
- `tensorflow`(Version 1.2.1)
- `keras` (Version 2.0.6)
- `ipywidgets` (Version 7.1.1)
- `bqplot` (Version 0.11.1)

## Notes
* All the examples in the talk were run in a black-themed notebook (and black-themed `ipywidgets` and `bqplot`). The colors used in the visualizations and plots were chosen to work well on the black background. Since the classic notebooks are white-themed, colors in the plots (especially lighter colors like yellow) need to be updated accordingly.
* Special add-ons like full-screen button etc. were used to render the visualizations on a full screen. These are not available by default in the classic Jupyter notebook.

(Hopefully with Jupyterlab, black-themed notebooks/widgets and full-screen rendering will be available for all)

The following models/algorithms/tools were covered in the talk:
* Kernel Regression
![alt text](images/kernel_regression.png "Kernel Regression")

* K-Means Clustering
![alt text](images/kmeans.png "K-Means Clustering")

* Dimensionality Reduction (PCA vs Autoencoder)
![alt text](images/pca_ae1.png "Visualization In 2 Dimensions")
![alt text](images/pca_ae2.png "PCA vs Autoencoder Comparison")

* Neural Network Builder
![alt text](images/netbuilder1.png "Neural Net Builder")
![alt text](images/netbuilder2.png "Neural Net Builder - Loss/Accuracy Curves")
![alt text](images/netbuilder3.png "Neural Net Builder - Distributions of Weights/Biases/Activations")

## TODO
* Improve documentation
* Make notebooks run in binder
