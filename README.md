# mnist-pca

a. Compute the mean image and principal components for a set of images (e.g. use the
training images of ‘5’ in the mnist dataset). Display the mean image and the first 2
principal components (associated with the highest eigenvalues).
b. Compute and display the reconstructions of a test image using the mean image and with
p principal components associated with the p highest eigenvalues (e.g. Fig 10.12) with
p=10 and p=50.
c. Read https://doi.org/10.1109/34.598227 ‘Probabilistic visual learning for object
representation’ (PAMI1997). Compute and display a DFFS (distance-from feature-space) and SSD (sum-of-square-differences) heat maps for detection using your PCA representation of a MNIST number. For the test image, use a composite image made of
MNIST test images (see example below).
d. Evaluate the performance of SSD and DFFS (i.e. illustrate when it works, and when it does
not work).
