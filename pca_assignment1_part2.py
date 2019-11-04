import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
import cv2
import seaborn as sns





def load_data_mnist():
    images,labels = fetch_openml('mnist_784', version=1, return_X_y=True)
    n_train= 60000 #The size of the training set
    train_images = images[:n_train]
    train_labels = labels[:n_train]
    test_images = images[n_train:]
    test_labels = labels[n_train:]
    return (train_images.astype(np.float32)/255,train_labels.astype(np.float32),
            test_images.astype(np.float32)/255,test_labels.astype(np.float32))
    





def compute_projectionMatrix_variance_mean(image_matrix):
    #Function that takes the image matrix as input and returns the projection
    #matrix, variance and mean image matrix 
    standard_deviation = np.std(image_matrix)
    num_data,dimension = image_matrix.shape
    
    #Centering by subtracting the mean from each data point
    mean_image_matrix = image_matrix.mean(axis=0)
    image_matrix = image_matrix - mean_image_matrix
    
    #Standardizing the dataset by dividing all the data points in the image matrix 
    #by the standard deviation to make the data unit free
    image_matrix = image_matrix/standard_deviation
    
    #if dimension>num_data:
    #Compute the co-variance matrix
    covariance_matrix = np.dot(image_matrix,image_matrix.T) 
        
    #Compute the Eigen Values and Eigen Vectors
    eigen_values,eigen_vectors = np.linalg.eigh(covariance_matrix) 
        
    #Get the highest Eigen vectors
    tmp = np.dot(image_matrix.T,eigen_vectors).T 
    highest_eigen_vectors = tmp[::-1] 
        
    #Get the highest Eigen values
    variance = np.sqrt(eigen_values)[::-1] 
    for i in range(highest_eigen_vectors.shape[1]):
        highest_eigen_vectors[:,i] /= variance
        
    '''        
    else:
        #Performing Single Value Decomposition
        U,variance,highest_eigen_vectors = np.linalg.svd(image_matrix)
    '''    
    projection_matrix = highest_eigen_vectors[:num_data]
    
    #Undo standardisation by multiplying the projection matrix with standard
    #deviation to obtain projection in the original data space   
    projection_matrix = projection_matrix * standard_deviation

    return projection_matrix, variance, mean_image_matrix





def displayMean(mean_image):
    plt.figure()
    plt.title('Mean Image')
    plt.gray()
    #plt.subplot(2,4,1)
    plt.imshow(mean_image.reshape(28,28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    
    
    
    
def displayPrincipalComponents(projection_matrix):
    plt.figure()
    plt.title('First two Principal Components')
    plt.gray()
    for i in range(2):
        #plt.subplot(2,4,i+2)
        plt.imshow(projection_matrix[i].reshape(28,28))
        plt.show()
    cv2.waitKey(0)
    cv2.destroyAllWindows()
        




def computeDisplayReconstructionMatrix(n,mean_image,variance,image_matrix,projection_matrix):
    
    pca_n = projection_matrix[:n]
    test_img = image_matrix[0]
    
    #Performing transformation on the test image with projection matrix
    data_reduced = np.dot(test_img, pca_n.T) 
    
    #Performing inverse transform to generate the reconstruction matrix
    reconstruction_matrix = np.dot(data_reduced, pca_n) 
	
    plt.figure()
    plt.title('Test Image')
    plt.gray()
    plt.imshow(test_img.reshape(28,28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    plt.figure()
    plt.title('Reconstructed Image with P = ' +str(n))
    plt.gray()
    plt.imshow(reconstruction_matrix.reshape(28,28))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return

    
    
    
def generateDisplaySSDHeatMap(train_images,train_labels,mean_image):    
    
    k =  []
    #Generate the test image - a composite image made of MNIST test images
    for i in range(4):
        for j in range(10):
            label_i = np.where(train_labels == j)
            k.append(train_images[label_i[0][i]])
    temp = np.hstack( (np.asarray([ i.reshape(28,28) for i in k ])))
    k =  []
    for i in range(0,temp.shape[1],280):
        k.append(np.array(temp[0:28,0+i:280+i]))
    test_image = np.vstack(x for x in k)
    
    plt.figure()
    plt.title('')
    plt.imshow(test_image,cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    a,b = test_image.shape
    #Construct the SSD vector
    ssd_vector = []
    for i in range(0,a,28):
        for j in range(0,b,28):
            ssd_vector.append(np.sum(np.square(np.subtract(test_image[i:28+i,j:28+j],mean_image.reshape(28,28)))))
    
    #Plot the SSD Hear Map    
    plt.figure()
    plt.title('SSD Heat map')
    sns.heatmap(np.array(ssd_vector).reshape(4,10),cmap='gray')
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return





def run():
    
    #Load the MNIST dataset
    train_images,train_labels,test_images,test_labels = load_data_mnist() 
    
    #2A - Compute the mean image and principal components for the training images of '5'
    labels_5 = np.where(train_labels == 5)
    image_matrix = train_images[list(labels_5)]
    
    projection_matrix,variance,mean_image = compute_projectionMatrix_variance_mean(image_matrix)
    
    #2A - Display the Mean image
    displayMean(mean_image)
    
    #2A - Display the first two Principal components
    displayPrincipalComponents(projection_matrix)
    
    generateDisplaySSDHeatMap(train_images,train_labels,mean_image)
    
    #2B - Compute and display the reconstructions of a test image using the learned projection matrix and with
    #matrix and with p=10 and p=50
    computeDisplayReconstructionMatrix(10,mean_image,variance,image_matrix,projection_matrix)
    computeDisplayReconstructionMatrix(50,mean_image,variance,image_matrix,projection_matrix)
    
    
    #For testing, try p = 500 - Reconstruction image should be near perfect to the test image
    computeDisplayReconstructionMatrix(500,mean_image,variance,image_matrix,projection_matrix)
    #For testing, try p = 784 - Reconstruction image should be exactly the same test image
    #without any compression loss
    computeDisplayReconstructionMatrix(784,mean_image,variance,image_matrix,projection_matrix)
    
    
    
    



if __name__ == '__main__':
    run()
    
def main():
    run()
    

