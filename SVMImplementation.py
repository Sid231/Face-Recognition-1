import numpy as np
import scipy.io as spio
from matplotlib import pyplot as plt
from PIL import Image
from sklearn import svm
from sklearn.metrics import confusion_matrix
	
###################################################################### FUNCTION TO PRINT THE IMAGE ##################################################################################

def plot_face(features):
    im=Image.fromarray(features.reshape(32,32).T)
    print(im)
    plt.imshow(im)
    plt.show()
    
############################################################# FUNCTION TO RANDOMLY SEPARATE TRAIN AND TEST DATA ######################################################################

def randomlySegmentMatrices (featureMatrix,labelMatrix,tempArray,numPerImages,featureMatrix_train,featureMatrix_test,labelMatrix_train,labelMatrix_test):

    indices = np.random.permutation(tempArray)
    training_idx, test_idx = indices[:numPerImages], indices[numPerImages:]

    for i in range(0, len(training_idx)):
        featureMatrix_train.append(featureMatrix[training_idx[i]])
        labelMatrix_train.append(np.asscalar(labelMatrix[training_idx[i]]))

    for i in range(0, len(test_idx)):
        featureMatrix_test.append(featureMatrix[test_idx[i]])
        labelMatrix_test.append(np.asscalar(labelMatrix[test_idx[i]]))

    return featureMatrix_train,featureMatrix_test,labelMatrix_train,labelMatrix_test

################################################################################## MAIN CODE BEGINS #################################################################################

#random seed for consistency
np.random.seed(1) 

###################################################### IMPORT DATA FROM THE YALEB_32X32.MAT FILE INTO FEATURE AND LABEL MATRIX ######################################################

mat = spio.loadmat('YaleB_32x32.mat', squeeze_me=True)
featureMatrix = np.matrix(mat.get('fea'))
labelMatrix = np.matrix(mat.get('gnd')).T
featureMatrix_train = []
featureMatrix_test = []
labelMatrix_train = []
labelMatrix_test = []
numPerImages = 50

###################################################### DIVIDE FEATURE AND LABEL MATRIX RANDOMLY INTO TRAINING AND TESTING DATA ######################################################

labelValue = labelMatrix[0]
tempArray = []
flag=0
if(featureMatrix.shape[0] == labelMatrix.shape[0]):
    for x in range(0, featureMatrix.shape[0]):
        if(labelMatrix[x,0] == labelValue):
            if(flag == 1):
                tempArray.append(x-1)
                flag=0
            tempArray.append(x)
        else:
            flag=1
            featureMatrixTrainSet,featureMatrixTestSet,labelMatrixTrainSet,labelMatrixTestSet = randomlySegmentMatrices(featureMatrix,labelMatrix,tempArray,numPerImages,
                                                                                                        featureMatrix_train,featureMatrix_test,labelMatrix_train,labelMatrix_test)
            labelValue = labelMatrix[x]
            tempArray = []

    featureMatrixTrainSet,featureMatrixTestSet,labelMatrixTrainSet,labelMatrixTestSet = randomlySegmentMatrices(featureMatrix,labelMatrix,tempArray,numPerImages,
                                                                                                        featureMatrix_train,featureMatrix_test,labelMatrix_train,labelMatrix_test)

featureMatrixTrainSet = np.array(featureMatrixTrainSet)
featureMatrixTestSet = np.array(featureMatrixTestSet)
labelMatrixTrainSet = np.array(labelMatrixTrainSet)
labelMatrixTestSet = np.array(labelMatrixTestSet)

featureMatrixTrainSet = featureMatrixTrainSet[:,0,:].T
featureMatrixTestSet = featureMatrixTestSet[:,0,:].T

###################################################################################### RUN SVM ###################################################################################

# training a linear SVM classifier
svm_model_linear = svm.SVC(kernel = 'linear', C = 1).fit(featureMatrixTrainSet.T,labelMatrixTrainSet)
svm_predictions = svm_model_linear.predict(featureMatrixTestSet.T)
accuracy = svm_model_linear.score(featureMatrixTestSet.T, labelMatrixTestSet)
print (accuracy*100)