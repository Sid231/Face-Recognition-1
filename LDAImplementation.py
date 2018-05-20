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
numPerImages = 10

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

###################################################################### GET THE PCA TRANSFORMATION MATRIX ######################################################################

meanFeatureMatrix = np.zeros([featureMatrixTrainSet.shape[0],1])
for x in range(0, featureMatrixTrainSet.shape[0]):
    meanFeatureMatrix[x,0] = np.mean(featureMatrixTrainSet[x,:])
    
imageDifferenceFeatureMatrix = np.zeros([featureMatrixTrainSet.shape[0],featureMatrixTrainSet.shape[1]])
imageDifferenceFeatureMatrix = featureMatrixTrainSet - meanFeatureMatrix
covarianceMatrix = np.matmul(imageDifferenceFeatureMatrix.T,imageDifferenceFeatureMatrix)
eigvalue, eigVector = np.linalg.eig(covarianceMatrix)
eigVectorForEigenfaces = np.matmul(imageDifferenceFeatureMatrix,eigVector)
eigValueForEigenfaces = eigvalue
eigPairs = [(eigValueForEigenfaces[i], eigVectorForEigenfaces[:,i]) for i in range(len(eigValueForEigenfaces))]
eigPairs.sort(key=lambda x: x[0], reverse=True)
eigVectorForEigenfacesSorted = np.zeros([eigVectorForEigenfaces.shape[0],eigVectorForEigenfaces.shape[1]])
eigValueForEigenfacesSorted = np.zeros([eigValueForEigenfaces.shape[0]])

for col in range(0,len(eigValueForEigenfaces)):
    eigVectorForEigenfacesSorted[:,col] = eigPairs[col][1]
    eigValueForEigenfacesSorted[col] = eigPairs[col][0]
#normalization
eigenFacesMatrixNormVector = eigVectorForEigenfacesSorted/np.linalg.norm(eigVectorForEigenfacesSorted)
print(eigVectorForEigenfacesSorted.shape)
maxNumberOfEV = 100
eigenValueForEigenFaceMatrix = eigValueForEigenfacesSorted[0:maxNumberOfEV]
eigenFacesMatrix = eigenFacesMatrixNormVector[:,0:maxNumberOfEV]
eigenFacesMatrix = eigenFacesMatrixNormVector[:,2:maxNumberOfEV]
LDAProjectedinput = np.matmul(eigenFacesMatrix.T,featureMatrixTrainSet)

###################################################################### GET THE SCATTER B MATRIX (SB) #############################################################################

totalNumberOfClasses = int(LDAProjectedinput.shape[1]/numPerImages)

meanMatrixOfClasses = np.zeros([LDAProjectedinput.shape[0],totalNumberOfClasses])
totalMeanMatrix = np.zeros([LDAProjectedinput.shape[0],1])

for i in range(0,totalNumberOfClasses):
    for j in range(0,LDAProjectedinput.shape[0]):
        meanMatrixOfClasses[j,i] = np.mean(LDAProjectedinput[j,numPerImages*i:numPerImages*i+numPerImages])

for k in range(0,meanMatrixOfClasses.shape[0]):
    totalMeanMatrix[k,0] = np.mean(meanMatrixOfClasses[k,:])

scatterBMatrix = np.zeros([meanMatrixOfClasses.shape[0],meanMatrixOfClasses.shape[0]])
meanDifferenceMatrix = np.zeros([meanMatrixOfClasses.shape[0],1])
for i in range(0,meanMatrixOfClasses.shape[1]):
    meanDifferenceMatrix = meanMatrixOfClasses[:,i] - totalMeanMatrix
    scatterBMatrixAdder = np.matmul(meanDifferenceMatrix,meanDifferenceMatrix.T)*numPerImages
    scatterBMatrix += scatterBMatrixAdder

###################################################################### GET THE SCATTER W MATRIX ####################################################################################

scatterWMatrix = np.zeros([meanMatrixOfClasses.shape[0],meanMatrixOfClasses.shape[0]])
for i in range(0,totalNumberOfClasses):
    meanDifferenceSetMatrix = np.zeros([meanMatrixOfClasses.shape[0],numPerImages])
    for j in range(0,numPerImages):
        tempArray = []
        for k in range(0,LDAProjectedinput[:,numPerImages*i+j:numPerImages*i+j+1].shape[0]):
            tempArray.append(LDAProjectedinput[:,numPerImages*i+j:numPerImages*i+j+1][k][0])
        tempArray = np.array(tempArray)
        meanDifferenceSetMatrix[:,j] = tempArray - meanMatrixOfClasses[:,i]
    scatterWMatrix = scatterWMatrix + np.matmul(meanDifferenceSetMatrix,meanDifferenceSetMatrix.T)

multiplierForEV = np.matmul(np.linalg.inv(scatterWMatrix),scatterBMatrix)
eigvalueFisher, eigVectorFisher = np.linalg.eig(multiplierForEV)

fisherPairs = [(eigvalueFisher[i], eigVectorFisher[:,i]) for i in range(len(eigvalueFisher))]
fisherPairs.sort(key=lambda x: x[0], reverse=True)
eigVectorForFisherfacesSorted = np.zeros([eigVectorFisher.shape[0],eigVectorFisher.shape[1]])
eigValueForFisherfacesSorted = np.zeros([eigvalueFisher.shape[0]])
for col in range(0,len(eigvalueFisher)):
    eigVectorForFisherfacesSorted[:,col] = fisherPairs[col][1]
    eigValueForFisherfacesSorted[col] = fisherPairs[col][0]
maxNumberOfEVFisher = len(eigvalueFisher)
eigenValueForFisherFaceMatrix = eigValueForFisherfacesSorted[0:maxNumberOfEVFisher]
eigenFisherFacesMatrix = eigVectorForFisherfacesSorted[:,0:maxNumberOfEVFisher]

#normalization
eigenFisherNormFacesMatrix = eigenFisherFacesMatrix/np.linalg.norm(eigenFisherFacesMatrix)

###################################################################### GET THE LDA TRANSFORMATION MATRIX ######################################################################

W_FLD = eigenFisherNormFacesMatrix

###################################################################### GET THE OPTIMUM TRANSFORMATION MATRIX ##################################################################

W_OPT = np.matmul(eigenFacesMatrix,W_FLD)

###################################################################### GET THE DIFFERENCE OF DATA WITH THE MEAN OF DATA ########################################################

featureTestMean = np.empty([featureMatrixTestSet.shape[0],1])
featureTrainMean = np.empty([featureMatrixTrainSet.shape[0],1])

featureImageDifferenceTestMatrix = np.empty([featureMatrixTestSet.shape[0],featureMatrixTestSet.shape[1]])
featureImageDifferenceTrainMatrix = np.empty([featureMatrixTrainSet.shape[0],featureMatrixTrainSet.shape[1]])

for x in range(0, featureMatrixTestSet.shape[0]):
    featureTestMean[x,0] = np.mean(featureMatrixTestSet[x,:])

for x in range(0, featureMatrixTrainSet.shape[0]):
    featureTrainMean[x,0] = np.mean(featureMatrixTrainSet[x,:])

featureImageDifferenceTestMatrix = featureMatrixTestSet - featureTestMean
featureImageDifferenceTrainMatrix = featureMatrixTrainSet - featureTrainMean

###################################################################### GET THE PROJECTION OF TRAIN AND TEST DATA #################################################################

testMatrixProjection = np.matmul(W_OPT.T,featureImageDifferenceTestMatrix)
trainMatrixProjection = np.matmul(W_OPT.T,featureImageDifferenceTrainMatrix)

###################################################################################### RUN SVM ###################################################################################

# training a linear SVM classifier
svm_model_linear = svm.SVC(kernel = 'linear', C = 1).fit(trainMatrixProjection.T,labelMatrixTrainSet)
svm_predictions = svm_model_linear.predict(testMatrixProjection.T)
# model accuracy for X_test  
accuracy = svm_model_linear.score(testMatrixProjection.T, labelMatrixTestSet)
print (accuracy*100)