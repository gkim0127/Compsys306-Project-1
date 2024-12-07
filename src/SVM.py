import numpy as np
from collections import Counter
import pandas as pd
from sklearn import svm
from sklearn.model_selection import GridSearchCV
import os
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import joblib
from sklearn.neural_network import MLPClassifier 
from PIL import Image
import tensorflow as tf
from skimage.feature import hog
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler, Normalizer
import skimage
import seaborn as sn
import matplotlib.pyplot as plt

# Read the categories from csv file 
data = pd.read_csv('labels.csv')
Categories = data['Name'].to_list()

data = []
labels = []
src= "C:\\Users\\GGPC\\Documents\\306 P1\\myData"
#Retrieving the images and their labels
for subdir in os.listdir(src):
        current_path = os.path.join(src, subdir)
        for file in os.listdir(current_path):
            if file[-3:] in {'jpg', 'png'}:
                im = imread(os.path.join(current_path, file))
                im= np.array(im)
                data.append(im)
                labels.append(subdir)

X = np.array(data)
y = np.array(labels)

#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

# functons for RGB to gray and HOG
from sklearn.base import BaseEstimator, TransformerMixin
 
class RGB2GrayTransformer(BaseEstimator, TransformerMixin):
    """
    Convert an array of RGB images to grayscale
    """
 
    def __init__(self):
        pass
 
    def fit(self, X, y=None):
        """returns itself"""
        return self
 
    def transform(self, X, y=None):
        """perform the transformation and return an array"""
        return np.array([skimage.color.rgb2gray(img) for img in X])
     
 
class HogTransformer(BaseEstimator, TransformerMixin):
    """
    Expects an array of 2d arrays (1 channel images)
    Calculates hog features for each img
    """
 
    def __init__(self, y=None, orientations=9,
                 pixels_per_cell=(8, 8),
                 cells_per_block=(3, 3), block_norm='L2-Hys'):
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
 
    def fit(self, X, y=None):
        return self
 
    def transform(self, X, y=None):
 
        def local_hog(X):
            return hog(X,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)
 
        try: # parallel
            return np.array([local_hog(img) for img in X])
        except:
            return np.array([local_hog(img) for img in X])
            from sklearn.linear_model import SGDClassifier


# create an instance of each transformer
grayify = RGB2GrayTransformer()
hogify = HogTransformer(
    pixels_per_cell=(14, 14), 
    cells_per_block=(2,2), 
    orientations=9, 
    block_norm='L2-Hys'
)

#Use standard scaler to normalise the data
scalify = StandardScaler()
 
# call fit_transform on each transform converting X_train step by step
X_train_gray = grayify.fit_transform(X_train)
X_train_hog = hogify.fit_transform(X_train_gray)
X_train_prepared = scalify.fit_transform(X_train_hog)

X_test_gray = grayify.transform(X_test)
X_test_hog = hogify.transform(X_test_gray)
X_test_prepared = scalify.transform(X_test_hog)
 
print(X_train_prepared.shape)

# # paramgrid search to find best parameters (commented out)
# param_grid={'C':[0.1,1,10,100],'gamma':[0.0001,0.001,0.1,1],'kernel':['linear','rbf','poly']}
# svc=svm.SVC(probability=True)
# print("The training of the model is started, please wait for while as it may take few minutes to complete")
# model=GridSearchCV(svc,param_grid)
# model.fit(X_train_prepared,y_train)
# print('The Model is trained well with the given images')
# print(model.best_params_)

# train the model with the parameters we found above by gridsearch
svc = svm.SVC(kernel= 'rbf', C= 10, gamma= 0.1, max_iter= 100000, probability = True)
model = svc.fit(X_train_prepared,y_train)

#To save the trained model in pkl file using joblib
filename = f"SVM_Trained_Model.joblib"  
joblib.dump(model, filename)
# load the model from saved joblib file
model_joblib = joblib.load('SVM_Trained_Model.joblib')

y_pred = model_joblib.predict(X_test_prepared)

#print classification report
print(classification_report(y_test, y_pred))
print('')

# show the confusion matrix (not normalized)
confusion_matrix=confusion_matrix(y_test, y_pred)
print(confusion_matrix)
plt.figure(figsize=(16,9))
sn.heatmap(confusion_matrix, annot=True, cmap=plt.cm.Blues)
plt.show()   

# normalized confusion matrix
C= confusion_matrix / confusion_matrix.astype(np.float64).sum(axis=1)
print(C)
plt.figure(figsize=(16,9))
sn.heatmap(C, annot=True, cmap=plt.cm.Greys)
plt.show() 
