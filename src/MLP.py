# this file is to split train and test model 

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
import pickle
import joblib
from sklearn.neural_network import MLPClassifier 
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, Normalizer
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score

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


# Split the train and test datasets 
images = np.array(data)
labels = np.array(labels)
print(labels)

#Splitting training and testing dataset
train_images, test_images, train_labels, test_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)
print(train_images.shape, test_images.shape, train_labels.shape, test_labels.shape)

#Preprosessing the data 
#remove the last element of each X train and X test datasets 
train_images = train_images[:,:,:,0]   
test_images = test_images[:,:,:,0]    

#Change from matrix to array of diemension 32*32 to array of dimension 1024
dim_data = np.prod(train_images.shape[1:])
train_data = train_images.reshape(train_images.shape[0], dim_data)
test_data = test_images.reshape(test_images.shape[0], dim_data)

# Normalization (increases accuracy)
# Convert the data to float and scale the values between 0 to 1


scalify = StandardScaler()
train_data = scalify.fit_transform(train_data)
test_data = scalify.transform(test_data)
# Change the labels from integer to categorical data
train_labels_one_hot = tf.keras.utils.to_categorical(train_labels)
test_labels_one_hot = tf.keras.utils.to_categorical(test_labels)
# print(train_data)
# print(test_data)
# print(train_labels_one_hot)
# print(test_labels_one_hot)

# create the network 
model = tf.keras.Sequential([
    #tf.keras.layers.Flatten(input_shape=(32, 32, 3)),
    tf.keras.layers.Dense(256, activation='relu',input_shape=(dim_data,)),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(43, activation='softmax'),
])


# #model.summary()
model.compile(optimizer='adam',
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

#checking the model's performance using model.evaluate 
model.fit(train_data, train_labels_one_hot, epochs = 20,batch_size=256, verbose =1, validation_data=(test_data,test_labels_one_hot))

# #This was used for checking for overfitting (commented out)
# #Plot the Loss Curves
# history = model.fit(train_data, train_labels_one_hot, epochs = 20,batch_size=256, verbose =1, validation_data=(test_data,test_labels_one_hot))
# plt.figure(figsize=[8,6])
# plt.plot(history.history['loss'],'r',linewidth=3.0)
# plt.plot(history.history['val_loss'],'b',linewidth=3.0)
# plt.legend(['Training loss', 'Validation Loss'],fontsize=18)
# plt.xlabel('Epochs ',fontsize=16)
# plt.ylabel('Loss',fontsize=16)
# plt.title('Loss Curves',fontsize=16)
# plt.show()

# #Plot the Accuracy Curves
# plt.figure(figsize=[8,6]) 
# plt.plot(history.history['accuracy'],'r',linewidth=3.0)
# plt.plot(history.history['val_accuracy'],'b',linewidth=3.0)
# plt.legend(['Training Accuracy', 'Validation Accuracy'],fontsize=18)
# plt.xlabel('Epochs ',fontsize=16) 
# plt.ylabel('Accuracy',fontsize=16) 
# plt.title('Accuracy Curves',fontsize=16)
# plt.show()

# # save the model in joblib file 
filename = f"MLP_Trained_Model.joblib"                                            
joblib.dump(model, filename)
#MLP_joblib = joblib.load('MLP_Trained_Model.joblib')

y_pred = model.predict(test_data).argmax(axis=1)
#print(y_pred)

# convert string list to int list (for classification report?)
test_labels= list(map(int,test_labels ))

#print classification report
print(classification_report(test_labels, y_pred))
print('')

# show the confusion matrix (not normalized)
confusion_matrix=confusion_matrix(test_labels, y_pred)
print(confusion_matrix) 
plt.figure(figsize=(16,9))
sn.heatmap(confusion_matrix, annot=True, cmap=plt.cm.OrRd)
plt.show()   

# normalized confusion matrix
C= confusion_matrix / confusion_matrix.astype(np.float).sum(axis=1)
print(C)
plt.figure(figsize=(16,9))
sn.heatmap(C, annot=True, cmap=plt.cm.Purples)
plt.show() 

