#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import cv2
from random import shuffle
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau , ModelCheckpoint
from collections import Counter
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.applications import InceptionV3,InceptionResNetV2
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.applications.resnet50 import ResNet50

from keras.applications import DenseNet201, Xception


# In[2]:


#data_path = 'H:/PhD/prof ref/INbreast1/work 7 with aug/'

data_path = 'H:/PhD/prof ref/INbreast1/Ensemble binary/INbreast 2 encoder/'

TrianImage = "H:/PhD/prof ref/INbreast1/Ensemble binary/INbreast 2 encoder/train/"
ValImage = "H:/PhD/prof ref/INbreast1/Ensemble binary/INbreast 2 encoder/val/"
TestImage = "H:/PhD/prof ref/INbreast1/Ensemble binary/INbreast 2 encoder/test/"



abnormal_images_trainig = os.listdir(TrianImage + "/Abnormal")
normal_images_trainig = os.listdir(TrianImage + "/Normal")


print('No of abnormal images training : ',len(abnormal_images_trainig) )
print('No of normal images training : ',len(normal_images_trainig) )

Nos_Train =  len(abnormal_images_trainig) + len(normal_images_trainig)

print('No of training images : ',Nos_Train )


# In[80]:



abnormal_images_testing = os.listdir(TestImage + "/Abnormal")
normal_images_testing = os.listdir(TestImage + "/Normal")


print('No of abnormal images testing : ',len(abnormal_images_testing) )
print('No of normal images testing : ',len(normal_images_testing) )

Nos_Test =  len(abnormal_images_testing) + len(normal_images_testing)

print('No of testing images : ',Nos_Test )


# In[81]:



abnormal_images_val = os.listdir(ValImage + "/Abnormal")
normal_images_val = os.listdir(ValImage + "/Normal")


print('No of abnormal images val : ',len(abnormal_images_val) )
print('No of normal images val : ',len(normal_images_val) )

Nos_Val =  len(abnormal_images_val) + len(normal_images_val)

print('No of validation images : ',Nos_Val )


# In[82]:


image_size = 512
BATCH_SIZE = 4
STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)


# In[83]:


train_datagen = ImageDataGenerator(#rescale = 1./255,
                                   zoom_range = 0.2,
                                   rotation_range=15,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator()
val_datagen = ImageDataGenerator()


training_set = train_datagen.flow_from_directory(data_path + '/train',
                                                 target_size = (image_size, image_size),
                                                 batch_size = BATCH_SIZE,
                                                 class_mode = 'categorical', shuffle=True)

testing_set = test_datagen.flow_from_directory(data_path + '/test',
                                               target_size = (image_size, image_size),
                                               batch_size = BATCH_SIZE,
                                               class_mode = 'categorical', shuffle = False)

validating_set = val_datagen.flow_from_directory(data_path + '/val',
                                               target_size = (image_size, image_size),
                                               batch_size = BATCH_SIZE,
                                               class_mode = 'categorical', shuffle = False)


# In[84]:


def display_training_curves(training_accuracy, validation_accuracy, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_accuracy)
    ax.plot(validation_accuracy)
    ax.set_title('Model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Accuracy', 'Val_Accuracy'])


# In[ ]:





# In[85]:


def display_training_curves2(training_loss, validation_loss, title, subplot):
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_loss)
    ax.plot(validation_loss)
    ax.set_title('Model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Loss', 'Val_Loss'])


# In[86]:


def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


# In[ ]:





# In[87]:


visible = tf.keras.layers.Input(shape=(image_size, image_size, 3))


# In[88]:


model_VGG16 = VGG16(include_top=False,weights='imagenet',input_shape=(image_size, image_size, 3))
model_ResNet50 = ResNet50(include_top=False,weights='imagenet',input_shape=(image_size, image_size, 3))

#model_ResNetV2 = InceptionResNetV2(include_top=False,weights='imagenet',input_shape=(image_size, image_size, 3))
model_DenseNet201 = DenseNet201(include_top=False,weights='imagenet',input_shape=(image_size, image_size, 3))

model_Xception = Xception(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False)


# In[ ]:





# In[92]:


history = model.fit(training_set, validation_data=validating_set, callbacks=[tl_checkpoint_1], epochs=5)


# In[93]:


display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
plt.show()


# In[118]:


model.load_weights('Ensemble_binary_model_v2.weights.best'+'.hdf5') # initialize the best trained weights
#vgg_model_ft=tf.keras.models.load_model('tl_model_v1.weights.best.hdf5') # initial'ize the best trained weights

model.evaluate(training_set)

model.evaluate(testing_set)

model.evaluate(validating_set)


# In[119]:


from sklearn.metrics import confusion_matrix,classification_report

Y_pred2 = model.predict(testing_set)
Y_pred2 = np.argmax(Y_pred2, axis=1)


# In[120]:


testing_set.classes


# In[121]:


Y_pred2


# In[122]:


print('Classification Report')
print(classification_report(testing_set.classes, Y_pred2))


# In[123]:


cm = confusion_matrix(testing_set.classes, Y_pred2)
print(cm)


# In[124]:


conf_matrix = confusion_matrix(y_true=testing_set.classes, y_pred=Y_pred2)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Reds, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='small')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[125]:


TP = cm[0,0]
TN = cm[1,1]
FP = cm[0,1]
FN = cm[1,0]

classification_accuracy = ((TP + TN) / float(TP + TN + FP + FN)*100)
precision = (TP / float(TP + FP)*100)
recall = (TP / float(TP + FN)*100)
specificity = (TN / (TN + FP)*100)
f1_scores= 2 * (precision*recall) /(precision+recall)


print('accuracy : {0:0.2f}'.format(classification_accuracy),'%')
print('')
print('Recall or Sensitivity : {0:0.2f}'.format(recall),"%")
print('Specificity : {0:0.2f}'.format(specificity),'%')
print('Precision : {0:0.2f}'.format(precision),"%")
print('f1_scores : {0:0.2f}'.format(f1_scores),"%")


# In[126]:


from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_fscore_support

# Recall Score (Sensitivity)
# Precision Score (Specificity)  
# accuracy, sensitivity, specificity, precision, F-1 score and AUC.
# (None, 'micro', 'macro', 'weighted', 'samples')

score = accuracy_score(testing_set.classes, Y_pred2)  


print('Accuracy: %.4f' % (accuracy_score(testing_set.classes, Y_pred2)))
print('Recall ( Sen): %.4f' % (recall_score(testing_set.classes, Y_pred2,average = 'macro')))
print('Precision (Spe): %.4f' % (precision_score(testing_set.classes, Y_pred2,average = 'macro')))
print('F1 score : %.4f' % (f1_score(testing_set.classes, Y_pred2, average='macro')))


# In[127]:


# def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
#     lb = LabelBinarizer()
#     lb.fit(y_test)
#     y_test = lb.transform(y_test)
#     y_pred = lb.transform(y_pred)
#     return roc_auc_score(y_test, y_pred, average=average)


# In[128]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, _ = roc_curve(testing_set.classes, Y_pred2)
print('AUC : %.4f' % (auc(fpr, tpr)))


# In[ ]:





# In[ ]:





# In[140]:


model.load_weights('Ensemble_binary_model_v4.weights.best'+'.hdf5')


# In[129]:



eval_path4='H:/PhD/prof ref/INbreast1/Ensemble binary/private eval encoder/'


# In[150]:



test_datagen1 = ImageDataGenerator()

testing_set1 = test_datagen1.flow_from_directory(eval_path4 ,
                                               target_size = (image_size, image_size),
                                               batch_size = BATCH_SIZE,
                                               class_mode = 'categorical', shuffle = False)


# In[151]:


from sklearn.metrics import confusion_matrix,classification_report

Y_pred3 = model.predict(testing_set1)
Y_pred3 = np.argmax(Y_pred3, axis=1)


# In[152]:


testing_set1.classes


# In[153]:


Y_pred3


# In[154]:



print('Classification Report')
print(classification_report(testing_set1.classes, Y_pred3))


# In[155]:


cm = confusion_matrix(testing_set1.classes, Y_pred3)
print(cm)


# In[156]:


print('Accuracy: %.3f' % accuracy_score(testing_set1.classes, Y_pred3))


# In[157]:


conf_matrix = confusion_matrix(y_true=testing_set1.classes, y_pred=Y_pred3)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Reds, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='small')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[160]:


from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_fscore_support

# Recall Score (Sensitivity)
# Precision Score (Specificity)  
# accuracy, sensitivity, specificity, precision, F-1 score and AUC.
# (None, 'micro', 'macro', 'weighted', 'samples')

score = accuracy_score(testing_set.classes, Y_pred2)  


print('Accuracy: %.4f' % (accuracy_score(testing_set1.classes, Y_pred3)))
print('Recall ( Sen): %.4f' % (recall_score(testing_set1.classes, Y_pred3,average = 'macro')))
print('Precision (Spe): %.4f' % (precision_score(testing_set1.classes, Y_pred3,average = 'macro')))
print('F1 score : %.4f' % (f1_score(testing_set1.classes, Y_pred3, average='macro')))


# In[162]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, _ = roc_curve(testing_set1.classes, Y_pred3)
print('AUC : %.4f' % (auc(fpr, tpr)))


# In[ ]:





# In[ ]:





# In[ ]:




