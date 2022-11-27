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

from keras.applications import InceptionV3,InceptionResNetV2
from efficientnet.tfkeras import EfficientNetB7


# In[2]:


#data_path = 'H:/PhD/prof ref/INbreast1/work 7 with aug/'

data_path = 'H:/PhD/prof ref/INbreast1/Ensemble binary/INbreast 2 encoder/'

TrianImage = data_path+'train/'

abnormal_images_trainig = os.listdir(TrianImage + "/Abnormal")
normal_images_trainig = os.listdir(TrianImage + "/Normal")


print('No of abnormal images training : ',len(abnormal_images_trainig) )
print('No of normal images training : ',len(normal_images_trainig) )

Nos_Train =  len(abnormal_images_trainig) + len(normal_images_trainig)

print('No of training images : ',Nos_Train )


# In[3]:


#data_path = 'H:/PhD/prof ref/INbreast1/work 7 with aug/'

TestImage = data_path+'test/'


abnormal_images_testing = os.listdir(TestImage + "/Abnormal")
normal_images_testing = os.listdir(TestImage + "/Normal")


print('No of abnormal images testing : ',len(abnormal_images_testing) )
print('No of normal images testing : ',len(normal_images_testing) )

Nos_Test =  len(abnormal_images_testing) + len(normal_images_testing)

print('No of testing images : ',Nos_Test )


# In[4]:


#data_path = 'H:/PhD/prof ref/INbreast1/work 7 with aug/'

ValImage = data_path+'val/'

abnormal_images_val = os.listdir(ValImage + "/Abnormal")
normal_images_val = os.listdir(ValImage + "/Normal")


print('No of abnormal images val : ',len(abnormal_images_val) )
print('No of normal images val : ',len(normal_images_val) )

Nos_Val =  len(abnormal_images_val) + len(normal_images_val)

print('No of Vald images : ',Nos_Val )


# In[5]:


image_size = 512
BATCH_SIZE = 4
STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)


# In[6]:


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


# In[7]:


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


# In[8]:


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


# In[9]:


def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


# In[10]:



from keras.regularizers import l2
from keras.models import Model
from keras.optimizers import Adam
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications.vgg19 import VGG19
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Dense, Dropout, Flatten,GlobalAveragePooling2D,BatchNormalization



def create_model(input_shape, n_classes, dense=1024, fine_tune=0):
   
    base_model = InceptionV3(weights='imagenet',input_shape=(image_size, image_size, 3))

    model = Model(base_model.input, name="EfficientNetGoogleNet_Binary2")
    
    from tensorflow.keras.optimizers import Adam

    optimizer =  Adam(learning_rate=1e-2)
    model.compile(
        optimizer=optimizer, loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
   #optimizer=optimizer, loss="categorical_crossentropy"
   #optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2),loss=categorical_smooth_loss,
    return model


# In[56]:


import time
start = time.time()

history = googleNet_binary_model_ft.fit(training_set, validation_data=validating_set, epochs=5,callbacks=[tl_checkpoint_1],verbose=1) #30

end = time.time()

print('time cost: ')
print(end - start, 'seconds')


# In[ ]:





# In[57]:


display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
plt.show()


# In[15]:



googleNet_binary_model_ft.evaluate(training_set)

googleNet_binary_model_ft.evaluate(validating_set)


# In[16]:


import time
start = time.time()


googleNet_binary_model_ft.evaluate(testing_set)


end = time.time()
print('time cost: ')
print(end - start, 'seconds')


# In[17]:


from sklearn.metrics import confusion_matrix,classification_report

Y_pred2 = googleNet_binary_model_ft.predict(testing_set)
Y_pred2 = np.argmax(Y_pred2, axis=1)


# In[18]:


testing_set.classes


# In[19]:


Y_pred2


# In[20]:


print('Classification Report')
print(classification_report(testing_set.classes, Y_pred2))


# In[21]:


cm = confusion_matrix(testing_set.classes, Y_pred2)
print(cm)


# In[22]:


conf_matrix = confusion_matrix(y_true=testing_set.classes, y_pred=Y_pred2)
#
# Print the confusion matrix using Matplotlib
#
fig, ax = plt.subplots(figsize=(5, 5))
ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='large')
 
plt.xlabel('Predictions', fontsize=18)
plt.ylabel('Actuals', fontsize=18)
plt.title('Confusion Matrix', fontsize=18)
plt.show()


# In[22]:


from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_fscore_support

score = accuracy_score(testing_set.classes, Y_pred2)  

print('Accuracy: %.4f' % (accuracy_score(testing_set.classes, Y_pred2)))
print('Recall ( Sen): %.4f' % (recall_score(testing_set.classes, Y_pred2,average = 'macro')))
print('Precision (Spe): %.4f' % (precision_score(testing_set.classes, Y_pred2,average = 'macro')))
print('F1 score : %.4f' % (f1_score(testing_set.classes, Y_pred2, average='macro')))


# In[23]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, _ = roc_curve(testing_set.classes, Y_pred2)
print('AUC : %.4f' % (auc(fpr, tpr)))


# In[ ]:





# In[ ]:


eval_path4='H:/PhD/prof ref/INbreast1/Ensemble binary/private eval encoder/'


# In[29]:



test_datagen1 = ImageDataGenerator()

testing_set1 = test_datagen1.flow_from_directory(eval_path4,
                                               target_size = (image_size, image_size),
                                               batch_size = BATCH_SIZE,
                                               class_mode = 'categorical', shuffle = False)


# In[30]:


testing_set1.classes


# In[31]:


from sklearn.metrics import confusion_matrix,classification_report

Y_pred3 = googleNet_binary_model_ft.predict(testing_set1)
Y_pred3 = np.argmax(Y_pred3, axis=1)


# In[32]:


Y_pred3


# In[33]:



print('Classification Report')
print(classification_report(testing_set1.classes, Y_pred3))


# In[34]:


cm = confusion_matrix(testing_set1.classes, Y_pred3)
print(cm)


# In[35]:


from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_fscore_support

score = accuracy_score(testing_set.classes, Y_pred2)  

print('Accuracy: %.4f' % (accuracy_score(testing_set1.classes, Y_pred3)))
print('Recall ( Sen): %.4f' % (recall_score(testing_set1.classes, Y_pred3,average = 'macro')))
print('Precision (Spe): %.4f' % (precision_score(testing_set1.classes, Y_pred3,average = 'macro')))
print('F1 score : %.4f' % (f1_score(testing_set1.classes, Y_pred3, average='macro')))


# In[36]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, _ = roc_curve(testing_set1.classes, Y_pred3)
print('AUC : %.4f' % (auc(fpr, tpr)))


# In[ ]:





# In[ ]:





# In[ ]:




