#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
# import cv2
from random import shuffle
from tensorflow.keras import layers
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import Sequence
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from collections import Counter
from tensorflow import keras
import efficientnet.tfkeras as efn
import swin_layers
import transformer_layers
from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D, Layer, Dropout, Flatten
import warnings
from tensorflow.keras.preprocessing.image import ImageDataGenerator

warnings.filterwarnings("ignore")


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


# In[4]:



abnormal_images_testing = os.listdir(TestImage + "/Abnormal")
normal_images_testing = os.listdir(TestImage + "/Normal")


print('No of abnormal images testing : ',len(abnormal_images_testing) )
print('No of normal images testing : ',len(normal_images_testing) )

Nos_Test =  len(abnormal_images_testing) + len(normal_images_testing)

print('No of testing images : ',Nos_Test )


# In[5]:



abnormal_images_val = os.listdir(ValImage + "/Abnormal")
normal_images_val = os.listdir(ValImage + "/Normal")


print('No of abnormal images val : ',len(abnormal_images_val) )
print('No of normal images val : ',len(normal_images_val) )

Nos_Val =  len(abnormal_images_val) + len(normal_images_val)

print('No of testing images : ',Nos_Val )


# In[6]:


image_size = 512
BATCH_SIZE = 4
#STEPS_PER_EPOCH = int(Nos_Train // BATCH_SIZE)


# In[7]:


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
    if subplot % 10 == 1:
        plt.subplots(figsize=(10, 10))
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.plot(training_loss)
    ax.plot(validation_loss)
    ax.set_title('Model ' + title)
    ax.set_ylabel(title)
    ax.set_xlabel('Epoch')
    ax.legend(['Train_Loss', 'Val_Loss'])


# In[9]:


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)

    # map the activations of the last conv layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, top_pred_index.numpy()


# In[10]:


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, classifier_layer_names):
    last_conv_layer_name = 'patch_merging'
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = keras.Model(model.inputs, last_conv_layer.output)
    classifier_layer_names = ['dense_head1', 'dense_1']

    # map the activations of the last conv layer to the final class predictions
    classifier_input = keras.Input(shape=last_conv_layer.output.shape[1:])
    x = classifier_input
    for layer_name in classifier_layer_names:
        x = model.get_layer(layer_name)(x)
    classifier_model = keras.Model(classifier_input, x)

    with tf.GradientTape() as tape:
        # Compute activations of the last conv layer and make the tape watch it
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_model(last_conv_layer_output)
        print(preds)
        top_pred_index = tf.argmax(preds[0])
        top_class_channel = preds[:, top_pred_index]

    grads = tape.gradient(top_class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output.numpy()[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(pooled_grads.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    heatmap = np.mean(last_conv_layer_output, axis=-1)
    heatmap = np.maximum(heatmap, 0) / np.max(heatmap)
    return heatmap, top_pred_index.numpy()


def superimposed_img(image, heatmap):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # We create an image with RGB colorized heatmap
    jet_heatmap = keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((image_size, image_size))
    jet_heatmap = keras.preprocessing.image.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * 0.4 + image
    superimposed_img = keras.preprocessing.image.array_to_img(superimposed_img)
    return superimposed_img


# label smoothing
def categorical_smooth_loss(y_true, y_pred, label_smoothing=0.1):
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred, label_smoothing=label_smoothing)
    return loss


# In[11]:


# training call backs
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, min_delta=0.0001, patience=10,
                                                 verbose=1)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, verbose=1)
counter = Counter(training_set.classes)
max_val = float(max(counter.values()))
class_weights = {class_id: max_val / num_images for class_id, num_images in counter.items()}


# In[12]:


# Lets build our Ensemble network -- this is the pretrained model
pretrained_DNet201 = tf.keras.applications.DenseNet201(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                 include_top=False)
pretrained_Xception = tf.keras.applications.Xception(input_shape=(image_size, image_size, 3), weights='imagenet', include_top=False)

pretrained_VGG = tf.keras.applications.VGG16(input_shape=(image_size, image_size, 3), weights='imagenet',include_top=False)

# pretrained_googleNet = tf.keras.applications.InceptionV3(input_shape=(image_size, image_size, 3), weights='imagenet',
#                                                         include_top=False)

pretrained_IRV2 = tf.keras.applications.InceptionResNetV2(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                          include_top=False)

pretrained_ResNet50 = tf.keras.applications.ResNet50(input_shape=(image_size, image_size, 3), weights='imagenet',
                                                          include_top=False)

# pretrained_Xception = tf.keras.applications.Xception(input_shape=(image_size, image_size, 3), weights='imagenet',
#                                                      include_top=False)


# In[14]:


visible = tf.keras.layers.Input(shape=(image_size, image_size, 3))


# In[17]:


patch_size = patch_size[0]
X = transformer_layers.patch_extract(patch_size)(part_A)
X = transformer_layers.patch_embedding(num_patch_x * num_patch_y, embed_dim)(X)


# In[19]:


X = transformer_layers.patch_merging((num_patch_x, num_patch_y), embed_dim=embed_dim, name='down{}'.format(i))(X)
X = GlobalAveragePooling1D()(X)
X = tf.keras.layers.Dense(1024, activation="relu", name="dense_head1")(X)
X = tf.keras.layers.Dropout(0.5, name="dropout_head1")(X)
OUT = Dense(2, activation='softmax')(X)


# In[20]:


model = Model(inputs=visible, outputs=OUT)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, clipvalue=0.2), loss=categorical_smooth_loss,
              metrics=['accuracy'])
model.summary()


# In[22]:


# history = model.fit(training_set, validation_data=testing_set, epochs=100)  # 30
history = model.fit(training_set, validation_data=validating_set, callbacks=[lr_reduce, es_callback], epochs=30)


# In[ ]:


display_training_curves2(history.history['loss'], history.history['val_loss'], 'loss', 211)
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 212)
plt.show()


# In[22]:



model.load_weights('model_Binary_Encoder3') 

model.evaluate(training_set)

model.evaluate(testing_set)

model.evaluate(validating_set)


# In[85]:


from sklearn.metrics import confusion_matrix,classification_report

Y_pred2 = model.predict(testing_set)
Y_pred2 = np.argmax(Y_pred2, axis=1)


# In[86]:


testing_set.classes


# In[87]:


Y_pred2


# In[88]:


print('Classification Report')
print(classification_report(testing_set.classes, Y_pred2))


# In[89]:


cm = confusion_matrix(testing_set.classes, Y_pred2)
print(cm)


# In[140]:


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


# In[141]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, _ = roc_curve(testing_set.classes, Y_pred2)
print('AUC : %.4f' % (auc(fpr, tpr)))


# In[142]:


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


# In[ ]:





# In[25]:



eval_path4='H:/PhD/prof ref/INbreast1/Ensemble binary/private eval encoder/'


# In[ ]:





# In[26]:



test_datagen1 = ImageDataGenerator()

testing_set1 = test_datagen1.flow_from_directory(eval_path4 ,
                                               target_size = (image_size, image_size),
                                               batch_size = BATCH_SIZE,
                                               class_mode = 'categorical', shuffle = False)


# In[152]:


from sklearn.metrics import confusion_matrix,classification_report

Y_pred3 = model.predict(testing_set1)
Y_pred3 = np.argmax(Y_pred3, axis=1)


# In[153]:


testing_set1.classes


# In[154]:


Y_pred3


# In[155]:



print('Classification Report')
print(classification_report(testing_set1.classes, Y_pred3))


# In[156]:


cm2 = confusion_matrix(testing_set1.classes, Y_pred3)
print(cm2)


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


# In[158]:


from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score,precision_recall_fscore_support
score = accuracy_score(testing_set.classes, Y_pred2)  

print('Accuracy: %.4f' % (accuracy_score(testing_set1.classes, Y_pred3)))
print('Recall ( Sen): %.4f' % (recall_score(testing_set1.classes, Y_pred3,average = 'macro')))
print('Precision (Spe): %.4f' % (precision_score(testing_set1.classes, Y_pred3,average = 'macro')))
print('F1 score : %.4f' % (f1_score(testing_set1.classes, Y_pred3, average='macro')))


# In[159]:


from sklearn.metrics import roc_curve
from sklearn.metrics import auc

fpr, tpr, _ = roc_curve(testing_set1.classes, Y_pred3)
print('AUC : %.4f' % (auc(fpr, tpr)))


# In[ ]:




