# -*- coding: utf-8 -*-
"""Untitled4.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1h_3LFpu5_ePzi_7VI32ICzyHsXskUbNh
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import numpy as np
from PIL import Image
import cv2
import pickle
import csv

# Defining function for loading dataset from 'pickle' file
def load_rgb_data(file):
    # Opening 'pickle' file and getting images
    with open(file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')  # dictionary type, we use 'latin1' for python3
        # At the same time method 'astype()' used for converting ndarray from int to float
        # It is needed to divide float by float when applying Normalization
        x = d['features']   # 4D numpy.ndarray type, for train = (34799, 32, 32, 3)
        y = d['labels']                        # 1D numpy.ndarray type, for train = (34799,)
        s = d['sizes']                         # 2D numpy.ndarray type, for train = (34799, 2)
        c = d['coords']                        # 2D numpy.ndarray type, for train = (34799, 4)
        """
        Data is a dictionary with four keys:
            'features' - is a 4D array with raw pixel data of the traffic sign images,
                         (number of examples, width, height, channels).
            'labels'   - is a 1D array containing the label id of the traffic sign image,
                         file label_names.csv contains id -> name mappings.
            'sizes'    - is a 2D array containing arrays (width, height),
                         representing the original width and height of the image.
            'coords'   - is a 2D array containing arrays (x1, y1, x2, y2),
                         representing coordinates of a bounding frame around the image.
        """
        # Returning ready data
    return x, y, s, c

train_f, train_l, train_s, train_c = load_rgb_data("/content/train.p")
test_f, test_l, test_s, test_c = load_rgb_data("/content/test.p")
valid_f, valid_l, valid_s, valid_c = load_rgb_data("/content/valid.p")

# Defining function for getting texts for every class - labels
def label_text(file):
    # Defining list for saving label in order from 0 to 42
    label_list = []

    # Opening 'csv' file and getting image's labels
    with open(file, 'r') as f:
        reader = csv.reader(f)
        # Going through all rows
        for row in reader:
            # Adding from every row second column with name of the label
            label_list.append(row[1])
        # Deleting the first element of list because it is the name of the column
        del label_list[0]
    # Returning resulted list
    return label_list

labels = label_text("/content/signname.csv")

# Getting no. of Unique elements with their no. of occurances and indices in the original array
classes, class_indices, class_counts  = np.unique(train_l, return_index=True, return_counts=True)

def plot_images(images, labels):
    plt.figure(figsize = (14, 14))

    for c, c_i, c_count in zip(classes, class_indices, class_counts):
        print(f"Class Name: {labels[c]}")
        fig = plt.figure(figsize = (18, 1))

        for i in range(15):
            plt.subplot(1,15, i+1)
            plt.imshow(images[np.random.randint(c_i, c_i + c_count, 15)[i],:,:,:])
            plt.axis("off")
        plt.show()

plot_images(train_f, labels)

plt.figure(figsize = (20,20))
plt.bar(classes, class_counts)

plt.title('Number of Occurances for each Class', fontsize=25)
plt.xlabel('Classes', fontsize=22)
plt.ylabel('Number of Occurances', fontsize=22)
plt.xticks(classes, labels , minor = True, rotation = 90)
plt.show()

"""Data Augmentation
The categorical_crossentropy loss function expects the target labels to be one-hot encoded (i.e., having the same shape as the network's output) so we use the to_categorical library
"""

# Convert labels to one-hot encoding
train_l = to_categorical(train_l, num_classes=43)
valid_l = to_categorical(valid_l, num_classes=43)
test_l = to_categorical(test_l, num_classes=43)

image_generator = ImageDataGenerator(rescale = 1/255)

train_data = image_generator.flow(train_f,train_l,
                                  batch_size = 43)

valid_data = image_generator.flow(valid_f, valid_l,
                                  batch_size = 43)

test_data = image_generator.flow(test_f,test_l,
                                  batch_size = 43)

"""**CNN Model Building**"""

model = Sequential()

## Add layers to cnn model

# INPUT AND HIDDEN LAYERS

# Convolutional Layer
model.add(Conv2D(filters = 43,
                 kernel_size = 5,
                 padding = "same",
                 activation = "relu",
                 input_shape = [32, 32, 3])
         )

# Pooling Layer
model.add(MaxPooling2D(pool_size = (2,2)))

# Convolutional Layer
model.add(Conv2D(filters = 64,
                 kernel_size = 3,
                 padding = "same",
                 activation = "relu",)
         )

# Pooling Layer
model.add(MaxPooling2D())

# Convolutional Layer
model.add(Conv2D(filters = 128,
                 kernel_size = 3,
                 padding = "same",
                 activation = "relu",)
         )

# Pooling Layer
model.add(MaxPooling2D())

# Dropout Layer to prevent overfitting
model.add(Dropout(0.1))


# CLASSIFICATION

# Flatten Layer
model.add(Flatten())

# Fully Connected Layer
model.add(Dense(128, activation = "relu"))
model.add(Dense(128, activation = "relu"))

# Output Layer
model.add(Dense(43, activation = "softmax"))

"""**Model Summary**"""

model.summary()

""" Total params: 386,287 (1.47 MB)
 Trainable params: 386,287 (1.47 MB)
 Non-trainable params: 0 (0.00 B)
"""

model.compile(optimizer = "adam",
             loss = "categorical_crossentropy",
             metrics = ["accuracy"])

# Define the Early Stopping Callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # The metric to monitor
    patience=5,          # Number of epochs with no improvement after which training will be stopped
    verbose=1,           # Verbosity mode, 1 = progress bar
    restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
)

model_history = model.fit(x = train_data,
                         epochs = 20,
                         validation_data = valid_data,
                         callbacks = [early_stopping])

"""Model Visualization"""

plt.plot(model_history.history["loss"], label = "Train Loss")
plt.plot(model_history.history["val_loss"], label = "Validation Loss")
plt.legend()
plt.show()

plt.plot(model_history.history["accuracy"], label = "Train Accuracy")
plt.plot(model_history.history["val_accuracy"], label = "Validation Accuracy")
plt.legend()
plt.show()

model.evaluate(test_data)

model.evaluate(train_data)

def prediction_with_classification_report(test_file, model):
    x_test, y_test, _, _ = load_rgb_data(test_file)

    # Define the data generator with shuffling enabled
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow(x_test, y_test, batch_size=43, shuffle=True)

    # Collect predictions and corresponding true labels
    y_pred = []
    y_true = []

    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        batch_pred = model.predict(batch_x)
        batch_pred_classes = np.argmax(batch_pred, axis=1)

        y_pred.extend(batch_pred_classes)
        y_true.extend(batch_y)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Generate classification report
    print(classification_report(y_true, y_pred, target_names=labels))

prediction_with_classification_report("/content/test.p", model)

def prediction_with_confusion_matrix(test_file, model):
    x_test, y_test, _, _ = load_rgb_data(test_file)

    # Define the data generator with shuffling enabled
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow(x_test, y_test, batch_size=43, shuffle=True)

    # Collect predictions and corresponding true labels
    y_pred = []
    y_true = []

    for i in range(len(generator)):
        batch_x, batch_y = generator[i]
        batch_pred = model.predict(batch_x)
        batch_pred_classes = np.argmax(batch_pred, axis=1)

        y_pred.extend(batch_pred_classes)
        y_true.extend(batch_y)

    y_pred = np.array(y_pred)
    y_true = np.array(y_true)

    # Generate classification report
    unique_labels = np.unique(y_true)
    return confusion_matrix(y_true, y_pred, labels=unique_labels)

cmt = prediction_with_confusion_matrix("/content/test.p", model)

plt.figure(figsize=(17, 17))
sns.heatmap(cmt, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

"""Predict New Images"""

def prediction(test_file):

     with open(test_file, 'rb') as f:
        d = pickle.load(f, encoding='latin1')
        x = d['features']
        y = d['labels']

        classes_test, class_indices_test, class_counts_test  = np.unique(y, return_index=True, return_counts=True)
        # Visulize the Predicited Image
        for c, c_i, c_count in zip(classes_test, class_indices_test, class_counts_test):
            fig = plt.figure(figsize = (43, 43))

            plt.subplot(43,1, c+1)
            idx = np.random.randint(c_i, c_i + c_count)
            img = x[idx]

            # Rescale image as done in the image generator
            img_rescaled = img.astype('float32') / 255

            prediction = model.predict(np.expand_dims(img_rescaled, axis=0))
            predicted_class = np.argmax(prediction, axis=1)

            plt.imshow(img)
            plt.title(f"Predicted: {labels[predicted_class[0]]}")
            plt.axis("off")
            plt.show()

prediction("/content/test.p")

def predict_external_image(img_path, model, labels):
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (32, 32))

    img_rescaled = img_resized.astype('float32') / 255.0

    img_rescaled = np.expand_dims(img_rescaled, axis=0)

    prediction = model.predict(img_rescaled)

    predicted_class = np.argmax(prediction, axis=1)

    print(f"Predicted Traffic Sign: {labels[predicted_class[0]]}")


    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Predicted: {labels[predicted_class[0]]}")
    plt.axis('off')
    plt.show()

image_path = '/content/acf6a2ba22333e440ecc174083dc0c64.jpg'
predict_external_image(image_path, model, labels)