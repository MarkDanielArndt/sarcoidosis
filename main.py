import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adamax
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten, Conv3D, MaxPooling3D, Dropout
from tensorflow.keras.models import Sequential
import os, os.path
from tensorflow import keras
import matplotlib.pyplot as plt
import PIL
from tensorflow.keras.layers import Dropout
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import datasets, layers, models
from os import walk
import matplotlib.image as mpimg
from tensorflow.keras.preprocessing import image
import pathlib
import glob
#import splitfolders
import random
import shutil
import seaborn as sns
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix
from tensorflow.keras.utils import to_categorical
import SimpleITK as sitk
from sklearn.model_selection import train_test_split, KFold
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2 
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from sac_dataset import features_array, labels_array, train_datagen
import wandb
from wandb.keras import WandbMetricsLogger
from utils import calculate_grad_cam



# Definition of the number of folds for cross-validation
n_splits = config.n_splits
input_shape = (240, 180, 3)  # PyTorch convention: (channels, height, width)

accuracy_scores = []
f1_scores = []


for i in range(n_splits):
    print(f"Iteration {i + 1}:")

    X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.3, random_state=i, stratify=labels_array)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.30, random_state=i, stratify=y_test)


    vgg16 = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)

    # Imposta i layer come non allenabili
    for layer in vgg16.layers:
        layer.trainable = False

   # Create a new sequential pattern
    x = Flatten()(vgg16.output)
    x = Dense(512, activation='relu' ,kernel_regularizer=l1(0.001))(x)
    x = Dropout(0.3)(x)  # Aggiungi un layer di Dropout con una probabilità del 50%
    predictions = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=vgg16.input, outputs=predictions)
    model.summary()

    X_train = X_train / 255
    X_val = X_val / 255
    X_test = X_test / 255

    train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

    model.compile(optimizer=Adamax(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

    wandb_callback = wandb.keras.WandbCallback(log_weights=False)

    history = model.fit(train_generator, epochs=config.num_epochs, validation_data=(X_val, y_val), callbacks=[WandbMetricsLogger()])

    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print("Test Accuracy:", test_accuracy)

    test_predictions = model.predict(X_test)
    test_predictions = (test_predictions > 0.5).astype(int)
    f1 = f1_score(y_test, test_predictions)
    print("F1-score:", f1)

    accuracy_scores.append(test_accuracy)
    f1_scores.append(f1)

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(config.image_path / ("Loss" + str(i)))
    plt.clf()

    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(config.image_path / ("Accuracy" + str(i)))
    plt.clf()


    # GRAD-CAM
    conv_layer = 'block5_conv3'
    num_images_to_display = min(len(X_test), 30)  
    rows = int(np.ceil(num_images_to_display / 5))  

    plt.figure(figsize=(15, 3 * rows))  
    for i in range(num_images_to_display):
        cam = calculate_grad_cam(model, conv_layer, X_test[i])
        plt.subplot(rows, 5, i+1)
        plt.imshow(X_test[i])
        plt.imshow(cam, alpha=0.5, cmap='jet')
        title = "Correct" if test_predictions[i] == y_test[i] else "Incorrect"
        title += f" (Class {y_test[i]})"  
        plt.title(title)
        plt.axis('off')
    plt.tight_layout()  
    plt.savefig(config.image_path / ("grad_cam" + str(i)))
    plt.clf()

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, test_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')
    plt.savefig(config.image_path / ("Confusion Matrix" + str(i)))
    plt.clf()


print("Average Accuracy:", np.median(accuracy_scores))
print("Average F1-score:", np.median(f1_scores))
print("Standard Deviation of Accuracy:", np.std(accuracy_scores))
print("Standard Deviation of F1-score:", np.std(f1_scores))