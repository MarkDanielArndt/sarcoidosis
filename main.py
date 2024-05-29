import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator#
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
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Flatten, Dense, Dropout, Concatenate, Input
from tensorflow.keras.models import Model
from sklearn.metrics import confusion_matrix, f1_score
import seaborn as sns
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split, StratifiedKFold
import cv2 
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import config
from sac_dataset import features_array, labels_array, train_datagen
import wandb
from wandb.keras import WandbMetricsLogger
from utils import calculate_grad_cam
from tensorflow.keras.applications import Xception, VGG16, VGG19, ResNet50V2, ResNet101V2, ResNet152V2, \
                                    InceptionResNetV2, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, \
                                    EfficientNetB0, EfficientNetB4, EfficientNetB7, NASNetLarge
from keras_tuner.tuners import RandomSearch



# Definition of the number of folds for cross-validation
n_splits = config.n_splits
input_shape = (240, 180, 3)  # PyTorch convention: (channels, height, width)



list_pretrained = [VGG16, VGG19, ResNet50V2, ResNet101V2, InceptionResNetV2, 
MobileNetV2, DenseNet121, DenseNet169, DenseNet201]

#list_pretrained = [Xception, ResNet50V2, VGG16, VGG19]



for starting_model in list_pretrained:
    accuracy_scores = []
    f1_scores = []

    name = starting_model.__name__

    starting_model = starting_model(
    include_top=False,
    weights='imagenet',
    input_shape=input_shape,
    )

    def build_model(hp):
        x = Flatten()(starting_model.output)
        x = Dense(
            units=256,
            activation=hp.Choice("activation", ["silu", "relu"]),
            kernel_regularizer=l2(0.05)
        )(x)
        x = Dropout(hp.Float('dropout', min_value=0.4, max_value=0.5, step=0.1))(x)
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=starting_model.input, outputs=predictions)

        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='binary_crossentropy',
            metrics=['accuracy', 'mse']
        )
        return model

    print(f"Model: {name}")

    for split in range(n_splits):
        print(f"Iteration {split + 1}:")

        X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.3, stratify=labels_array) #, random_state=split

        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.30, stratify=y_test) #, random_state=split


        #starting_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        #starting_model = tf.keras.models.load_model(config.checkpoint_path / "vgg16.h5")

    
        # Imposta i layer come non allenabili
        for layer in starting_model.layers:
            layer.trainable = False

        tuner = RandomSearch(
        build_model,
        objective='val_accuracy',
        max_trials=10,
        executions_per_trial=3,
        directory= config.image_path,
        project_name='image_classification_tuning'
        )
        tuner.search(X_train, y_train, epochs=24, validation_data=(X_val, y_val), callbacks=[WandbMetricsLogger()])

        # Get the optimal hyperparameters
        best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Print the optimal hyperparameters
        # print(f"""
        # The optimal number of units in the first dense layer is {best_hps.get('units')}.
        # The optimal regularizer is {best_hps.get('regularizer')}.
        # The optimal dropout rate is {best_hps.get('dropout')}.
        # The optimal learning rate is {best_hps.get('learning_rate')}.
        # """)

        # Build the model with the optimal hyperparameters and train it
        model = tuner.hypermodel.build(best_hps)

        train_generator = train_datagen.flow(X_train, y_train, batch_size=32)

        history = model.fit(train_generator, epochs=config.num_epochs, validation_data=(X_val, y_val), callbacks=[WandbMetricsLogger()])

        # Evaluate the model on the test data
        test_loss, test_accuracy, _ = model.evaluate(X_test, y_test)
        print("Test Accuracy:", test_accuracy)
        # Create a new sequential pattern
        #x = Conv2D(512, (3, 3), activation='relu')(starting_model.output)
        #x = GlobalAveragePooling2D()(starting_model.output) #-> Try this instead of Flatten
        x = Flatten()(starting_model.output)
        x = Dense(512, activation='relu' ,kernel_regularizer=l1(0.001))(x) #try l2 regularization and SiLu
        x = Dropout(0.3)(x)  #tf.keras.activations.silu for activation
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=starting_model.input, outputs=predictions)
        #model.summary()

        X_train = X_train / 255.0
        X_val = X_val / 255.0
        X_test = X_test / 255.0


        model.compile(optimizer=Adamax(learning_rate=config.learning_rate),
                    loss='binary_crossentropy', metrics=['accuracy', 'mse'])

        wandb_callback = wandb.keras.WandbCallback(log_weights=False)

        history = model.fit(train_generator, epochs=config.num_epochs, validation_data=(X_val, y_val), callbacks=[WandbMetricsLogger()])

        test_loss, test_accuracy, _  = model.evaluate(X_test, y_test)
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
        plt.savefig(config.image_path / ("Loss" + str(split)))
        plt.clf()

        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig(config.image_path / ("Accuracy" + str(split)))
        plt.clf()


        # GRAD-CAM
        # conv_layer = 'conv5_block3_3_conv'
        # num_images_to_display = min(len(X_test), 30)  
        # rows = int(np.ceil(num_images_to_display / 5))  

        # plt.figure(figsize=(15, 3 * rows))  
        # for i in range(num_images_to_display):
        #     cam = calculate_grad_cam(model, conv_layer, X_test[i])
        #     plt.subplot(rows, 5, i+1)
        #     plt.imshow(X_test[i])
        #     plt.imshow(cam, alpha=0.5, cmap='jet')
        #     title = "Correct" if test_predictions[i] == y_test[i] else "Incorrect"
        #     title += f" (Class {y_test[i]})"  
        #     plt.title(title)
        #     plt.axis('off')
        # plt.tight_layout()  
        # plt.savefig(config.image_path / ("grad_cam" + str(split)))
        # plt.clf()

        # Confusion Matrix
        conf_matrix = confusion_matrix(y_test, test_predictions)
        print("Confusion Matrix:")
        print(conf_matrix)
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title('Confusion Matrix')
        plt.savefig(config.image_path / ("Confusion Matrix" + str(split)))
        plt.clf()

    print(f"Model {name} done with:")
    print("Average Accuracy:", np.median(accuracy_scores))
    print("Average F1-score:", np.median(f1_scores))
    print("Standard Deviation of Accuracy:", np.std(accuracy_scores))
    print("Standard Deviation of F1-score:", np.std(f1_scores))

    content = f"Model: {name} {n_splits}-fold ({config.num_epochs} epochs)\n" + \
    f"Average Accuracy: {np.median(accuracy_scores)}\n" + \
    f"Average F1-score: {np.median(f1_scores)}\n" + \
    f"Standard Deviation of Accuracy: {np.std(accuracy_scores)}\n" + \
    f"Standard Deviation of F1-score: {np.std(f1_scores)}\n \n"

    # Specify the file path
    file_path = config.image_path / "evaluation_results.txt"

    # Write content to the text file
    with open(file_path, "a") as file:
        file.write(content)

    print("Content has been written to", file_path)