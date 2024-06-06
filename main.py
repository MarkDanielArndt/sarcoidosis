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
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
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
from sklearn.metrics import roc_curve, auc
#from wandb.keras import WandbMetricsLogger
from utils import calculate_grad_cam
from tensorflow.keras.applications import Xception, VGG16, VGG19, ResNet50V2, ResNet101V2, ResNet152V2, \
                                    InceptionResNetV2, MobileNetV2, DenseNet121, DenseNet169, DenseNet201, \
                                    EfficientNetB0, EfficientNetB4, EfficientNetB7, NASNetLarge
#from keras_tuner.tuners import RandomSearch



# Definition of the number of folds for cross-validation
n_splits = config.n_splits
input_shape = (240, 180, 3)  # PyTorch convention: (channels, height, width)



list_pretrained = [VGG16, VGG19, ResNet50V2, ResNet101V2, InceptionResNetV2, 
MobileNetV2, DenseNet121, DenseNet169, DenseNet201]

#list_pretrained = [Xception, ResNet50V2, VGG16, VGG19]



for starting_model in list_pretrained:
    accuracy_scores = []
    f1_scores = []
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

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

        if config.ensemble:
            X_train1, X_test1, y_train1, y_test1 = train_test_split(features_array[0::3,:,:,:], labels_array[0::3], test_size=0.3, stratify=labels_array[0::3], random_state=split) #, random_state=split
            X_val1, X_test1, y_val1, y_test1 = train_test_split(X_test1, y_test1, test_size=0.95, stratify=y_test1, random_state=split) #, random_state=split
            
            X_train2, X_test2, y_train2, y_test2 = train_test_split(features_array[1::3,:,:,:], labels_array[1::3], test_size=0.3, stratify=labels_array[1::3], random_state=split) #, random_state=split
            X_val2, X_test2, y_val2, y_test2 = train_test_split(X_test2, y_test2, test_size=0.95, stratify=y_test2, random_state=split) #, random_state=split
            
            X_train3, X_test3, y_train3, y_test3 = train_test_split(features_array[2::3,:,:,:], labels_array[2::3], test_size=0.3, stratify=labels_array[2::3], random_state=split) #, random_state=split
            X_val3, X_test3, y_val3, y_test3 = train_test_split(X_test3, y_test3, test_size=0.95, stratify=y_test3, random_state=split) #, random_state=split
            
        else:
            X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.3, stratify=labels_array, random_state=split) #, random_state=split
            X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.95, stratify=y_test, random_state=split) #, random_state=split


        #starting_model = VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
        
        #starting_model = tf.keras.models.load_model(config.checkpoint_path / "vgg16.h5")

    
        # Imposta i layer come non allenabili
        for layer in starting_model.layers:
            layer.trainable = False

        # tuner = RandomSearch(
        # build_model,
        # objective='val_accuracy',
        # max_trials=10,
        # executions_per_trial=3,
        # directory= config.image_path,
        # project_name='image_classification_tuning'
        # )
        # tuner.search(X_train, y_train, epochs=24, validation_data=(X_val, y_val), callbacks=[WandbMetricsLogger()])

        # # Get the optimal hyperparameters
        # best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

        # Print the optimal hyperparameters
        # print(f"""
        # The optimal number of units in the first dense layer is {best_hps.get('units')}.
        # The optimal regularizer is {best_hps.get('regularizer')}.
        # The optimal dropout rate is {best_hps.get('dropout')}.
        # The optimal learning rate is {best_hps.get('learning_rate')}.
        # """)

        # Build the model with the optimal hyperparameters and train it
        #model = tuner.hypermodel.build(best_hps)

        if config.ensemble:
            X_train1, X_val1, X_test1 = X_train1 / 255.0, X_val1 / 255.0, X_test1 / 255.0
            X_train2, X_val2, X_test2 = X_train2 / 255.0, X_val2 / 255.0, X_test2 / 255.0
            X_train3, X_val3, X_test3 = X_train3 / 255.0, X_val3 / 255.0, X_test3 / 255.0
        else:
            X_train = X_train / 255.0
            X_val = X_val / 255.0
            X_test = X_test / 255.0	
            train_generator = train_datagen.flow(X_train, y_train, batch_size=32)
  
        if config.ensemble:
            train_generator1 = train_datagen.flow(X_train1, y_train1, batch_size=32)
            train_generator2 = train_datagen.flow(X_train2, y_train2, batch_size=32)
            train_generator3 = train_datagen.flow(X_train3, y_train3, batch_size=32)

        #history = model.fit(train_generator, epochs=config.num_epochs, validation_data=(X_val, y_val), callbacks=[WandbMetricsLogger()])

        # Evaluate the model on the test data
        #test_loss, test_accuracy, _ = model.evaluate(X_test, y_test)
        #print("Test Accuracy:", test_accuracy)
        # Create a new sequential pattern
        #x = Conv2D(512, (3, 3), activation='relu')(starting_model.output)
        #x = GlobalAveragePooling2D()(starting_model.output) #-> Try this instead of Flatten
        x = Flatten()(starting_model.output)
        x = Dense(config.num_nodes, activation='relu', kernel_regularizer=l1(0.001))(x) #try l2 regularization and SiLu
        x = Dropout(0.3)(x)  #tf.keras.activations.silu for activation
        predictions = Dense(1, activation='sigmoid')(x)
        model = Model(inputs=starting_model.input, outputs=predictions)
        #model.summary()

        if config.ensemble:
            x1 = Flatten()(starting_model.output)
            x1 = Dense(config.num_nodes, activation='relu', kernel_regularizer=l1(0.001))(x1) #try l2 regularization and SiLu
            x1 = Dropout(0.3)(x1)  #tf.keras.activations.silu for activation
            predictions1 = Dense(1, activation='sigmoid')(x1)
            model1 = Model(inputs=starting_model.input, outputs=predictions1)

            x2 = Flatten()(starting_model.output)
            x2 = Dense(config.num_nodes, activation='relu', kernel_regularizer=l1(0.001))(x2) #try l2 regularization and SiLu
            x2 = Dropout(0.3)(x2)  #tf.keras.activations.silu for activation
            predictions2 = Dense(1, activation='sigmoid')(x2)
            model2 = Model(inputs=starting_model.input, outputs=predictions2)

            x3 = Flatten()(starting_model.output)
            x3 = Dense(config.num_nodes, activation='relu', kernel_regularizer=l1(0.001))(x3) #try l2 regularization and SiLu
            x3 = Dropout(0.3)(x3)  #tf.keras.activations.silu for activation
            predictions3 = Dense(1, activation='sigmoid')(x3)
            model3 = Model(inputs=starting_model.input, outputs=predictions3)

            model1.compile(optimizer=Adamax(learning_rate=config.learning_rate),
                loss='binary_crossentropy', metrics=['accuracy', 'mse'])

            model2.compile(optimizer=Adamax(learning_rate=config.learning_rate),
                loss='binary_crossentropy', metrics=['accuracy', 'mse'])

            model3.compile(optimizer=Adamax(learning_rate=config.learning_rate),
                loss='binary_crossentropy', metrics=['accuracy', 'mse'])
            
            history1 = model1.fit(train_generator1, epochs=config.num_epochs, validation_data=(X_val1, y_val1))
            history2 = model1.fit(train_generator2, epochs=config.num_epochs, validation_data=(X_val2, y_val2))
            history3 = model1.fit(train_generator3, epochs=config.num_epochs, validation_data=(X_val3, y_val3))

            test_predictions1 = model1.predict(X_test1)
            test_predictions1 = (test_predictions1 > 0.5).astype(int)
            test_predictions2 = model2.predict(X_test2)
            test_predictions2 = (test_predictions2 > 0.5).astype(int)
            test_predictions3 = model1.predict(X_test3)
            test_predictions3 = (test_predictions3 > 0.5).astype(int)

            test_predictions = (test_predictions1 + test_predictions2 + test_predictions3)/3.
            test_predictions = (test_predictions > 0.5).astype(int)
            
            acc = accuracy_score(y_test1, test_predictions)
            acc1 = accuracy_score(y_test1, test_predictions1)
            acc2 = accuracy_score(y_test1, test_predictions2)
            acc3 = accuracy_score(y_test1, test_predictions3)
            f1 = f1_score(y_test1, test_predictions)

            accuracy_scores.append(acc)
            f1_scores.append(f1)

            print("F1-score:", f1)
            print("Ensemble Test Accuracy:", acc)
            print("Slice 0 Test Accuracy:", acc1)
            print("Slice 3 Test Accuracy:", acc2)
            print("Slice 5 Test Accuracy:", acc3)

            y_prob1 = model1.predict(X_test1).ravel()
            y_prob2 = model1.predict(X_test2).ravel()
            y_prob3 = model1.predict(X_test3).ravel()

            # Step 6: Compute ROC curve and AUC
            fpr1, tpr1, thresholds1 = roc_curve(y_test1, y_prob1)
            fpr2, tpr2, thresholds2 = roc_curve(y_test2, y_prob2)
            fpr3, tpr3, thresholds3 = roc_curve(y_test3, y_prob3)

            tpr1_interp = np.interp(mean_fpr, fpr1, tpr1)
            tpr2_interp = np.interp(mean_fpr, fpr2, tpr2)
            tpr3_interp = np.interp(mean_fpr, fpr3, tpr3)

            mean_tpr = (tpr1_interp + tpr2_interp + tpr3_interp) / 3
            mean_auc = auc(mean_fpr, mean_tpr)

            tprs.append(mean_tpr)
            tprs[-1][0] = 0.0
            aucs.append(mean_auc)

        else:
            model.compile(optimizer=Adamax(learning_rate=config.learning_rate),
                        loss='binary_crossentropy', metrics=['accuracy', 'mse'])

            #wandb_callback = wandb.keras.WandbCallback(log_weights=False)

            history = model.fit(train_generator, epochs=config.num_epochs, validation_data=(X_val, y_val))

            test_loss, test_accuracy, _  = model.evaluate(X_test, y_test)
            print("Test Accuracy:", test_accuracy)

            test_predictions = model.predict(X_test)
            test_predictions = (test_predictions > 0.5).astype(int)
            f1 = f1_score(y_test, test_predictions)
            print("F1-score:", f1)

            accuracy_scores.append(test_accuracy)
            f1_scores.append(f1)

            y_prob = model.predict(X_test).ravel()

            # Step 6: Compute ROC curve and AUC
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)

            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc)

        if config.ensemble:           
            conf_matrix = confusion_matrix(y_test1, test_predictions)
            print("Confusion Matrix:")
            print(conf_matrix)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            plt.savefig(config.image_path / ("Confusion Matrix" + str(split)))
            plt.clf()

        else:
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

            conf_matrix = confusion_matrix(y_test, test_predictions)
            print("Confusion Matrix:")
            print(conf_matrix)
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted labels')
            plt.ylabel('True labels')
            plt.title('Confusion Matrix')
            plt.savefig(config.image_path / ("Confusion Matrix" + str(split)))
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

    print(f"Model {name} done with:")
    print("Average Accuracy:", np.median(accuracy_scores))
    print("Average F1-score:", np.median(f1_scores))
    print("Standard Deviation of Accuracy:", np.std(accuracy_scores))
    print("Standard Deviation of F1-score:", np.std(f1_scores))

    if config.ensemble:
        name = name + "-ensemble"
    content = f"Model: {name} {config.num_nodes}-nodes {n_splits}-fold slice:{config.slice}({config.num_epochs} epochs)\n" + \
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

    plt.figure()
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
            lw=2, alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', color='navy', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic with Cross-Validation')
    plt.legend(loc="lower right")
    plt.savefig(config.image_path / ("ROC_AUC" + str(split)))
    plt.clf()
