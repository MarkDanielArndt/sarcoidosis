import torch 
from torch.utils.data import DataLoader, Dataset, Subset
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torchvision.transforms as transforms
import config
import random
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator


csv_file_path = config.csv_file_path
data_ = pd.read_csv(csv_file_path, sep=";")

features = data_[['Study']]  # Seleziona solo la colonna 'Study'
labels = data_['Systemic Sarcoidosis (POSITIVE = 1)']  # Utilizza solo la label per la sarcoidosi sistemica


folder_path = config.folder_path

# List to save data loaded from each numpy file
data_list = []
new_height = config.new_height 
new_width = config.new_width
crop_x_start, crop_x_end = config.crop_x_start, config.crop_x_end 
crop_y_start, crop_y_end = config.crop_y_start, config.crop_y_end 

def saturate_image(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_image[..., 1] *= np.random.uniform(0.5, 1.5)  
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)

def adjust_contrast(image, factor):
    return tf.image.adjust_contrast(image, factor)

def custom_contrast(image):
    factor = tf.random.uniform([], 0.4, 1)  
    return adjust_contrast(image, factor)

train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=custom_contrast,
    #preprocessing_function=saturate_image,

    #vertical_flip=True,
    #brightness_range=[0.2, 0.5]
        
        
    )

# Scan folder and load each numpy file
for filename in os.listdir(folder_path):
    if filename.endswith('.npy'):
        file_path = os.path.join(folder_path, filename)
        data = np.load(file_path)
        image=[]
        slice1=data[0]
        #slice2=data[3]
        #slice3=data[5]
        #slice4=data[7]

        # Crop before resizing        
        resized_slice1 = cv2.resize(slice1, (new_width, new_height))
        #resized_slice2 = cv2.resize(slice2, (new_width, new_height))
        #resized_slice3 = cv2.resize(slice3, (new_width, new_height))
        #resized_slice4 = cv2.resize(slice4, (new_width, new_height))

        # Crop to the resized image
        cropped_slice1 = resized_slice1[crop_x_start: crop_x_end, crop_y_start: crop_y_end]
        #cropped_slice2 = resized_slice2[50:370, :200]
        #cropped_slice3 = resized_slice3[50:370, :200]
        #cropped_slice4 = resized_slice4[50:370, :200]

        #cropped_slice=cropped_slice / 255.0
        #image.append(cropped_slice1)
        #image.append(cropped_slice2)
        #image.append(cropped_slice3)
        #image.append(cropped_slice4)
        study_id = int(filename.split('_')[0])

        # Initialize a list to store matching labels
        matching_labels = []

        # ID patient

        # Find the corresponding patient label in the DataFrame
        matching_label = data_[data_['Study'] == study_id]['Systemic Sarcoidosis (POSITIVE = 1)'].values
        if len(matching_label) > 0:
            matching_labels.append(matching_label[0])
        else:
            matching_labels.append(None)  
            
    	# Add the corresponding patient image and label to the data list
        data_list.append((np.array(cropped_slice1), matching_labels))

# Inizializza le liste per le features e le label
features_list = []
labels_list = []

# Estrai le features e le label da data_list
for item in data_list:
    features, label = item
    features_list.append(features)
    labels_list.append(label)

# Converti le liste in array numpy
features_array = np.array(features_list)
labels_array = np.array(labels_list)




# if __name__ == "__main__":
    # images, label = next(iter(train_loader))

    # print(np.shape(images))
    # plt.imshow(images[0])
    # plt.show()