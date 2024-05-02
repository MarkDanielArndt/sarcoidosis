import pandas as pd
import os 
import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import cv2
import config
import sac_dataset
from tqdm import tqdm
from sac_dataset import features_array, labels_array
from sklearn.model_selection import train_test_split
import wandb

def softmax(x):
    exp_x = np.exp(x) 
    return exp_x / exp_x.sum(axis=0)

# Definition of the number of folds for cross-validation
n_splits = config.n_splits
input_shape = (3, 240, 180)  # PyTorch convention: (channels, height, width)

accuracy_scores = []
f1_scores = []

for i in range(n_splits):
    print(f"Iteration {i + 1}:")

    # Creation of the complete model
    vgg16 = models.vgg16(weights="VGG16_Weights.DEFAULT")

    # Set parameters of pretrained layers to non-trainable
    for param in vgg16.parameters():
        param.requires_grad = False

    num_features = vgg16.classifier[-1].in_features
    vgg16.classifier[-1] = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, 1),
        nn.Sigmoid()
    )

    # Move model to device
    vgg16.to(config.device)

    X_train, X_test, y_train, y_test = train_test_split(features_array, labels_array, test_size=0.3, random_state=i, stratify=labels_array)

    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.30, random_state=i, stratify=y_test)

    train_dataset = sac_dataset.CustomImageDataset(X_train, y_train)

    batch_size = config.batchsize

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    X_val = torch.tensor(X_val, dtype=torch.float32).permute(0, 3, 1, 2) / 255
    X_test = torch.tensor(X_test, dtype=torch.float32).permute(0, 3, 1, 2) / 255

    criterion = nn.BCELoss()
    optimizer = optim.Adamax(vgg16.parameters(), lr=config.learning_rate)

    num_epochs = config.num_epochs
    train_loss_array = []
    train_accuracy_array = []

    for epoch in range(num_epochs):
        vgg16.train()
        running_loss = 0.0
        loop = tqdm(train_loader, leave=True)
        outputs_array = np.array([])
        for idx, (inputs, labels) in enumerate(loop):
            inputs, labels = inputs.to(config.device), labels.to(config.device)
            optimizer.zero_grad()
            outputs = vgg16(inputs)
            loss = criterion(outputs.squeeze(), labels.float().squeeze())
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            outputs_array = np.append(outputs_array, np.array(outputs.cpu().detach()))
        
        test_predictions = ((outputs_array) > 0.5).astype(int)

        train_accuracy = (test_predictions == y_train.reshape(-1, 1)).mean()

        train_loss = running_loss / len(train_loader.dataset)
        train_loss_array.append(train_loss)
        train_accuracy_array.append(train_accuracy)
        
        wandb.log({"train_loss": train_loss, "Train acc": train_accuracy} )
        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Train acc: {train_accuracy:.4f}")



    vgg16.eval()

    with torch.no_grad():
        outputs = vgg16(X_test.to(config.device))
        test_predictions = (outputs.cpu() > 0.5).numpy().astype(int)
        test_loss = criterion(outputs.squeeze(),
                               torch.tensor(y_test, dtype=torch.float32).squeeze()).item()

    test_accuracy = (test_predictions == y_test.reshape(-1, 1)).mean()
    print("Test Accuracy:", test_accuracy)

    f1 = f1_score(y_test, test_predictions)
    print("F1-score:", f1)

    accuracy_scores.append(test_accuracy)
    f1_scores.append(f1)

    # Plot Loss
    plt.plot(range(num_epochs), train_loss_array, label='Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    

    #Plot Accuracy
    plt.plot(range(num_epochs), train_accuracy_array, label='Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # GRAD-CAM
    # You need to implement this part using PyTorch's hooks and gradcam techniques

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, test_predictions)
    print("Confusion Matrix:")
    print(conf_matrix)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.title('Confusion Matrix')

    plt.show()

print("Average Accuracy:", np.mean(accuracy_scores))
print("Average F1-score:", np.mean(f1_scores))
print("Standard Deviation of Accuracy:", np.std(accuracy_scores))
print("Standard Deviation of F1-score:", np.std(f1_scores))

wandb.finish()

