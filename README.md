# Knee-Osteoarthritis-Detection-Using-YOLO and ResNext50 with Accuracy+93%
Knee Osteoarthritis Classification

![From Dataset](https://github.com/alirzx/Knee-Osteoarthritis-Detection-Using-YOLO/blob/main/KNEE.jpg?raw=true)


Knee Osteoarthritis Classification
This repository contains the code and resources for a classification model to predict Knee Osteoarthritis (KOA) from X-ray images. The goal of the project is to categorize knee X-ray images into three classes: Normal, Osteopenia, and Osteoporosis. The project utilizes multiple models including YOLOv8, VGG16, ResNeXt-50, and ResNeXt-101 for this classification task.

Table of Contents
Project Overview
Dataset
Model Architecture
Training Process
Performance
Installation
Usage
License
Project Overview
The project involves the classification of knee osteoarthritis based on X-ray images. These images are labeled into one of the three categories:

Normal: Healthy knee joints.
Osteopenia: Low bone density in the knee joints.
Osteoporosis: Severe loss of bone density.
The models used for this classification are:

YOLOv8: A state-of-the-art object detection model for detecting and classifying knee abnormalities.
VGG16: A deep convolutional neural network (CNN) for image classification.
ResNeXt-50: A more efficient version of ResNet for higher performance.
ResNeXt-101: A deeper version of ResNeXt for potentially higher accuracy.
The final model achieved 93% accuracy on the test set.

Dataset
The dataset used in this project is the Knee Osteoarthritis Classification dataset, which contains X-ray images of knee joints along with their labels. The dataset is divided into the following splits:

Training Set: 80% of the images used to train the models.
Validation Set: 10% of the images used for model tuning.
Test Set: 10% of the images used to evaluate model performance.
The images are organized into folders based on the class labels:

Normal
Osteopenia
Osteoporosis
Dataset Preprocessing
The preprocessing steps involve:

Resizing images to a fixed size (224x224 pixels) to feed them into the neural network.
Splitting the dataset into training, validation, and test sets.
Converting mask files to YOLO format for YOLO-based model training.
Model Architecture
1. YOLOv8
YOLOv8 (You Only Look Once version 8) is used for detecting the knee osteoarthritis classification. YOLO is a real-time object detection model, and we leveraged its ability to classify knee abnormalities based on segmentation maps of the X-ray images.

2. VGG16
VGG16 is a deep convolutional neural network that is widely used for image classification tasks. The architecture consists of 16 layers, with 13 convolutional layers and 3 fully connected layers. The final classification layer was replaced to match the 3 output classes (Normal, Osteopenia, Osteoporosis).

3. ResNeXt-50 & ResNeXt-101
ResNeXt is a variant of the ResNet architecture that uses grouped convolutions for improved performance. Both ResNeXt-50 and ResNeXt-101 were used to benchmark the classification task. These models are known for their efficiency and scalability, offering improved accuracy and faster convergence during training.

Training Process
Data Loading and Preprocessing
The dataset was downloaded and organized into three separate folders for training, validation, and testing.
Image paths and labels were collected into a pandas dataframe.
The images were resized to 224x224 pixels for compatibility with VGG16, ResNeXt, and YOLO models.
Model Fine-Tuning and Transfer Learning
For each model, pre-trained weights were loaded (where applicable), and the final classification layer was replaced to match the number of output classes (3 classes: Normal, Osteopenia, Osteoporosis). Models were fine-tuned on the dataset for better accuracy:

VGG16: Pre-trained on ImageNet, with the final classification layer replaced for 3 classes.
ResNeXt-50 & ResNeXt-101: Pre-trained on ImageNet, fine-tuned for KOA classification.
YOLOv8: Fine-tuned for segmentation tasks in the KOA dataset.
Training Setup
Optimizer: AdamW optimizer was used for its efficiency in training deep networks.
Learning Rate Scheduler: A learning rate scheduler (CosineAnnealingLR) was used to adjust the learning rate during training.
Loss Function: CrossEntropyLoss was used for multi-class classification tasks.
Gradient Scaling: Mixed precision training was applied for improved speed and stability.
Hyperparameters
Learning Rate: 1e-4
Weight Decay: 1e-5
Epochs: 50
Performance
After training and evaluation, the model achieved 93% accuracy on the test set. The results were evaluated on the following metrics:

Accuracy: 93%
Precision: High for all classes.
Recall: Excellent recall, ensuring few false negatives.
Model Comparison
YOLOv8: Achieved good performance for detecting class labels and handling segmentation tasks.
VGG16: Strong results for image classification.
ResNeXt-50 and ResNeXt-101: Best performance in terms of accuracy, especially the ResNeXt-101 model, due to its deeper architecture.
Installation
To set up and run this project, follow these steps:

Clone the repository:

bash
Copy
Edit
git clone https://github.com/yourusername/knee-osteoarthritis-classification.git
cd knee-osteoarthritis-classification
Install dependencies:

bash
Copy
Edit
pip install -r requirements.txt
Setup Google Drive (if using Google Colab):

Mount your Google Drive and ensure the dataset is placed in the correct directory.
Download the dataset: You can download the dataset directly from Kaggle or any provided links.

Usage
To train and evaluate the model:

Open the Jupyter notebook or Python script for training.
Make sure that your data is correctly placed in the designated folder.
Run the training code with the desired model.
The model will save checkpoints and the final model weights after training.
Evaluate the model using the test dataset.
Example:

python
Copy
Edit
# Train VGG16 model
train_model(model='vgg16')

# Evaluate on test set
evaluate_model(model='vgg16', dataset='test')
License
This project is licensed under the MIT License - see the LICENSE file for details.

Acknowledgments
Knee Osteoarthritis Dataset: Dataset used for training the models.
YOLOv8, VGG16, ResNeXt: Pretrained models from PyTorch and other sources used to fine-tune on the dataset.
