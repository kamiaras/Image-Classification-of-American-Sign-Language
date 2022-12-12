# Image Classification of American Sign Language

Final project for:

- Course title: **EE 541 – Computational Introduction to Deep Learning**

- University: **University of Southern California (USC)**

- Instructor: **Dr. Brandon Franzke**

- Semester: **Fall 2022**

# Requirements
- [PyTorch](https://pytorch.org/) 
- [Torch-Summary](https://pypi.org/project/torch-summary/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Numpy](https://numpy.org/)

# Table Of Contents
- [Dataset](#Dataset)


# Dataset
The dataset used for this project was created by [Akash Nagaraj](https://github.com/grassknoted). It is available on Kaggle as the [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) Dataset.

## About the Dataset
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.
- The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters *"A"* through *"Z"* and 3 classes for *"SPACE"*, *"DELETE"* and *"NOTHIN"*.
- The test data set contains a mere 29 images, to encourage the use of real-world test images.

# Handaling the data
We used a generic data loader called `ImageFolder` imported from `torchvision` library. It works well when the images are arranged similarly to our data-set.
```python
from torchvision import datasets
data_set = datasets.ImageFolder(<PATH to DATA FOLDER>, transform=transforms.ToTensor())
```
The data-set is divided into three parts: *training*, *testing*, and *validation*, with ratios of 80%, 10%, and 10%, respectively.

# Model
We employed a convolutional neural network (CNN) with the following [Torch-Summary](https://pypi.org/project/torch-summary/)-generated description:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [64, 32, 196, 196]           2,432
         MaxPool2d-2           [64, 32, 98, 98]               0
            Conv2d-3           [64, 64, 96, 96]          18,496
         MaxPool2d-4           [64, 64, 48, 48]               0
            Linear-5                  [64, 128]      18,874,496
            Linear-6                   [64, 29]           3,741
================================================================
Total params: 18,899,165
Trainable params: 18,899,165
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 29.30
Forward/backward pass size (MB): 1110.39
Params size (MB): 72.09
Estimated Total Size (MB): 1211.78
----------------------------------------------------------------
```

The selected *Loss function* is *Cross-Entropy Loss*, which is a common solution for multiclass classification problems.
```python
loss_func = nn.CrossEntropyLoss()
```
Stochastic Gradient Descent (SGD) was chosen as the optimizer algorithm with an initial *learning rate* of 'lr=0.05' and a *momentum* of'momentum=0.9'.
```python
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
```
The learning rate scheduler modifies the 'learning rate' by multiplying it by 'gamma=0.7' after each epoch ('step size=1').
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
```
