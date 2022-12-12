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
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# Dataset
The dataset used for this project was created by [Akash Nagaraj](https://github.com/grassknoted). It is available on Kaggle as the [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) Dataset.

## About the Dataset
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.
- The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters *"A"* through *"Z"* and 3 classes for *"SPACE"*, *"DELETE"* and *"NOTHIN"*.
- The test data set contains a mere 29 images, to encourage the use of real-world test images.

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
The optimazer algorithm was chosen to be Stochastic Gradient Descent (SGD) with a starting learing rate 
