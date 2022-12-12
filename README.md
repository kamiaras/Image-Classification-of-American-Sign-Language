# Image Classification of American Sign Language

Final project for:
- Course title: **EE 541 – Computational Introduction to Deep Learning**
- University: **University of Southern California (USC)**
- Instructor: **Dr. Brandon Franzke**
- Semester: **Fall 2022**
- Student Team: [**Kamiar Asgari**](https://github.com/kamiarasgari) and [**Mohammadmahdi Sajedi**]()

# Required Packages
- [PyTorch](https://pytorch.org/) 
- [Torch-Summary](https://pypi.org/project/torch-summary/)
- [Matplotlib](https://matplotlib.org/)
- [Pandas](https://pandas.pydata.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [Numpy](https://numpy.org/)

# Table Of Contents
- [Dataset](#Dataset) 
- [Data Handling](#Data-Handling)
- [Model](#Model)
- [Training Parameters](#Training-Parameters)
- [Results](#Results)
- [#Acknowledgement](#Acknowledgement)
- [#License](#License)


# Dataset
The dataset used for this project was created by [Akash Nagaraj](https://github.com/grassknoted). It is available on Kaggle as the [ASL Alphabet](https://www.kaggle.com/grassknoted/asl-alphabet) Dataset.

## About the Dataset
The data set is a collection of images of alphabets from the American Sign Language, separated in 29 folders which represent the various classes.
- The training data set contains 87,000 images which are 200x200 pixels. There are 29 classes, of which 26 are for the letters *"A"* through *"Z"* and 3 classes for *"SPACE"*, *"DELETE"* and *"NOTHIN"*.
- The test data set contains a mere 29 images, to encourage the use of real-world test images.

# Data Handling
We used a generic data loader called `ImageFolder` imported from `torchvision` library. It works well when the images are arranged similarly to our data-set.
```python
from torchvision import datasets
data_set = datasets.ImageFolder(<PATH to DATA FOLDER>, transform=transforms.ToTensor())
```
Using `torch.utils.data.random_split`, the data-set is divided into three parts: *training*, *testing*, and *validation*, with ratios of 80%, 10%, and 10%, respectively.
```python
train_test_val_split = [0.8, 0.1, 0.1]
train_set, test_set, val_set = torch.utils.data.random_split(data_set,
                                [round(p * len(data_set)) for p in train_test_val_split],
                                generator=torch.Generator().manual_seed(42))
```
We fixed the generator for reproducible results `generator=torch.Generator().manual_seed(42)` (*here 42 is an arbitrary number*). After spliting the data-set, corresponding data-loaders were created with `torch.utils.data.DataLoader` and a bach size of `batch_size=64`.

## Optinal Garbage Collection
Using `gc` library, it is possible to remove unwanted variable which are stored in RAM. For example after creating the data-loaders, we can remove the following variables: `data_set`, `train_set`, `test_set`, and `val_set`.
```python
import gc
del data_set
del train_set
del test_set
del val_set
gc.collect()
```

# Model
We employed a Convolutional Neural Network (CNN) with three hidden layer. 
- ***Input layer:*** Image 200 (height) x 200 (width) x 3 (RGB channels) .
- ***Hidden layer #1:*** `Conv2d(kernel_size=(5,5),out_channels=32,in_channels=3)` &rarr; `ReLu` &rarr; `MaxPool2d(2,2)`.
- ***Hidden layer #2:*** `Conv2d(kernel_size=(3,3),out_channels=64,in_channels=32)` &rarr; `ReLu` &rarr; `MaxPool2d(2,2)`.
- &rarr; **`Flatten`** &rarr;
- ***Hidden layer #3:*** `Linear(147456,128)` &rarr; `ReLu`.
- ***Output layer:*** `Linear(128,29)` &rarr; `Softmax`.

However, we did not include the the last `Softmax` function in our model because the `CrossEntropyLoss` function implemented in `pytoch` will take it into account on its own during training. Also, once the network has been trained, we can simply use the `max` function because gradient is no longer required.

Using `Torch-Summary` we can generated a description of our model:
```python
from torchsummary import  summary
summary(model=model, input_size=(3,200,200), batch_size=batch_size)
````
```python
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
```

# Training Parameters
A *bach size* of `batch_size=64` is chosen (as mentioned in the section [Data Handling](#Data-Handling).) The selected *Loss function* is *Cross-Entropy Loss*, which is a common solution for multiclass classification problems.
```python
loss_func = nn.CrossEntropyLoss()
```
Stochastic Gradient Descent (SGD) was chosen as the optimizer algorithm with an initial *learning rate* of `lr=0.05` and a *momentum* of `momentum=0.9`.
```python
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
```
The learning rate scheduler modifies the *learning rate* by multiplying it by `gamma=0.7` after each epoch (`step size=1`).
```python
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)
```
We ran the algorithm for five *epochs* (`num_epochs=5`).

# Results
Our algorithm was tested on a `cuda` device, a *NVIDIA Quadro RTX 5000* graphics card. Over a *AMD RYZENTM 7 PRO 5850U PROCESSOR*, this improved performance by about 5 times, from 74 minutes to 15 minutes.

## Final Model
Our final model gave us:
- Accuracy in the training data-set: 99.64%. 
- Accuracy in the validation data-set: 97.41%.
- Accuracy in the testing data-set: 97.56%. 

For the testing data-set, the confusion matrix was computed. For this matrix, a heatmap [heatmap](https://github.com/kamiarasgari/Image_Classification_of_American_Sign_Language/blob/b36871d739b255ba70ce19865692186487ace32a/Full_Confusion_Matrix.png) was created. However, the algorithm's high accuracy renders it useless. So we created another [heatmap](https://github.com/kamiarasgari/Image_Classification_of_American_Sign_Language/blob/b36871d739b255ba70ce19865692186487ace32a/diog_zeroed_conf_matr.png) for a new matrix that was identical to the original confusion matrix except for the diogonal elements, which were set to zero.

## Training time
The *Loss* and the *Accuracy* of each 50 iteration (for 50 batches)  were averaged and plotted in the following graph: [Training_Time_Loss](https://github.com/kamiarasgari/Image_Classification_of_American_Sign_Language/blob/0a2c5c0f7f1b8a859b660369112eb973282cb5f9/Training_Time_Loss.png) and [Training_Time_Accuracy](https://github.com/kamiarasgari/Image_Classification_of_American_Sign_Language/blob/0a2c5c0f7f1b8a859b660369112eb973282cb5f9/Training_Time_Accuracy.png) respectively.

# Acknowledgements
We'd like to thank [Zalan Fabian](https://z-fabian.github.io/) for his help and advice.

# License

[MIT](https://choosealicense.com/licenses/mit/)



