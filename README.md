# Glare Detection in Photographs using Transfer learning

This was a project given to me by Abyss Solutions as an offline coding test, the full details can be found in the pdf. All the items required for submission may be found in this repository, and the write-up discussing the algorithm and future ideas will be inlcuded in this readme. For a full view of the process of building the model I recomend you check out the IPython notebook Transfer_learning.ipynb 

## Introduction

Abyss Solutions is a robotics company specialising in using remotely operated underwater vehicles to inspect underwater assets. However an issue that they have encountered is the corruption of their image data due to lens flare and glare. 
The task at hand is to build a model that will be able to classify an image as being corruptted by glare or not.

For this I'll be using transfer learning on the [ResNet-34](https://arxiv.org/abs/1512.03385) Convolutional Neural Network. 

### Background information on CNNs and Transfer Learning

Convolutional neural networks (CNN) are a class of artificial neural networks that have been used extensively in computer vision. One of the main advantages that CNNs have over normal Multi-layer Perceptron networks is that they are translationally invariate, meaning that a dog, say, would be recognised regardless of its position in an image. It achieves this by passing several filters over the image, typically being a simple linear function on each pixel and its neighbours to create several feature maps. Initially these feature maps will recognise simple features such as outlines but after a few iterations of this it will soon be recognise more complex features such as noses and eyes. 

ResNet-34 is a 34 layer CNN that has been trained extensively on the [ImageNet](http://www.image-net.org/) dataset achieving a top-1 error of 26.70% and and top-5 error of 8.58. The complexity of this CNNs architecture far surpasses the practical limit of what we could train for our task, however by taking it as a pre-trained model we can then build ontop of it. This is the essence of transfer learning and the main intuition behind choosing this method to classify glare in images.

## Implementation

Due to the small size of the training dataset (80 images), more training images were engineered from the originals in order to increase the dataset size and help the model assess images with varying degrees of rotation (found in a few images) and corruption (found in one image):


The stucture of the model is as follows:
ResNet-34 gives a 512 length feature vector as its output


