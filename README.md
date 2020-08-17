# COVID-19-2020-Project
# COVID-19 XRAY Detector X
## Using Image classification in machine learning
Presented By : Ali Shhadi (11059) & Hadi Jaber (9623)


# OVERVIEW
1) Introduction To Machine Learning .
2) Image Classification .
3) COVID-19 .
4) Used Model .
5) Model Data Set .
6) Data Pre-Processing .
7) Binary Classification .
8) Class Imbalance .
9) Model Result .
10) Introduction To TensorFlow .
11) TensorFlow Lite .
12) TensorFlow Converter .
13) Convert Keras To Quantized_TFLite .
14) Why Quantized ?
15) COVID-19 XRAY Detector X .
16) COVID-19 XRAY Detector X (Import Photo).
17) COVID-19 XRAY Detector X (Real-Time Camera).
18) References .
19) Thank You .

# Introduction To Machine Learning
- Machine learning is a subfield of artificial intelligence (AI). The goal of machine learning generally is to understand the structure of data and fit that data into models that can be understood and utilized by people. 

- Machine Learning mean making prediction based on data.

- Feature extraction starts from an initial set of measured data and builds derived values (features) intended to be informative and non-redundant.

- Supervised : Output are Known ( Labeled ).
- Unsupervised : Output are Unknown ( Unlabeled ).

# Image Classification

- Image classification is a supervised learning problem: define a set of target classes (objects to identify in images), and train a model to recognize them using labeled example photos.
- The task of predicting what an image represents is called image classification. An image classification model is trained to recognize various classes of images. For example, a model might be trained to recognize photos representing three different types of animals: rabbits, hamsters, and dogs.

- When we subsequently provide a new image as input to the model, it will output the probabilities of the image representing each of the types of animal it was trained on. 

# COVID-19
- Since reverse transcription polymerase chain reaction (RT-PCR) test kits are in limited supply, there exists a need to explore alternative means of identifying and prioritizing suspected cases of COVID-19.

- Moreover, large scale implementation of the COVID-19 tests which are extremely expensive cannot be afforded by many of the developing & underdeveloped countries hence if we can have some parallel diagnosis/testing procedures using Artificial Intelligence & Machine Learning and leveraging the historical data, it will be extremely helpful. This can also help in the process to select the ones to be tested primarily.

- This can be done by classifying X-Ray images that are non expensive with high availability.

# Used Model
- A pretrained Keras model was used as  the main model.

- A deep convolutional neural network architecture was trained to perform binary classification COVID-19 positive or negative.

- DenseNet architecture was used in the model. Which is faster and every feature has skip to all following features.

# Model Data	Set
- For the purpose of this experiment, data was taken from two repositories:
https://github.com/ieee8023/covid-chestxray-dataset
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia

- A sample of the data was taken which is a set of x-ray images some which are positive and some which are normal.

# Data Pre-Processing
- Prior to training, preprocessing was implemented on the images themselves. ImageDataGenerator (from tensorflow.keras) was employed to perform preprocessing of image batches prior to training.

 - The following transformations were applied to images: Images were resized to have the following shape: 500×500×3. By reducing the image size, the number of parameters in the neural network was decreased. images are normalized by scaling them so their pixel values are in the range [0, 1].

# Binary Classification
- We first considered a binary classification problem where the goal was to detect whether an X-ray shows evidence of COVID-19 infection.

- The classifier was to assign X-ray images to either a non-COVID-19 class or a COVID-19 class.

- A deep convolutional neural network architecture was trained to perform binary classification.

# Class Imbalance
- Due to the imbalance in severe COVID-19 cases     imbalance of images .

- Accuracy can be misleadingly high in classification problems when a class is underrepresented.

- We apply Image Augmentation to overcome this issue. ( Increase the number of images by creating images of difference contrast/size from existing image )

# Model Result
- Test Accuracy 99.33%
- Loss:0.0123

# Introduction To TensorFlow
TensorFlow provides a collection of workflows to develop and train models using Python, JavaScript, or Swift, and to easily deploy in the cloud, on-prem, in the browser, or on-device no matter what language you use.

# TensorFlow Lite
Tensorflow for mobile.

# Converting Keras To Quantized_TFLite
- We converted the pre-trained Keras model to quantized_tflite format.

- This allows us to use the model in our android application.

#Why Quantized ?
- The simplest way to create a small model is to quantize the weights to 8 bits and quantize the inputs/activations "on-the-fly", during inference. This has latency benefits, but prioritizes size reduction.

- This technique helps reduce both the memory requirement and computational cost of using neural networks at the cost of modest decrease in accuracy.

# Implement Model Android App


