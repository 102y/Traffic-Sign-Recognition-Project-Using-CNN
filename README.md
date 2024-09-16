Project Description

This project aims to build an artificial intelligence model using Convolutional Neural Networks (CNN) to recognize traffic signs from images. The model is trained on a dataset containing multiple images of traffic signs and classifying them into 43 different categories. After training, the model can be used to recognize traffic signs from new images (outdoor images).

Advantages
Training on an advanced dataset: The traffic signs dataset was used to train the model, containing diverse images covering 43 different types of traffic signs.

Predicting signs from outdoor images: You can upload a new outdoor image (such as an image taken from the road) to classify the traffic sign based on the trained model.

Data normalization and using ImageDataGenerator: To improve the accuracy of the model, the data was normalized and image enhancement techniques were used to increase the effectiveness of learning.

Structure
The project consists of several components:

Data loading and processing: Data is loaded from pickle files and the necessary data processing is applied.
Model building: The model is based on CNN and consists of several layers Convolutional, Pooling and Dense to classify traffic signs.
Training and Evaluation: The model is trained on data and calibrated using validation data, and performance is evaluated using the test set.
External Image Prediction: After training the model, a new image can be uploaded to classify the traffic light and display the result.
Requirements
To run the project locally, you need to install the following packages:
pip install tensorflow keras numpy matplotlib seaborn opencv-python pillow
How to use
Download the project and run it locally.
Train the model using the available traffic light dataset.
After training, use the model to load a new image to predict the traffic light in it.
Example of use:
image_path = '/path/to/your/external/image.jpg'  #Put the external image path here.
predict_external_image(image_path, model, labels)
Dataset
The dataset used in the project can be found on Kaggle. The dataset includes labeled traffic sign images, divided into training, validation, and testing data.

Results
The accuracy of the model is displayed through training and validation graphs.

The Classification Report and Confusion Matrix are displayed to understand the detailed performance of the model.
