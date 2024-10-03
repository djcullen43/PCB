# Car Parking Spot Detection using CNN

This project is a Convolutional Neural Network (CNN) implementation for detecting whether a parking spot is occupied or vacant in images. The model is trained using a dataset of car parking spots, and it classifies each image as either containing a car or not.

## Model Architecture

The CNN model is composed of the following layers:

- **Convolutional Layers**: 3 convolutional layers that extract spatial features from the input images.
- **Pooling Layers**: Max pooling layers that reduce the spatial dimensions.
- **Fully Connected Layers**: Two fully connected layers that process the extracted features and output the class predictions (car/no car).
### Prerequisites
- Python 3.x
- PyTorch
- torchvision
- NumPy
- scikit-learn
- matplotlib

