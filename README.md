# CIFAR-10 CNN

This project implements a Convolutional Neural Network (CNN) to classify images from the CIFAR-10 dataset. The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 different classes. This project demonstrates the use of deep learning techniques for image classification.

## Features

- **Data Preprocessing**: 
  - Normalization of image data for faster convergence.
  - Splitting dataset into training, validation, and testing sets.

- **Data Augmentation**: 
  - Techniques like random cropping, flipping, and rotation to increase dataset variability.

- **Model Architecture**:
  - A custom-built CNN designed to balance performance and computational efficiency.
  - Includes multiple convolutional, pooling, and fully connected layers.

- **Training Enhancements**:
  - Regularization techniques like dropout to prevent overfitting.
  - Learning rate scheduling for optimized training.
  - Early stopping to avoid unnecessary training cycles.

- **Evaluation**:
  - Metrics like accuracy, precision, and recall to measure model performance.
  - Visualizations of training progress (e.g., loss and accuracy curves).

## Technologies Used

- **Programming Language**: Python
- **Deep Learning Library**: TensorFlow / PyTorch
- **Visualization**: Matplotlib, Seaborn
- **Dataset**: CIFAR-10 (available via `torchvision` or `tensorflow_datasets`)

## Installation and Usage

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/zeyame/CIFAR10-CNN.git
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Project**:
   - Train the model:
     ```bash
     python train.py
     ```
   - Evaluate the model:
     ```bash
     python evaluate.py
     ```

4. **Explore Results**:
   - Visualize training and validation metrics.
   - Test the model with custom images.
