Digit Recognizer

This project is a simple handwritten digit recognition program built using the MNIST dataset. It trains a neural network model to recognize and classify single-digit images (0-9). This repository is perfect for machine learning beginners who want to understand the basics of neural networks and image classification.

Features

Dataset: Uses the MNIST dataset of handwritten digits.

Model: A fully connected feedforward neural network with two hidden layers.

Training: The model is trained for 10 epochs with validation.

Prediction: Includes examples of predicting handwritten digits from the test set.

Prerequisites

To run this project, you'll need:

Python 3.7+

Dependencies:

TensorFlow

NumPy

Matplotlib

scikit-learn

Install dependencies using:

pip install tensorflow numpy matplotlib scikit-learn

Code Overview

1. Load and Preprocess Data

The MNIST dataset is loaded and normalized to have pixel values between 0 and 1.

Labels are one-hot encoded for multi-class classification.

2. Split Data

Training data is split into training and validation sets.

3. Build the Model

A sequential model is constructed with the following layers:

Flatten: Converts 28x28 images into a 1D array.

Dense: Fully connected layers with 128 and 64 neurons using ReLU activation.

Output: A softmax layer with 10 neurons for classification.

4. Train the Model

The model is trained using the Adam optimizer and categorical crossentropy loss.

Validation is performed during training to monitor accuracy.

5. Evaluate and Save

The model is evaluated on the test set, and its performance metrics (accuracy and loss) are displayed.

The trained model is saved in the Keras format (digit_recognizer_model.keras).

6. Predict Handwritten Digits

Includes a predict_digit function to preprocess and predict handwritten digits.

Test cases are provided to display sample predictions.

How to Use

Clone the repository:

git clone https://github.com/your_username/digit_recognizer.git
cd digit_recognizer

Run the script:

python digit_recognizer.py

Use the provided test cases to visualize and predict sample handwritten digits from the MNIST test dataset.

Example Outputs

Sample Image:
Displays a grayscale image of the digit.

Actual vs Predicted:
Shows the true label and the model's prediction.

Actual Label: 5
Predicted Label: 5

Multiple Examples:
Includes predictions for test indices like 10, 25, 50, and 75.

Improvements

Extend the Model: Add convolutional layers for better accuracy.

Data Augmentation: Increase dataset size using transformations like rotation and scaling.

Hyperparameter Tuning: Experiment with learning rates, batch sizes, and number of neurons.

License

This project is licensed under the MIT License. See the LICENSE file for details.

Acknowledgments

The MNIST dataset: Yann LeCun's website

TensorFlow Documentation: TensorFlow.org

Feel free to fork this repository and contribute! Happy coding!

