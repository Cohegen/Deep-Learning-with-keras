# Deep-Learning-with-keras

# **MNIST Classification using Functional API**

This project demonstrates the use of the **Functional API** in Keras to build a Convolutional Neural Network (CNN) for classifying handwritten digits from the **MNIST dataset**.

---

## **Overview**

The Functional API is a flexible way to define models in Keras, allowing for more complex architectures like multi-input, multi-output, and shared-layer models. This project uses the MNIST dataset to implement a simple CNN for digit recognition.

---

## **Dataset**

The **MNIST dataset** contains 60,000 training images and 10,000 testing images of handwritten digits (0-9). Each image is a grayscale image of size **28x28 pixels**.

---

## **Model Architecture**

The model is built using the Functional API and consists of:

1. **Input Layer**: Accepts 28x28 grayscale images with a single channel.
2. **Convolutional Layers**: Extracts features from the input images.
3. **Batch Normalization and Activation Layers**: Stabilizes and accelerates training.
4. **Pooling Layers**: Reduces the spatial dimensions of the feature maps.
5. **Fully Connected Layers**: Combines features for classification.
6. **Output Layer**: Uses a softmax activation function for multi-class classification (10 classes).

### **Architecture Summary**
| Layer                 | Type                  | Output Shape   |
|-----------------------|-----------------------|----------------|
| Input Layer           | Input                | (28, 28, 1)    |
| Conv2D + BatchNorm    | Convolution + BN     | (28, 28, 32)   |
| ReLU + MaxPooling2D   | Activation + Pooling | (14, 14, 32)   |
| Conv2D + BatchNorm    | Convolution + BN     | (14, 14, 64)   |
| ReLU + MaxPooling2D   | Activation + Pooling | (7, 7, 64)     |
| Flatten               | Flatten              | (3136)         |
| Dense + Dropout       | Fully Connected      | (128)          |
| Output                | Dense (Softmax)      | (10)           |

---

## **Dependencies**

- Python 3.8+
- TensorFlow 2.x
- NumPy
- Matplotlib (optional, for visualizing results)

Install the dependencies using:
```bash
pip install tensorflow numpy matplotlib
```

---

## **Project Files**

- `mnist_functional_api.py`: The Python script containing the code for the project.
- `README.md`: Documentation for the project.

---

## **How to Run**

1. Clone this repository:
   ```bash
   git clone https://github.com/your-username/mnist-functional-api.git
   cd mnist-functional-api
   ```

2. Run the Python script:
   ```bash
   python mnist_functional_api.py
   ```

---

## **Results**

After training the model for 10 epochs:

- **Validation Accuracy**: ~98%
- **Test Accuracy**: ~98%

---

## **Key Features of the Functional API**

1. **Flexibility**: Allows building models with multiple inputs and outputs.
2. **Custom Architectures**: Enables non-linear connectivity between layers.
3. **Extensibility**: Easily integrates with custom layers or external features.

---

## **Future Work**

- Experiment with advanced architectures like ResNet or Inception.
- Implement multi-task learning with additional outputs.
- Explore transfer learning for faster convergence.

---

## **References**

- [Keras Functional API Documentation](https://keras.io/guides/functional_api/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

---

## **License**

This project is licensed under the MIT License.
