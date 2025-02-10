1.#Import Libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

2.#Load and process the data
# Load the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize pixel values to [0, 1] range
x_train = x_train / 255.0
x_test = x_test / 255.0

# Add a channel dimension (for grayscale images)
x_train = x_train.reshape(-1, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

#3.Define the functional API model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization, ReLU

# Input Layer
inputs = Input(shape=(28, 28, 1), name="input_layer")

# Convolutional Block 1
x = Conv2D(32, (3, 3), padding="same", activation=None, name="conv1")(inputs)
x = BatchNormalization(name="batch_norm1")(x)
x = ReLU(name="relu1")(x)
x = MaxPooling2D((2, 2), name="maxpool1")(x)

# Convolutional Block 2
x = Conv2D(64, (3, 3), padding="same", activation=None, name="conv2")(x)
x = BatchNormalization(name="batch_norm2")(x)
x = ReLU(name="relu2")(x)
x = MaxPooling2D((2, 2), name="maxpool2")(x)

# Flatten and Fully Connected Layers
x = Flatten(name="flatten")(x)
x = Dense(128, activation="relu", name="fc1")(x)
x = Dropout(0.5, name="dropout")(x)
outputs = Dense(10, activation="softmax", name="output_layer")(x)

# Define the Model
model = Model(inputs=inputs, outputs=outputs, name="MNIST_Functional_API_Model")

# Compile the Model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

#4.Train the model
# Train the model
history = model.fit(
    x_train, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32
)

#5.Evaluate the model
# Evaluate on test data
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

#6.Model Summary
# Display model architecture
model.summary()


#Disclaimer for proper running of the code place each numbered code blocks 
# in their individual jupyter notebook cells
