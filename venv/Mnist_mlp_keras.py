import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
import matplotlib.pyplot as plt

# Load the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.dtype)
print(x_train.shape)
print(y_test.shape)
print(f"label is : {y_train[0]}")

# Display first image
plt.imshow(x_train[0], cmap='gray')
plt.show()

# Normalize
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# To categorical
print(f"Before : label is : {y_train[100]}")
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
print(f"After  : label is : {y_train[100]}")

# Architecture
model = Sequential()
model.add(Flatten(input_shape=(28, 28)))         # Flatten layer
model.add(Dense(128, activation='relu'))         # Hidden layer
model.add(Dense(10, activation='softmax'))       # Output layer

# Compile
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train
result = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

# evaluate
loss, accuracy = model.evaluate(x_test, y_test)
print(f"test loss:{loss}")
print(f"test accuracy:{accuracy}")
print(result.history.keys())
print(result.history.values())
print(result.history)

#visualization 
plt.plot(result.history['val_accuracy'], label="validation accuracy",color='blue')
plt.plot(result.history['accuracy'], label="training accuracy",color='red')
plt.title("Train_accuracy vs val_accuracy")
plt.xlabel("Epochs")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(result.history['val_loss'], label="validation loss",color='blue')
plt.plot(result.history['loss'], label="training loss",color='red')
plt.title("Train_accuracy vs val_loss")
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()
