import numpy as np
import os
from PIL import Image
import matplotlib.pyplot as plt
import keras
from keras.utils import to_categorical
from keras import layers
from keras import models
from sklearn.model_selection import train_test_split

# Modify the dataset path to your local directory
dataset_path = '/path/to/your/local/dataset/'

# Dictionary to map filenames to numeric labels
lookup = dict()
# Dictionary to map numeric labels to filenames
reverselookup = dict()
# Initialize count variable
count = 0

x_data = []  # List to store input images
y_data = []  # List to store corresponding labels
datacount = 0  # Initialize variable to tally the total number of images in the dataset

# Iterate over top-level folders
for i in range(10):
    # Iterate over subfolders
    for j in os.listdir(os.path.join(dataset_path, '0' + str(i))):
        if not j.startswith('.'):
            count = 0  # Initialize variable to tally images of a given gesture
            # Iterate over images
            for k in os.listdir(os.path.join(dataset_path, '0' + str(i), j)):
                img = Image.open(os.path.join(dataset_path, '0' + str(i), j, k)).convert('L')
                img = img.resize((320, 120))
                arr = np.array(img)
                x_data.append(arr)
                count += 1
            lookup[j] = count
            reverselookup[count] = j
            y_values = np.full((count, 1), lookup[j])  # Generate labels for the images
            y_data.append(y_values)
            datacount += count

x_data = np.array(x_data, dtype='float32')
y_data = np.array(y_data)
y_data = y_data.reshape(datacount, 1)

# Apply one-hot encoding to the labels in y_data
y_data = to_categorical(y_data)

# Reshape x_data array to match the input shape expected by the neural network model
x_data = x_data.reshape((datacount, 120, 320, 1))

# Normalize pixel values to be in the range [0, 1]
x_data /= 255

# Split the dataset into training and further subsets
x_train, x_further, y_train, y_further = train_test_split(x_data, y_data, test_size=0.2)

# Further split the further subset into validation and test sets
x_validate, x_test, y_validate, y_test = train_test_split(x_further, y_further, test_size=0.5)

model = models.Sequential()

# Add convolutional layers
model.add(layers.Conv2D(32, (5, 5), strides=(2, 2), activation='relu', input_shape=(120, 320, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Flatten the output
model.add(layers.Flatten())

# Add fully connected layers
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))  # Output layer with softmax activation

# Define and Compile the Model
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the Model
history = model.fit(x_train, y_train, epochs=10, batch_size=64, verbose=1, validation_data=(x_validate, y_validate))

# Evaluate the Model on Test Data
[loss, acc] = model.evaluate(x_test, y_test, verbose=1)

# Print the Accuracy
print("Accuracy: " + str(acc))

# Save the trained model
model.save('trained_model.h5')

# Plot the accuracy and loss curves
plt.figure(figsize=(8, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()