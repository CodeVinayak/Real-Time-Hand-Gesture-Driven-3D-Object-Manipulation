import cv2
from pynput.keyboard import Controller as KeyboardController, Key
import time
import pyautogui
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load your custom model
gesture_model = keras.models.load_model('gesture_model.keras')

# Define gesture class names
gesture_classes = ['class_0', 'class_1', 'class_2', 'class_3', 'class_4', 'class_5', 'class_6', 'class_7', 'class_8', 'class_9',
                   'class_10', 'class_11', 'class_12', 'class_13', 'class_14', 'class_15', 'class_16', 'class_17', 'class_18', 'class_19']

# For webcam input:
cap = cv2.VideoCapture(0)

keyboard = KeyboardController()

def preprocess_image(image):
    # Resize the image to match your model's input size
    target_size = (64, 64)  # Replace with your model's input size
    resized_image = cv2.resize(image, target_size)

    # Convert the image to a numpy array and normalize pixel values
    normalized_image = resized_image / 255.0

    # Add batch dimension if your model expects it
    input_image = np.expand_dims(normalized_image, axis=0)

    return input_image

while cap.isOpened():
    # Read a frame from the camera
    success, image = cap.read()
    if not success:
        print("Unable to capture a frame from the camera.")
        continue

    # Flip the image horizontally
    image = cv2.flip(image, 1)

    # Preprocess the image for your model
    preprocessed_image = preprocess_image(image)

    # Make a prediction with your custom model
    prediction = gesture_model.predict(preprocessed_image)
    predicted_gesture_class = np.argmax(prediction)

    # Get the corresponding gesture name
    predicted_gesture_name = gesture_classes[predicted_gesture_class]

    # Print the predicted gesture name
    print("Predicted gesture:", predicted_gesture_name)

    # Display the predicted gesture name on the image
    height, width, _ = image.shape
    gesture_position = ((width - cv2.getTextSize(predicted_gesture_name, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]) // 2, 30)
    cv2.putText(image, predicted_gesture_name, gesture_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the input image with the predicted gesture name
    cv2.imshow("Prediction", image)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()