import cv2
from pynput.keyboard import Controller as KeyboardController, Key
import time
import pyautogui
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load your custom model
gesture_model = keras.models.load_model('gesture_model.keras')

# For webcam input:
cap = cv2.VideoCapture(0)

keyboard = KeyboardController()

def preprocess_image(image):
    # Resize the image to match your model's input size
    target_size = (224, 224)  # Replace with your model's input size
    resized_image = cv2.resize(image, target_size)

    # Convert the image to a numpy array and normalize pixel values (if needed)
    normalized_image = resized_image / 255.0

    # Add batch dimension if your model expects it
    input_image = np.expand_dims(normalized_image, axis=0)

    return input_image

def convert_to_finger_count(predicted_gesture_class):
    # Map the predicted gesture class (0-19) to the corresponding finger count
    # Implement this function based on your gesture class definitions
    # Return the corresponding finger count
    pass

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

    # Convert the predicted class to the corresponding finger count
    fingerCount = convert_to_finger_count(predicted_gesture_class)

    # Perform key press based on raised finger count
    if fingerCount == 1:
        keyboard.press(Key.alt)
        keyboard.press(Key.right)
        time.sleep(0.1)
        keyboard.release(Key.right)
        keyboard.release(Key.alt)
        action_text = "Right Rotation"
    elif fingerCount == 2:
        keyboard.press(Key.alt)
        keyboard.press(Key.left)
        time.sleep(0.1)
        keyboard.release(Key.left)
        keyboard.release(Key.alt)
        action_text = "Left Rotation"
    elif fingerCount == 3:
        keyboard.press(Key.alt)
        keyboard.press(Key.up)
        time.sleep(0.1)
        keyboard.release(Key.up)
        keyboard.release(Key.alt)
        action_text = "Up Rotation"
    elif fingerCount == 4:
        keyboard.press(Key.alt)
        keyboard.press(Key.down)
        time.sleep(0.1)
        keyboard.release(Key.down)
        keyboard.release(Key.alt)
        action_text = "Down Rotation"
    elif fingerCount == 5:
        pyautogui.hscroll(+10)
        action_text = "Zoom In"
    elif fingerCount == 6:
        pyautogui.hscroll(-10)
        action_text = "Zoom out"
    else:
        action_text = "No Action"

    # Display the count of raised fingers and action text on the image
    height, width, _ = image.shape
    label_position = ((width - cv2.getTextSize("No. of raised fingers:", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]) // 2, 40)
    count_position = ((width - cv2.getTextSize(str(fingerCount), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0][0]) // 2, 95)
    action_position = ((width - cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]) // 2, 140)

    cv2.putText(image, "No. of raised fingers:", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(image, str(fingerCount), count_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)
    cv2.putText(image, action_text, action_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # Display the image
    cv2.imshow("Raised Finger Counter", image)

    # Break the loop if 'Esc' key is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()