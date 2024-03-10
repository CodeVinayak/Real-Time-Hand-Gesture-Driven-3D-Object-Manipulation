import cv2
import mediapipe as mp
from pynput.keyboard import Controller as KeyboardController, Key
import time
import pyautogui
import tensorflow as tf
import numpy as np

# Load your custom gesture model
custom_model = tf.keras.models.load_model('gesture_model.keras')

# Initialize Mediapipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

keyboard = KeyboardController()

with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        # Read a frame from the camera
        success, image = cap.read()
        if not success:
            print("Unable to capture a frame from the camera.")
            continue

        # Flip the image horizontally
        image = cv2.flip(image, 1)

        # Convert image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Process the image with Mediapipe hands
        results = hands.process(image_rgb)

        # Convert image back to BGR
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        gesture_class = -1  # Initialize with an invalid value

        if results.multi_hand_landmarks:
            for hand_landmarks, hand_label in zip(results.multi_hand_landmarks, results.multi_handedness):
                handLandmarks = [[landmark.x, landmark.y] for landmark in hand_landmarks.landmark]

                # Check for raised fingers
                thumb_condition = (hand_label.classification[0].label == "Left" and handLandmarks[4][0] > handLandmarks[3][0]) or \
                                  (hand_label.classification[0].label == "Right" and handLandmarks[4][0] < handLandmarks[3][0])

                index_finger_condition = handLandmarks[8][1] < handLandmarks[6][1]
                middle_finger_condition = handLandmarks[12][1] < handLandmarks[10][1]
                ring_finger_condition = handLandmarks[16][1] < handLandmarks[14][1]
                pinky_finger_condition = handLandmarks[20][1] < handLandmarks[18][1]

                if thumb_condition:
                    gesture_class += 1
                if index_finger_condition:
                    gesture_class += 1
                if middle_finger_condition:
                    gesture_class += 1
                if ring_finger_condition:
                    gesture_class += 1
                if pinky_finger_condition:
                    gesture_class += 1

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

            if gesture_class != -1:
                # Reshape the input to match the model's expected shape
                input_data = np.expand_dims(np.expand_dims(np.asarray(gesture_class), axis=0), axis=0)
                
                # Classify the gesture using your custom model
                gesture_scores = custom_model.predict(input_data)
                predicted_class = np.argmax(gesture_scores, axis=-1)

                # Perform corresponding action based on the predicted gesture class
                print("Predicted Gesture Class:", predicted_class)
                action_text = f"Gesture Class: {predicted_class[0]}"

                # Display the action text on the image
                height, width, _ = image.shape
                action_position = ((width - cv2.getTextSize(action_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]) // 2, 40)
                cv2.putText(image, action_text, action_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display the image
        cv2.imshow("Gesture Classification", image)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
