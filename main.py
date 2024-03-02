import cv2
import mediapipe as mp
import keyboard
import json

# Set up Mediapipe Hand module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# Function to perform key press based on finger count
def perform_key_press(finger_count):
    if finger_count == 1:
         # Press Alt + Right arrow key when 1 fingers are detected
        keyboard.press_and_release("alt+right")
    elif finger_count == 2:
        # Press Alt + Left arrow key when 2 fingers are detected
        keyboard.press_and_release("alt+left")
    elif finger_count == 3:
        # Press Alt + Up arrow key when 3 fingers are detected
        keyboard.press_and_release("alt+up")
    elif finger_count == 4:
        # Press Alt + Down arrow key when 4 fingers are detected
        keyboard.press_and_release("alt+down")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    h, w, _ = image.shape

    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and detect hands
    results = hands.process(rgb_image)

    finger_count = 0  # Variable to store the number of fingers

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Count the number of extended fingers
            finger_count = sum([1 for lm in hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP:] if lm.y < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP].y])

            # Draw hand landmarks and finger count on the image
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            cv2.putText(image, f"Fingers: {finger_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Perform key press based on finger count
            perform_key_press(finger_count)

    # Display the image
    image = cv2.resize(image, (0, 0), None, 0.5, 0.5)
    cv2.imshow("Hand Tracking", image)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
