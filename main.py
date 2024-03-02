import cv2
import mediapipe as mp

# Initialize Mediapipe hands module
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# For webcam input:
cap = cv2.VideoCapture(0)

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

        fingerCount = 0

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
                    fingerCount += 1
                if index_finger_condition:
                    fingerCount += 1
                if middle_finger_condition:
                    fingerCount += 1
                if ring_finger_condition:
                    fingerCount += 1
                if pinky_finger_condition:
                    fingerCount += 1

                # Draw landmarks and connections
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                    connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2))

        # Display the count of raised fingers
        height, width, _ = image.shape
        label_position = ((width - cv2.getTextSize("No. of raised fingers:", cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0]) // 2, 40)
        count_position = ((width - cv2.getTextSize(str(fingerCount), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 4)[0][0]) // 2, 95)

        cv2.putText(image, "No. of raised fingers:", label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, str(fingerCount), count_position, cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 4)

        # Display the image
        cv2.imshow("Raised Finger Counter", image)

        # Break the loop if 'Esc' key is pressed
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
