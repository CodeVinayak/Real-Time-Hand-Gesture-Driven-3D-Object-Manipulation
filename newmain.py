import cv2
import mediapipe as mp
import numpy as np
from joblib import load
from sklearn.preprocessing import Normalizer
from pynput.keyboard import Key, Controller as KeyboardController
import pyautogui
import time

## Open capture with video path
capture = cv2.VideoCapture(0)

## Initialize mediapipe hand detection function
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

## Load trained model and initialize a normalizer 
model = load("model.joblib")
normalizer = Normalizer()

## Define variables for output video
h = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
size = (w, h)

## Create VideoWriter instance with variables taken from input
outputVid = cv2.VideoWriter("result.avi", cv2.VideoWriter_fourcc('M','J','P','G'), 24, size, isColor=True)

## Create a keyboard controller
keyboard = KeyboardController()

## Helper function to create a bounding box around each hand.
## Takes in video frame img and hand landmarks lm
def createBoundingBox(img, lm):

    ## Initialize empty array to store all landmarks of 
    ## hand landmark lm
    lm_array = np.empty((0,2), int)

    ## For each landmark in hand landmark, append 
    ## minimum points to array
    for _, landmark in enumerate(lm.landmark):
        
        width, height = img.shape[1], img.shape[0]
        ## Calculate minimum point between landmark
        ## position and size of video frame
        lm_x = min(int(landmark.x * width), width - 1)
        lm_y = min(int(landmark.y * height), height - 1)

        ## Create a point using the minimum for landmark
        lm_point = [np.array((lm_x, lm_y))]

        ## Append point to array
        lm_array = np.append(lm_array, lm_point, axis=0)

    ## Using built-in method boundingRect, get the x,y,w,h
    ## from the bounding box of lm_array
    x, y, w, h = cv2.boundingRect(lm_array)

    ## Define positions for bouding box to encapsulate hand
    x_min = x - 20
    y_min = y - 15
    x_max = x + w + 20
    y_max = y + h + 15

    return [x_min, y_min, x_max, y_max]

## While capture is open
while(capture.isOpened()):

    ## Read the frame from capture
    read, frame = capture.read()

    frame = cv2.flip(frame, 1)

    ## If frame was properly read
    if read == True:
        
        ## Convert frame to RGB for proper mediapipe detection
        rgbFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        ## Process each frame to get hand landmarks
        results = hands.process(rgbFrame)

        ## If results exists
        if results.multi_hand_landmarks:

            ## For each hand detected
            for handLms in results.multi_hand_landmarks:

                ## Call upon createBoundingBox() method to get bounding box coordinates
                boudingBox = createBoundingBox(frame, handLms)

                ## Draw a rectangle around each processed bounding box
                cv2.rectangle(frame, (boudingBox[0], boudingBox[1]), (boudingBox[2], boudingBox[3]), (0, 255, 0), 2)

                ## Draw the connections between landmarks for better visualization
                mp_drawing.draw_landmarks(frame, handLms, mpHands.HAND_CONNECTIONS)

                ## Define coords as the landmark's x and y coordinates and normalize them
                coords = handLms.landmark
                coords = list(np.array([[landmark.x, landmark.y] for landmark in coords]).flatten())
                coords = normalizer.transform([coords])

                ## Predict which letter is being gestured using the trained model
                predicted_letter = model.predict(coords)

                # Write above the bouding box the predicted letter
                cv2.putText(frame, str(predicted_letter[0]),(boudingBox[0], boudingBox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

                # Perform action based on predicted letter
                if str(predicted_letter[0]).upper() == 'A' or str(predicted_letter[0]).upper() == 'M':
                    keyboard.press(Key.alt)
                    keyboard.press(Key.right)
                    time.sleep(0.1)
                    keyboard.release(Key.right)
                    keyboard.release(Key.alt)
                    action_text = "Right Rotation"
                elif str(predicted_letter[0]).upper() == 'B':
                    keyboard.press(Key.alt)
                    keyboard.press(Key.left)
                    time.sleep(0.1)
                    keyboard.release(Key.left)
                    keyboard.release(Key.alt)
                    action_text = "Left Rotation"
                elif str(predicted_letter[0]).upper() == 'C':
                    keyboard.press(Key.alt)
                    keyboard.press(Key.up)
                    time.sleep(0.1)
                    keyboard.release(Key.up)
                    keyboard.release(Key.alt)
                    action_text = "Up Rotation"
                elif str(predicted_letter[0]).upper() == 'D':
                    keyboard.press(Key.alt)
                    keyboard.press(Key.down)
                    time.sleep(0.1)
                    keyboard.release(Key.down)
                    keyboard.release(Key.alt)
                    action_text = "Down Rotation"
                elif str(predicted_letter[0]).upper() == 'R':
                    keyboard.press(Key.home)
                    time.sleep(0.1)
                    keyboard.release(Key.home)
                    action_text = "Home Button Pressed"
                elif str(predicted_letter[0]).upper() == 'V' or str(predicted_letter[0]).upper() == 'K':
                    pyautogui.hscroll(+10)
                    action_text = "Zoom In"
                elif str(predicted_letter[0]).upper() == 'Y':
                    pyautogui.hscroll(-10)
                    action_text = "Zoom Out"
                

    cv2.imshow("Frame", frame)

    ## Write frame with detection results to VideoWriter 
    ## instance outputVid
    outputVid.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
outputVid.release()
cv2.destroyAllWindows()
