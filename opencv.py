# Importing required libraries
import cv2  # For video capturing and image processing
import mediapipe as mp  # For hand tracking and landmark detection
import pyautogui  # For simulating keyboard inputs to control media playback
import time  # For adding delay between gesture detections

# Initialize MediaPipe drawing utilities and hand solutions
mp_drawing = mp.solutions.drawing_utils  # For drawing landmarks and connections on the image
mp_hands = mp.solutions.hands  # MediaPipe hands solution for hand tracking

# IDs of fingertip landmarks as defined in MediaPipe's hand model
tipIds = [4, 8, 12, 16, 20]

# Set camera resolution
wCam, hCam = 720, 640

# Start webcam capture and set resolution
cap = cv2.VideoCapture(0)  # Use camera index 0 (change if needed)
cap.set(3, wCam)  # Set width of the webcam feed
cap.set(4, hCam)  # Set height of the webcam feed

def fingers_up(hand_landmarks):
    fingers = []
    # Thumb
    if hand_landmarks.landmark[tipIds[0]].x < hand_landmarks.landmark[tipIds[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Fingers
    for id in range(1, 5):
        if hand_landmarks.landmark[tipIds[id]].y < hand_landmarks.landmark[tipIds[id] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Initialize MediaPipe Hands with confidence thresholds
with mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands:
    prev_gesture_time = 0
    gesture_delay = 1.0  # seconds

    while cap.isOpened():  # Continue while the webcam is open
        success, image = cap.read()  # Read a frame from the webcam
        if not success:  # Skip if the frame is empty
            print("Ignoring empty camera frame.")
            continue

        # Process the frame for hand tracking
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)  # Flip and convert BGR to RGB
        image.flags.writeable = False  # Make the image unwriteable to improve performance
        results = hands.process(image)  # Process the image for hand landmarks
        image.flags.writeable = True  # Make the image writable again
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for OpenCV

        current_time = time.time()

        # Draw hand landmarks and detect gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = fingers_up(hand_landmarks)

                if current_time - prev_gesture_time > gesture_delay:
                    # Gesture controls
                    # Play/Pause: Only index finger up
                    if fingers == [0, 1, 0, 0, 0]:
                        pyautogui.press('space')
                        cv2.putText(image, "Play/Pause", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        prev_gesture_time = current_time

                    # Volume Up: Thumb up
                    elif fingers == [1, 0, 0, 0, 0]:
                        pyautogui.press('volumeup')
                        cv2.putText(image, "Volume Up", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        prev_gesture_time = current_time

                    # Volume Down: Pinky up
                    elif fingers == [0, 0, 0, 0, 1]:
                        pyautogui.press('volumedown')
                        cv2.putText(image, "Volume Down", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        prev_gesture_time = current_time

                    # Next Track: Two fingers up (index and middle)
                    elif fingers == [0, 1, 1, 0, 0]:
                        pyautogui.press('nexttrack')
                        cv2.putText(image, "Next Track", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        prev_gesture_time = current_time

                    # Previous Track: Three fingers up (index, middle, ring)
                    elif fingers == [0, 1, 1, 1, 0]:
                        pyautogui.press('prevtrack')
                        cv2.putText(image, "Previous Track", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        prev_gesture_time = current_time

        # Display instructions overlay
        instructions = [
            "Gestures:",
            "Index finger up: Play/Pause",
            "Thumb up: Volume Up",
            "Pinky up: Volume Down",
            "Index+Middle: Next Track",
            "Index+Middle+Ring: Previous Track",
            "Press 'q' to quit"
        ]
        y0, dy = 30, 25
        for i, line in enumerate(instructions):
            y = y0 + i * dy
            cv2.putText(image, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Hand Controlled Media Player", image)  # Display the image with hand landmarks

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Exit on 'q' key press
            break

cap.release()  # Release the webcam
cv2.destroyAllWindows()  # Close all OpenCV windows
