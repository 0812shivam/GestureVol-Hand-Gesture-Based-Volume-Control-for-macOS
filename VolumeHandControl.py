import cv2
import mediapipe as mp
import math
import numpy as np
import subprocess

# Solution APIs
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Volume Control Function for macOS
def set_volume(volume):
    subprocess.call(["osascript", "-e", f"set volume output volume {volume}"])

# Webcam Setup
wCam, hCam = 640, 480
cam = cv2.VideoCapture(0)
cam.set(3, wCam)
cam.set(4, hCam)

# Mediapipe Hand Landmark Model
with mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:

    while cam.isOpened():
        success, image = cam.read()
        if not success:
            break

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

            # Multi-hand landmarks method for finding position of hand landmarks
            lmList = []
            myHand = results.multi_hand_landmarks[0]
            for id, lm in enumerate(myHand.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])

            # Assigning variables for Thumb and Index finger position
            if len(lmList) != 0:
                x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
                x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip

                # Marking Thumb and Index finger (optional, can be removed if not needed)
                cv2.circle(image, (x1, y1), 15, (255, 255, 255))
                cv2.circle(image, (x2, y2), 15, (255, 255, 255))
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Calculate distance between thumb and index finger
                length = math.hypot(x2 - x1, y2 - y1)

                # Volume settings
                minVol = 0
                maxVol = 100  # macOS volume range is from 0 to 100
                vol = np.interp(length, [50, 220], [minVol, maxVol])
                vol = int(np.clip(vol, minVol, maxVol))  # Ensure volume is within range

                # Set system volume
                set_volume(vol)

                # Draw volume bar
                bar_length = 300
                bar_x = 50
                bar_y = 50
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + bar_length, bar_y + 30), (255, 255, 255), -1)  # Background
                cv2.rectangle(image, (bar_x, bar_y), (bar_x + int(bar_length * (vol / maxVol)), bar_y + 30), (0, 255, 0), -1)  # Volume level

                # Display current volume
                cv2.putText(image, f'Volume: {vol}', (bar_x, bar_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow('handDetector', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cam.release()
cv2.destroyAllWindows()
