import sys
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Camera setup
cap = cv2.VideoCapture(0)

# Get screen size
screen_width, screen_height = pyautogui.size()

def hand_landmarks(image):
    landmark_list = []
    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = image.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                landmark_list.append([id, cx, cy])
    return landmark_list

def fingers_status(landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    # Thumb
    if landmarks[tip_ids[0]][1] > landmarks[tip_ids[0] - 1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    # Other fingers
    for id in range(1, 5):
        if landmarks[tip_ids[id]][2] < landmarks[tip_ids[id] - 2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def detect_gesture(fingers, landmarks):
    if fingers == [1, 1, 1, 1, 1]:
        return "play_pause"
    elif fingers == [0, 0, 0, 0, 0]:
        return "stop"
    elif fingers == [1, 0, 0, 0, 0]:
        return "volume_up"
    elif fingers == [0, 1, 0, 0, 0]:
        return "volume_down"
    elif fingers == [0, 1, 0, 0, 0] and landmarks[8][1] < landmarks[6][1]:
        return "skip_previous"
    elif fingers == [0, 1, 0, 0, 0] and landmarks[8][1] > landmarks[6][1]:
        return "skip_next"
    elif fingers == [0, 0, 0, 0, 1]:  # Forward gesture (pinky finger up)
        return "forward"
    elif fingers == [0, 1, 1, 0, 0]:  # Rewind gesture (index and middle finger up)
        return "rewind"
    return None

while True:
    success, img = cap.read()
    if not success:
        print("Failed to capture image")
        break

    lm_list = hand_landmarks(img)
    if lm_list:
        fingers = fingers_status(lm_list)
        gesture = detect_gesture(fingers, lm_list)

        print(f"Fingers: {fingers}, Gesture: {gesture}")

        if gesture == "play_pause":
            pyautogui.press("space")
        elif gesture == "stop":
            pyautogui.press("k")
        elif gesture == "volume_up":
            pyautogui.press("volumeup")
        elif gesture == "volume_down":
            pyautogui.press("volumedown")
        elif gesture == "skip_next":
            pyautogui.press("nexttrack")
        elif gesture == "skip_previous":
            pyautogui.press("prevtrack")
        elif gesture == "forward":
            pyautogui.press("right")
        elif gesture == "rewind":
            pyautogui.press("left")

    cv2.imshow("Hand Gesture Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
sys.exit()
