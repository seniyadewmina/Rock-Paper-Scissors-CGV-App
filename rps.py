import cv2
import mediapipe as mp
import numpy as np
import random
import time

# Initialize MediaPipe
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Define gestures
def get_gesture(finger_states):
    if finger_states == [0, 0, 0, 0, 0]:
        return "Rock"
    elif finger_states == [1, 1, 0, 0, 0] or finger_states == [0, 1, 1, 0, 0]:
        return "Scissors"
    elif finger_states == [1, 1, 1, 1, 1]:
        return "Paper"
    else:
        return "Unknown"

def decide_winner(user, computer):
    if user == computer:
        return "Tie"
    elif (user == "Rock" and computer == "Scissors") or \
         (user == "Paper" and computer == "Rock") or \
         (user == "Scissors" and computer == "Paper"):
        return "You Win!"
    else:
        return "You Lose!"

# Webcam capture
cap = cv2.VideoCapture(0)

while True:
    # Countdown
    for i in range(3, 0, -1):
        success, frame = cap.read()
        if not success:
            break
        cv2.putText(frame, str(i), (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
        cv2.imshow("Rock Paper Scissors", frame)
        cv2.waitKey(1000)

    # Show "Shoot!"
    success, frame = cap.read()
    cv2.putText(frame, "Shoot!", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.imshow("Rock Paper Scissors", frame)
    cv2.waitKey(1000)

    # Preprocess and detect
    success, frame = cap.read()
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    user_gesture = "Unknown"
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Analyze landmarks
            landmarks = hand_landmarks.landmark
            finger_states = []
            tips_ids = [4, 8, 12, 16, 20]
            hand_label = results.multi_handedness[0].classification[0].label  # "Right" or "Left"

            for i, tip in enumerate(tips_ids):
                if i == 0:
                    # Thumb: direction depends on left or right hand
                    if hand_label == "Right":
                        finger_states.append(landmarks[tip].x < landmarks[tip - 1].x)
                    else:
                        finger_states.append(landmarks[tip].x > landmarks[tip - 1].x)
                else:
                    # Other fingers: y position check
                    finger_states.append(landmarks[tip].y < landmarks[tip - 2].y)

            user_gesture = get_gesture([int(state) for state in finger_states])
            break

    computer_gesture = random.choice(["Rock", "Paper", "Scissors"])
    result = decide_winner(user_gesture, computer_gesture)

    # Show result
    cv2.putText(frame, f"You: {user_gesture}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    cv2.putText(frame, f"Computer: {computer_gesture}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.putText(frame, f"Result: {result}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("Game Result", frame)
    print(f"You: {user_gesture} | Computer: {computer_gesture} -> {result}")
    print("Press 'q' to quit or any other key to play again.")

    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()