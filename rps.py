import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import mediapipe as mp
import time

def capture_image_with_countdown():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return None

    print("Starting camera... Get ready!")
    for i in range(3, 0, -1):
        print(f"{i}...")
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.putText(frame, f"{i}", (250, 250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 5)
        cv2.imshow("Countdown", frame)
        cv2.waitKey(1000)  # Wait 1 second per number

    print("Shoot!")
    ret, frame = cap.read()
    if ret:
        cv2.putText(frame, "Shoot!", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.imshow("Shoot", frame)
        cv2.waitKey(1000)
    cap.release()
    cv2.destroyAllWindows()
    return frame

def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    return gray, blurred, thresholded

def show_processing_steps(original, gray, blurred, thresholded):
    images = [original, gray, blurred, thresholded]
    titles = ['Original', 'Grayscale', 'Blurred', 'Thresholded']
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()

def recognize_gesture(frame):
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils

    hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.75)
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if not results.multi_hand_landmarks:
        print("No hand detected.")
        return "Unknown"

    hand_landmarks = results.multi_hand_landmarks[0]
    landmarks = hand_landmarks.landmark

    fingers_up = 0
    # Thumb logic (left hand vs. right hand check skipped for simplicity)
    if landmarks[4].x < landmarks[3].x:  # Thumb tip vs. thumb base
        fingers_up += 1

    # Index, middle, ring, pinky
    for tip_id in [8, 12, 16, 20]:
        if landmarks[tip_id].y < landmarks[tip_id - 2].y:
            fingers_up += 1

    print("Fingers up:", fingers_up)

    if fingers_up == 0 or fingers_up == 1:
        return "Rock"
    elif fingers_up == 2:
        return "Scissors"
    elif fingers_up >= 4:
        return "Paper"
    else:
        return "Unknown"

def decide_winner(user, computer):
    if user == computer:
        return "Tie"
    elif (user == "Rock" and computer == "Scissors") or          (user == "Paper" and computer == "Rock") or          (user == "Scissors" and computer == "Paper"):
        return "You Win!"
    else:
        return "You Lose!"

def main():
    frame = capture_image_with_countdown()
    if frame is None:
        return

    gray, blurred, thresholded = preprocess_image(frame)
    show_processing_steps(frame, gray, blurred, thresholded)

    gesture = recognize_gesture(frame)
    computer_choice = random.choice(["Rock", "Paper", "Scissors"])
    result = decide_winner(gesture, computer_choice)

    cv2.putText(frame, f"You: {gesture}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Computer: {computer_choice}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(frame, f"Result: {result}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Game Result", frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
