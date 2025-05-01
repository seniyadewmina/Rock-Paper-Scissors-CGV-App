import cv2
import numpy as np
import random

BG_PATH     = "Resources/BG.png"
AI_ROCK     = "Resources/1.png"
AI_PAPER    = "Resources/2.png"
AI_SCISSORS = "Resources/3.png"

TOP, RIGHT, BOTTOM, LEFT = 10, 350, 225, 590
PLAY_Y1, PLAY_Y2 = 234, 234 + (BOTTOM - TOP)
PLAY_X1, PLAY_X2 = 795, 795 + (LEFT - RIGHT)
AI_POS = (149, 310)

bg = None

def run_avg(image, aW=0.5):
    global bg
    if bg is None:
        bg = image.copy().astype("float")
    else:
        cv2.accumulateWeighted(image, bg, aW)

def segment(image, threshold=15):
    global bg
    diff = cv2.absdiff(bg.astype("uint8"), image)
    cv2.imshow("Background Subtraction", diff)  # debug: show diff from background
    _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    hand = max(cnts, key=cv2.contourArea)
    return thresh, hand

def count_fingers(thresh, hand):
    hull = cv2.convexHull(hand)
    extT = tuple(hull[hull[:,:,1].argmin()][0])
    extB = tuple(hull[hull[:,:,1].argmax()][0])
    extL = tuple(hull[hull[:,:,0].argmin()][0])
    extR = tuple(hull[hull[:,:,0].argmax()][0])
    cX = (extL[0] + extR[0]) // 2
    cY = (extT[1] + extB[1]) // 2

    def dist(p1, p2): return np.hypot(p1[0]-p2[0], p1[1]-p2[1])
    dists = np.array([dist((cX, cY), pt) for pt in (extT, extB, extL, extR)])
    radius = int(0.8 * dists.max())

    mask = np.zeros(thresh.shape[:2], dtype="uint8")
    cv2.circle(mask, (cX, cY), radius, 255, 1)
    circular_roi = cv2.bitwise_and(thresh, thresh, mask=mask)

    cnts, _ = cv2.findContours(circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(f"[DEBUG] Found {len(cnts)} contours in circular ROI")

    # Sort contours by area (largest first), assume wrist is the biggest
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    if len(cnts) > 0:
        cnts = cnts[1:]  # Ignore the wrist (first one)

    # Draw and count remaining as fingers
    for c in cnts:
        cv2.drawContours(circular_roi, [c], -1, (255, 255, 255), 2)

    return len(cnts), circular_roi


def map_gesture(n):
    if n in (1, 0): return "Rock"
    if n in (2, 3): return "Scissors"
    if n >= 4: return "Paper"
    return "Unknown"

def decide(u, a):
    if u == a: return "Tie"
    wins = (u=="Rock" and a=="Scissors") or (u=="Scissors" and a=="Paper") or (u=="Paper" and a=="Rock")
    return "You Win!" if wins else "You Lose!"

def load_icon(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return img[:,:,:3], img[:,:,3]

def overlay_alpha(bg, fg, mask, x, y):
    h,w = fg.shape[:2]
    roi = bg[y:y+h, x:x+w]
    inv = cv2.bitwise_not(mask)
    bg_part = cv2.bitwise_and(roi, roi, mask=inv)
    fg_part = cv2.bitwise_and(fg, fg, mask=mask)
    bg[y:y+h, x:x+w] = cv2.add(bg_part, fg_part)

def show_countdown(cam):
    for i in (3,2,1):
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)
        cv2.rectangle(frame, (LEFT,TOP),(RIGHT,BOTTOM), (0,255,0), 2)
        cv2.putText(frame, str(i), (250,250), cv2.FONT_HERSHEY_SIMPLEX, 5, (0,0,255), 5)
        cv2.imshow("Rock Paper Scissors", frame)
        cv2.waitKey(1000)
    ret, frame = cam.read()
    frame = cv2.flip(frame,1)
    cv2.rectangle(frame, (LEFT,TOP),(RIGHT,BOTTOM), (0,255,0), 2)
    cv2.putText(frame, "Shoot!", (180,250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,255,0), 3)
    cv2.imshow("Rock Paper Scissors", frame)
    cv2.waitKey(1000)

def main():
    bg_img       = cv2.imread(BG_PATH)
    ai_rock, mr  = load_icon(AI_ROCK)
    ai_paper, mp = load_icon(AI_PAPER)
    ai_scis, ms  = load_icon(AI_SCISSORS)

    cam = cv2.VideoCapture(0)
    scores = [0,0]

    print("[INFO] Calibrating background...")
    for _ in range(30):
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)
        roi = frame[TOP:BOTTOM, RIGHT:LEFT]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)
        run_avg(gray, 0.5)

    while True:
        show_countdown(cam)
        ret, frame = cam.read()
        frame = cv2.flip(frame,1)
        clone = frame.copy()

        roi = frame[TOP:BOTTOM, RIGHT:LEFT]
        roi_color = roi.copy()
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.GaussianBlur(gray, (7, 7), 0)

        seg = segment(gray_blur)
        user = "Unknown"
        circular = np.zeros_like(gray)
        if seg:
            thr, hand = seg
            fingers, circular = count_fingers(thr, hand)
            user = map_gesture(fingers)

        ai = random.choice(["Rock","Paper","Scissors"])
        fg, mask = (ai_rock, mr) if ai=="Rock" else (ai_paper, mp) if ai=="Paper" else (ai_scis, ms)
        result = decide(user, ai)
        if result == "You Win!": scores[1] += 1
        elif result == "You Lose!": scores[0] += 1

        disp = bg_img.copy()
        disp[PLAY_Y1:PLAY_Y2, PLAY_X1:PLAY_X2] = roi_color
        overlay_alpha(disp, fg, mask, AI_POS[0], AI_POS[1])
        cv2.putText(disp, str(scores[1]), (1112,215), cv2.FONT_HERSHEY_PLAIN, 4, (0,0,0),6)
        cv2.putText(disp, str(scores[0]), (410,215), cv2.FONT_HERSHEY_PLAIN, 4, (255,255,255),6)
        cv2.putText(disp, user, (900,50), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),2)
        cv2.putText(disp, ai, (50,50), cv2.FONT_HERSHEY_SIMPLEX,1.5,(0,0,0),2)

        # Show image processing steps
        gray_bgr = cv2.cvtColor(gray_blur, cv2.COLOR_GRAY2BGR)
        thr_vis = cv2.cvtColor(seg[0], cv2.COLOR_GRAY2BGR) if seg else gray_bgr.copy()
        circ_bgr = cv2.cvtColor(circular, cv2.COLOR_GRAY2BGR)
        stack = np.hstack([cv2.resize(roi_color, (200,200)), cv2.resize(gray_bgr, (200,200)),
                           cv2.resize(thr_vis, (200,200)), cv2.resize(circ_bgr, (200,200))])
        cv2.imshow("Image Processing Stages", stack)
        cv2.imshow("BG", disp)

        key = cv2.waitKey(0) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            print("[INFO] Recalibrating background...")
            global bg
            bg = None
            for _ in range(30):
                ret, frame = cam.read()
                frame = cv2.flip(frame,1)
                roi = frame[TOP:BOTTOM, RIGHT:LEFT]
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (7, 7), 0)
                run_avg(gray, 0.5)

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
