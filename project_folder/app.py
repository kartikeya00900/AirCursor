from flask import Flask, render_template, redirect, url_for, flash
import threading
import cv2
import mediapipe as mp
import pyautogui
import time

app = Flask(__name__)
app.secret_key = 'supersecretkey'

is_running = False

def run_virtual_mouse():
    global is_running
    is_running = True
    cap = cv2.VideoCapture(0)
    hand_detector = mp.solutions.hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    drawing_utils = mp.solutions.drawing_utils
    screen_width, screen_height = pyautogui.size()
    right_hand_index_y = 0
    left_hand_index_x = 0
    left_hand_index_y = 0

    sensitivity_factor = 2.5
    prev_selection_time = time.time()
    space_pressed = False
    dragging = False
    last_position = None
    last_click_time = 0

    def count_fingers(hand_landmarks):
        count = 0
        threshold = (hand_landmarks.landmark[0].y - hand_landmarks.landmark[9].y) / 2
        if (hand_landmarks.landmark[5].y - hand_landmarks.landmark[8].y) > threshold:
            count += 1
        if (hand_landmarks.landmark[9].y - hand_landmarks.landmark[12].y) > threshold:
            count += 1
        if (hand_landmarks.landmark[13].y - hand_landmarks.landmark[16].y) > threshold:
            count += 1
        if (hand_landmarks.landmark[17].y - hand_landmarks.landmark[20].y) > threshold:
            count += 1
        if (hand_landmarks.landmark[5].x - hand_landmarks.landmark[4].x) > 0.02:
            count += 1
        return count

    while is_running:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand_detector.process(rgb_frame)
        hands = results.multi_hand_landmarks

        if hands:
            for hand_landmarks in hands:
                drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                landmarks = hand_landmarks.landmark
                thumb_tip = landmarks[4]
                index_tip = landmarks[8]
                middle_tip = landmarks[12]
                thumb_x = int(thumb_tip.x * frame_width)
                thumb_y = int(thumb_tip.y * frame_height)
                index_x = int(index_tip.x * frame_width)
                index_y = int(index_tip.y * frame_height)
                middle_x = int(middle_tip.x * frame_width)
                middle_y = int(middle_tip.y * frame_height)
                cv2.circle(frame, (thumb_x, thumb_y), 10, (255, 0, 0), -1)
                cv2.circle(frame, (index_x, index_y), 10, (0, 255, 255), -1)
                cv2.circle(frame, (middle_x, middle_y), 10, (0, 0, 255), -1)
                index_middle_distance = ((index_x - middle_x) ** 2 + (index_y - middle_y) ** 2) ** 0.5
                thumb_middle_distance = ((thumb_x - middle_x) ** 2 + (thumb_y - middle_y) ** 2) ** 0.5
                thumb_index_distance = ((thumb_x - index_x) ** 2 + (thumb_y - index_y) ** 2) ** 0.5
                if index_middle_distance < 50 and thumb_middle_distance < 50 and thumb_index_distance < 50:
                    if not dragging:
                        pyautogui.mouseDown()
                        dragging = True
                else:
                    if dragging:
                        pyautogui.mouseUp()
                        dragging = False
                if index_x < screen_width and index_y < screen_height:
                    pyautogui.moveTo(index_x * sensitivity_factor, index_y * sensitivity_factor)
                finger_count = count_fingers(hand_landmarks)
                if finger_count == 5 and not space_pressed:
                    pyautogui.press('space')
                    space_pressed = True
                elif finger_count < 5:
                    space_pressed = False

        cv2.imshow('Virtual Mouse', frame)

        if cv2.waitKey(1) == 27 or not is_running:
            break

    cap.release()
    cv2.destroyAllWindows()
    is_running = False

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/run', methods=['POST'])
def run():
    global is_running
    if not is_running:
        threading.Thread(target=run_virtual_mouse).start()
        flash('Virtual Mouse started successfully!')
    else:
        flash('Virtual Mouse is already running!')
    return redirect(url_for('home'))

@app.route('/stop', methods=['POST'])
def stop():
    global is_running
    if is_running:
        is_running = False
        flash('Virtual Mouse stopped successfully!')
    else:
        flash('Virtual Mouse is not running!')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
