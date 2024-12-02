# vel-vision 1.0
import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# pinching gesture detection function (thumb and index finger touching)
def detect_pinching(landmarks):
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]
    distance = np.sqrt((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2)
    return distance < 0.05

# draw hand landmarks function
def draw_hand_skeleton(frame, hand_landmarks):
    mp_drawing.draw_landmarks(
        frame, 
        hand_landmarks, 
        mp_hands.HAND_CONNECTIONS, 
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),  # Red for landmarks and connections
        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
    )

# get hand type (left or right)
def get_hand_type(handedness):
    return handedness.classification[0].label

# bounding box
def get_hand_bounding_box(hand_landmarks, frame_width, frame_height, padding=50):
    x_coords = [lm.x * frame_width for lm in hand_landmarks.landmark]
    y_coords = [lm.y * frame_height for lm in hand_landmarks.landmark]
    
    x_min, x_max = max(int(min(x_coords)) - padding, 0), min(int(max(x_coords)) + padding, frame_width)
    y_min, y_max = max(int(min(y_coords)) - padding, 0), min(int(max(y_coords)) + padding, frame_height)
    
    return x_min, y_min, x_max, y_max

# wrist coordinates
def display_wrist_coords(frame, wrist_coords):
    coord_text = f"X: {wrist_coords[0]}, Y: {wrist_coords[1]}"
    cv2.putText(frame, coord_text, (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

# initialize window positions
left_window_pos = [100, 100]  # Initial position for the left hand window
right_window_pos = [400, 100]  # Initial position for the right hand window
window_size = (300, 300)

# variables to track the last known wrist positions
last_wrist_pos_left = None
last_wrist_pos_right = None

# start capturing the webcam
cap = cv2.VideoCapture(0)

# initialize MediaPipe hands object
with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Ignoring empty frame.")
            continue
        
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape
        
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        main_frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

        if results.multi_hand_landmarks:
            for i, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # get the type of hand (left or right)
                hand_type = get_hand_type(results.multi_handedness[i])

                draw_hand_skeleton(main_frame, hand_landmarks)
                wrist = hand_landmarks.landmark[0]
                wrist_coords = (int(wrist.x * frame_width), int(wrist.y * frame_height))
                cv2.putText(main_frame, hand_type[0], wrist_coords, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                x_min, y_min, x_max, y_max = get_hand_bounding_box(hand_landmarks, frame_width, frame_height)

                color = (255, 0, 0)  
                cv2.rectangle(main_frame, (x_min, y_min), (x_max, y_max), color, 2) 

                # detect pinching gesture
                if detect_pinching(hand_landmarks):
                    if hand_type == "Left":
                        if last_wrist_pos_left is None:  # first pinch detection for left hand
                            last_wrist_pos_left = wrist_coords
                        else:
                            delta_x = wrist_coords[0] - last_wrist_pos_left[0]
                            delta_y = wrist_coords[1] - last_wrist_pos_left[1]
                            left_window_pos[0] += delta_x
                            left_window_pos[1] += delta_y
                            last_wrist_pos_left = wrist_coords  # update last position
                    elif hand_type == "Right":
                        if last_wrist_pos_right is None:  # first pinch detection for right hand
                            last_wrist_pos_right = wrist_coords
                        else:
                            delta_x = wrist_coords[0] - last_wrist_pos_right[0]
                            delta_y = wrist_coords[1] - last_wrist_pos_right[1]
                            right_window_pos[0] += delta_x
                            right_window_pos[1] += delta_y
                            last_wrist_pos_right = wrist_coords  # update last position
                else:
                    if hand_type == "Left":
                        last_wrist_pos_left = None
                    elif hand_type == "Right":
                        last_wrist_pos_right = None

                padding = 50
                x_min = max(x_min - padding, 0)
                y_min = max(y_min - padding, 0)
                x_max = min(x_max + padding, frame_width)
                y_max = min(y_max + padding, frame_height)

                hand_crop = frame[y_min:y_max, x_min:x_max]
                hand_crop_resized = cv2.resize(hand_crop, window_size)

                if hand_type == "Left":
                    draw_hand_skeleton(hand_crop_resized, hand_landmarks)
                    display_wrist_coords(hand_crop_resized, wrist_coords)
                    cv2.imshow('vel.vision - left', hand_crop_resized)
                    cv2.moveWindow('vel.vision - left', left_window_pos[0], left_window_pos[1])
                elif hand_type == "Right":
                    draw_hand_skeleton(hand_crop_resized, hand_landmarks)
                    display_wrist_coords(hand_crop_resized, wrist_coords)
                    cv2.imshow('vel.vision - right', hand_crop_resized)
                    cv2.moveWindow('vel.vision - right', right_window_pos[0], right_window_pos[1])
        else:
            black_image_left = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
            black_image_right = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
            
            cv2.imshow('vel.vision - left', black_image_left)
            cv2.moveWindow('vel.vision - left', left_window_pos[0], left_window_pos[1])

            cv2.imshow('vel.vision - right', black_image_right)
            cv2.moveWindow('vel.vision - right', right_window_pos[0], right_window_pos[1])

        cv2.imshow('vel.vision - main', main_frame)

        # 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()