import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)

cap = cv2.VideoCapture(0)

def is_lip_moving(landmarks, width, height, thresh=15):
    # Landmarks for lips
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]

    upper_lip_y = upper_lip.y * height
    lower_lip_y = lower_lip.y * height

    lip_dist = abs(upper_lip_y - lower_lip_y)

    # You can print this value to help adjust threshold
    print(f"Lip distance: {lip_dist:.2f}")

    return lip_dist > thresh  # Adjusted threshold for webcam use

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark

            x_coords = [lm.x * width for lm in landmarks]
            y_coords = [lm.y * height for lm in landmarks]
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))

            lips_moving = is_lip_moving(landmarks, width, height)

            color = (0, 255, 0) if lips_moving else (0, 0, 255)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)

            text = "YES" if lips_moving else "NO"
            cv2.putText(frame, text, (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    scale_percent = 75
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    cv2.imshow("Lip Movement Detection", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
