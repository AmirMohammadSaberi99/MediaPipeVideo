import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
# Specify the maximum number of faces you expect per frame
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)  # You can set this number higher if needed

cap = cv2.VideoCapture("video.mp4")

while True:
    ret, image = cap.read()
    if not ret:
        break

    height, width, _ = image.shape
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Facial landmark detection
    result = face_mesh.process(rgb_image)

    if result.multi_face_landmarks:
        for facial_landmarks in result.multi_face_landmarks:
            for landmark in facial_landmarks.landmark:
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                cv2.circle(image, (x, y), 1, (0, 255, 0), -1)

    # Resize for display
    scale_percent = 50
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_image = cv2.resize(image, (new_width, new_height))

    cv2.imshow("Face Mesh - Multiple Faces", resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
