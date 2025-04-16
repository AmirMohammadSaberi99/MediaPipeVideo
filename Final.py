import cv2
import mediapipe as mp
import numpy as np
import math

# Distance threshold (in pixels) for matching face centroids between frames.
DIST_THRESHOLD = 50


def euclidean_distance(pt1, pt2):
    return math.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def get_lip_info(landmarks, width, height, thresh=1):
    # Use landmarks 13 and 14 for the upper and lower lip.
    upper_lip = landmarks[13]
    lower_lip = landmarks[14]
    upper_lip_y = upper_lip.y * height
    lower_lip_y = lower_lip.y * height
    lip_dist = abs(upper_lip_y - lower_lip_y)
    is_active = lip_dist > thresh
    # Print the lip distance for debugging/threshold adjustment.
    print(f"Lip distance: {lip_dist:.2f}")
    return lip_dist, is_active


# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=5)
cap = cv2.VideoCapture(0)

# Dictionary to maintain persistent tracking.
# Keys are persistent IDs and values are dictionaries holding detection info.
tracked_faces = {}
next_face_id = 1  # Next available persistent ID

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    height, width, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    # List to collect detections from the current frame.
    current_detections = []
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = face_landmarks.landmark
            # Compute bounding box.
            x_coords = [lm.x * width for lm in landmarks]
            y_coords = [lm.y * height for lm in landmarks]
            x_min, y_min = int(min(x_coords)), int(min(y_coords))
            x_max, y_max = int(max(x_coords)), int(max(y_coords))
            # Compute centroid of the bounding box.
            cx = (x_min + x_max) // 2
            cy = (y_min + y_max) // 2
            # Get lip movement info.
            lip_dist, is_active = get_lip_info(landmarks, width, height, thresh=1)
            detection = {
                'bbox': (x_min, y_min, x_max, y_max),
                'centroid': (cx, cy),
                'landmarks': landmarks,
                'lip_dist': lip_dist,
                'active': is_active
            }
            current_detections.append(detection)

    # ------------------
    # Matching detections to previously tracked faces.
    # We try to match each new detection with a stored centroid (within DIST_THRESHOLD).
    new_tracked = {}
    used_ids = set()
    for detection in current_detections:
        centroid = detection['centroid']
        best_match_id = None
        best_distance = float('inf')
        for tid, info in tracked_faces.items():
            # Compute distance between current detection and tracked face.
            dist = euclidean_distance(centroid, info['centroid'])
            if dist < DIST_THRESHOLD and dist < best_distance and tid not in used_ids:
                best_distance = dist
                best_match_id = tid
        if best_match_id is not None:
            detection['id'] = best_match_id
            used_ids.add(best_match_id)
        else:
            # No match found; assign new persistent ID.
            detection['id'] = next_face_id
            used_ids.add(next_face_id)
            next_face_id += 1
        new_tracked[detection['id']] = detection
    tracked_faces = new_tracked  # Update the tracker

    # ------------------
    # Assign display labels for each tracked face.
    # If at least one face is active, assign label #1 to the one with the highest lip_dist.
    detections_list = list(tracked_faces.values())
    label_assignment = {}  # Mapping: persistent id -> display label
    active_detections = [d for d in detections_list if d['active']]
    if active_detections:
        active_face = max(active_detections, key=lambda d: d['lip_dist'])
        label_assignment[active_face['id']] = 1
        others = [d for d in detections_list if d['id'] != active_face['id']]
        others_sorted = sorted(others, key=lambda d: d['id'])
        label = 2
        for d in others_sorted:
            label_assignment[d['id']] = label
            label += 1
    else:
        # If no face is active, assign labels based on persistent ID order.
        sorted_detections = sorted(detections_list, key=lambda d: d['id'])
        label = 1
        for d in sorted_detections:
            label_assignment[d['id']] = label
            label += 1

    # ------------------
    # Draw bounding boxes and labels.
    for detection in detections_list:
        x_min, y_min, x_max, y_max = detection['bbox']
        pid = detection['id']
        display_label = label_assignment.get(pid, 0)
        # Use green if active, otherwise red.
        color = (0, 255, 0) if detection['active'] else (0, 0, 255)
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, 2)
        status_text = "YES" if detection['active'] else "NO"
        cv2.putText(frame, status_text, (x_min, y_min - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        # Draw the display label above the bounding box.
        label_text = f"#{display_label}"
        cv2.putText(frame, label_text, (x_min, y_min - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Resize the frame for display.
    scale_percent = 75
    new_width = int(width * scale_percent / 100)
    new_height = int(height * scale_percent / 100)
    resized_frame = cv2.resize(frame, (new_width, new_height))

    cv2.imshow("Lip Movement Detection", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
