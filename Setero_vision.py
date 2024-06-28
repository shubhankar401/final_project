import cv2
import mediapipe as mp
import numpy as np
import dlib
import face_recognition
from collections import deque
from scipy.spatial import distance as dist

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Initialize face landmarks predictor
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Queue to store landmarks
landmarks_queue = deque(maxlen=10)

# Function to detect faces and draw bounding boxes
def detect_faces_and_draw_boxes(frame, face_id, trackers):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    new_trackers = []
    ids_detected = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            cx, cy = x + w // 2, y + h // 2

            # Find if this face matches a previously detected face
            min_dist = float('inf')
            min_id = -1
            for tracker in trackers:
                tracker_id, (tx, ty, tw, th), (tcx, tcy) = tracker
                d = dist.euclidean((cx, cy), (tcx, tcy))
                if d < min_dist:
                    min_dist = d
                    min_id = tracker_id

            if min_dist < 50:  # Threshold to determine if it's the same face
                ids_detected.append(min_id)
                new_trackers.append((min_id, (x, y, w, h), (cx, cy)))
            else:
                ids_detected.append(face_id)
                new_trackers.append((face_id, (x, y, w, h), (cx, cy)))
                face_id += 1

            # Draw a green rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Put a unique ID for each face
            cv2.putText(frame, f'ID: {ids_detected[-1]}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame, face_id, new_trackers

def detect_landmarks_and_compare(frame, landmarks_queue):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    current_landmarks = []

    for (top, right, bottom, left) in face_locations:
        rect = dlib.rectangle(left, top, right, bottom)
        shape = predictor(rgb_frame, rect)
        landmarks = np.array([(p.x, p.y) for p in shape.parts()])
        current_landmarks.append(landmarks)

        for (x, y) in landmarks:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    similarity_ratios = []
    if current_landmarks:
        for stored_landmarks in landmarks_queue:
            similarities = []
            for landmarks in current_landmarks:
                similarity = np.mean([dist.euclidean(l1, l2) for l1, l2 in zip(stored_landmarks, landmarks)])
                similarities.append(similarity)
            similarity_ratios.append(np.mean(similarities))

    return frame, similarity_ratios

# Open webcam
cap = cv2.VideoCapture(0)
frames_with_faces = []
face_id = 0
trackers = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_with_boxes, face_id, trackers = detect_faces_and_draw_boxes(frame, face_id, trackers)
    
    # Detect landmarks and compare
    frame_with_landmarks, similarity_ratios = detect_landmarks_and_compare(frame_with_boxes, landmarks_queue)
    
    # Display similarity ratios
    if similarity_ratios:
        for i, similarity in enumerate(similarity_ratios):
            cv2.putText(frame_with_landmarks, f'Similarity: {similarity:.2f}', (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    
    # Save the frame with detected faces to the list
    frames_with_faces.append(frame_with_landmarks)

    # Display the resulting frame
    cv2.imshow("Frame", frame_with_landmarks)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # Save the current frame as a screenshot
        cv2.imwrite('screenshot.png', frame_with_landmarks)
        # Extract and store landmarks of the current frame
        face_locations = face_recognition.face_locations(frame)
        for (top, right, bottom, left) in face_locations:
            rect = dlib.rectangle(left, top, right, bottom)
            shape = predictor(frame, rect)
            landmarks = np.array([(p.x, p.y) for p in shape.parts()])
            landmarks_queue.append(landmarks)

# Convert the list of frames to a NumPy array
frames_array = np.array(frames_with_faces)

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

# Save the numpy array to a file
np.save('frames_with_faces.npy', frames_array)
