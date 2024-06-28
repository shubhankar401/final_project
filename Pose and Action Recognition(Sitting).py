import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Function to calculate angle between three points
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    return np.degrees(angle)

# Function to detect sitting pose based on angles
def detect_sitting_pose(joints):
    # Calculate angles
    left_hip_angle = calculate_angle(joints[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
                                     joints[mp_pose.PoseLandmark.LEFT_HIP.value],
                                     joints[mp_pose.PoseLandmark.LEFT_KNEE.value])
    
    right_hip_angle = calculate_angle(joints[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
                                      joints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                      joints[mp_pose.PoseLandmark.RIGHT_KNEE.value])
    
    left_knee_angle = calculate_angle(joints[mp_pose.PoseLandmark.LEFT_HIP.value],
                                      joints[mp_pose.PoseLandmark.LEFT_KNEE.value],
                                      joints[mp_pose.PoseLandmark.LEFT_ANKLE.value])
    
    right_knee_angle = calculate_angle(joints[mp_pose.PoseLandmark.RIGHT_HIP.value],
                                       joints[mp_pose.PoseLandmark.RIGHT_KNEE.value],
                                       joints[mp_pose.PoseLandmark.RIGHT_ANKLE.value])
    
    # Criteria for sitting pose
    if 60 < left_hip_angle < 120 and 60 < right_hip_angle < 120 and 60 < left_knee_angle < 120 and 60 < right_knee_angle < 120:
        return "Sitting"

# Capture video from webcam
cap = cv2.VideoCapture(0)
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = pose.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Render detections and calculate angles if landmarks are detected
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = results.pose_landmarks.landmark
            joints = [(lm.x, lm.y, lm.z) for lm in landmarks]  # Normalized coordinates

            # Detect sitting pose
            pose_name = detect_sitting_pose(joints)

            # Display the pose name on the video feed
            cv2.putText(image, pose_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        # Display the resulting frame
        cv2.imshow('Pose Detection', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()
