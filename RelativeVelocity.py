import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Capture video from webcam
cap = cv2.VideoCapture(0)

# Set camera width and height
cap.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

previous_landmarks = None
previous_time = None

def calculate_velocity(current_landmarks, previous_landmarks, time_elapsed):
    velocities = []
    for curr, prev in zip(current_landmarks, previous_landmarks):
        # Calculate the Euclidean distance between the current and previous landmarks
        distance = np.linalg.norm(np.array([curr.x, curr.y, curr.z]) - np.array([prev.x, prev.y, prev.z]))
        # Velocity is distance over time
        velocity = distance / time_elapsed
        velocities.append(velocity)
    return velocities

def recognize_action(velocity):
    if velocity < 0.8:
        return "Sitting/Still"
    elif 0.8 < velocity < 1.2:
        return "Walking/moving"
    elif 1.2 < velocity < 1.6:
        return "Jogging/brisk"
    else:
        return "Running"

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Pose
    result = pose.process(rgb_frame)
    
    if result.pose_landmarks:
        # Get the list of landmark positions
        current_landmarks = result.pose_landmarks.landmark
        
        if previous_landmarks is not None:
            # Calculate the time elapsed between frames
            current_time = cv2.getTickCount() / cv2.getTickFrequency()
            time_elapsed = current_time - previous_time
            previous_time = current_time

            # Calculate velocities
            velocities = calculate_velocity(current_landmarks, previous_landmarks, time_elapsed)
            print(f'Velocities: {velocities}')
            
            # Calculate the overall speed (average of all landmark velocities)
            avg_velocity = np.mean(velocities)
            
            # Calculate the approximate speed in meters per second (assuming a frame rate of 30 FPS and scale)
            speed_m_s = avg_velocity  # This needs calibration for real world speed
            
            # Recognize the action based on speed
            action = recognize_action(speed_m_s)
            
            # Display the speed and action in a separate box
            cv2.putText(frame, f'Speed: {speed_m_s:.2f} m/s', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Action: {action}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        else:
            # Initialize the previous_time for the first frame
            previous_time = cv2.getTickCount() / cv2.getTickFrequency()
        
        # Update the previous landmarks
        previous_landmarks = current_landmarks

        # Draw landmarks on the frame
        mp.solutions.drawing_utils.draw_landmarks(
            frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    # Display the frame
    cv2.imshow('MediaPipe Pose', frame)
    
    if cv2.waitKey(5) & 0xFF == ord('q'):  # Press 'Q' key to exit
        break

cap.release()
cv2.destroyAllWindows()
