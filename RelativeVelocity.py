import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Objectron.
mp_objectron = mp.solutions.objectron
objectron = mp_objectron.Objectron(static_image_mode=False,
                                   max_num_objects=1,
                                   min_detection_confidence=0.5,
                                   min_tracking_confidence=0.5,
                                   model_name='Cup')  # You can change the model as needed.

# Initialize variables for calculating velocity.
previous_center = None
previous_time = None

# Open the webcam.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the BGR image to RGB.
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with MediaPipe Objectron.
    results = objectron.process(rgb_frame)

    # Get the current time.
    current_time = cv2.getTickCount()

    if results.detected_objects:
        for detected_object in results.detected_objects:
            # Get the bounding box and key points.
            bbox = detected_object.location_data.relative_bounding_box
            keypoints = detected_object.keypoints

            # Calculate the center of the bounding box.
            center_x = int(bbox.xmin * frame.shape[1] + bbox.width * frame.shape[1] / 2)
            center_y = int(bbox.ymin * frame.shape[0] + bbox.height * frame.shape[0] / 2)
            current_center = (center_x, center_y)

            # Draw the bounding box and center point.
            cv2.rectangle(frame, (int(bbox.xmin * frame.shape[1]), int(bbox.ymin * frame.shape[0])),
                          (int((bbox.xmin + bbox.width) * frame.shape[1]), int((bbox.ymin + bbox.height) * frame.shape[0])),
                          (0, 255, 0), 2)
            cv2.circle(frame, current_center, 5, (0, 0, 255), -1)

            # Calculate the relative velocity.
            if previous_center is not None and previous_time is not None:
                time_elapsed = (current_time - previous_time) / cv2.getTickFrequency()
                distance = np.linalg.norm(np.array(current_center) - np.array(previous_center))
                velocity = distance / time_elapsed

                # Display the velocity.
                cv2.putText(frame, f'Velocity: {velocity:.2f} pixels/sec', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)

            # Update the previous center and time.
            previous_center = current_center
            previous_time = current_time

    # Display the frame.
    cv2.imshow('Object Tracking', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows.
cap.release()
cv2.destroyAllWindows()
