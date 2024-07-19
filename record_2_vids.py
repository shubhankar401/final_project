import cv2

# Define the video capture objects for two cameras
i = 0
cap1 = cv2.VideoCapture(i)  # Device index 0 for the first camera
cap2 = cv2.VideoCapture(i+4)  # Device index 1 for the second camera

# Check if the cameras are opened correctly
if not cap1.isOpened():
    print("Error: Could not open video device 0.")
    exit()

if not cap2.isOpened():
    print("Error: Could not open video device 1.")
    exit()

# Get the default video frame width and height for both cameras
frame_width1 = int(cap1.get(3))
frame_height1 = int(cap1.get(4))

frame_width2 = int(cap2.get(3))
frame_height2 = int(cap2.get(4))

# Define the codec and create VideoWriter objects for both cameras
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out1 = cv2.VideoWriter('output1.avi', fourcc, 20.0, (frame_width1, frame_height1))
out2 = cv2.VideoWriter('output2.avi', fourcc, 20.0, (frame_width2, frame_height2))

print("Recording... Press 'q' to stop and save the videos.")

while True:
    ret1, frame1 = cap1.read()  # Capture frame-by-frame from the first camera
    ret2, frame2 = cap2.read()  # Capture frame-by-frame from the second camera

    if not ret1:
        print("Error: Failed to capture image from camera 0.")
        break

    if not ret2:
        print("Error: Failed to capture image from camera 1.")
        break

    out1.write(frame1)  # Write the frame from the first camera into the file 'output1.avi'
    out2.write(frame2)  # Write the frame from the second camera into the file 'output2.avi'

    # Display the resulting frames
    cv2.imshow('Camera 1', frame1)
    cv2.imshow('Camera 2', frame2)

    # Press 'q' on the keyboard to stop recording
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and writer objects
cap1.release()
cap2.release()
out1.release()
out2.release()
cv2.destroyAllWindows()

print("Videos saved as output1.avi and output2.avi")
