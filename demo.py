import cv2
import mediapipe as mp
import numpy as np
import time

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, image = cap.read()
    if not success:
        break

    start = time.time()

    # Flip the image horizontally for a later selfie-view display
    # Also convert the color space from BGR to RGB
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # To improve performance
    image.flags.writeable = False
    
    # Get the number of faces in the frame
    with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=2) as face_mesh:
        
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            num_faces = len(results.multi_face_landmarks)
        else:
            num_faces = 0

    print("num_faces: ", num_faces)
    # Initialize FaceMesh instances for each face
    face_meshes = []
    for _ in range(num_faces):
        face_meshes.append(mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5, max_num_faces=1))

    # To improve performance
    image.flags.writeable = True
    
    # Convert the color space from RGB to BGR
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    img_h, img_w, img_c = image.shape

    # Process results for each face
    for idx, face_mesh in enumerate(face_meshes):
        results = face_mesh.process(image)
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                face_3d = []
                face_2d = []
                for lm in face_landmarks.landmark:
                    x, y = int(lm.x * img_w), int(lm.y * img_h)

                    # Get the 2D Coordinates
                    face_2d.append([x, y])

                    # Get the 3D Coordinates
                    face_3d.append([x, y, lm.z])       
                
                # Convert it to the NumPy array
                face_2d = np.array(face_2d, dtype=np.float64)

                # Convert it to the NumPy array
                face_3d = np.array(face_3d, dtype=np.float64)

                # The camera matrix
                focal_length = 1 * img_w

                cam_matrix = np.array([[focal_length, 0, img_w / 2],
                                       [0, focal_length, img_h / 2],
                                       [0, 0, 1]])

                # The distortion parameters
                dist_matrix = np.zeros((4, 1), dtype=np.float64)

                # Solve PnP
                success, rot_vec, trans_vec = cv2.solvePnP(face_3d, face_2d, cam_matrix, dist_matrix)

                if success:
                    # Get rotational matrix
                    rmat, _ = cv2.Rodrigues(rot_vec)

                    # Get angles
                    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)

                    # Get the y rotation degree
                    x = angles[0] * 360
                    y = angles[1] * 360

                    # See where the user's head tilting
                    if y < -5:
                        text = "Looking Left"
                    elif y > 5:
                        text = "Looking Right"
                    elif x < -2:
                        text = "Looking Down"
                    elif x > 5:
                        text = "Looking Up"
                    else:
                        text = "Forward"

                    # Display the nose direction
                    nose_3d = np.array([[lm.x * img_w, lm.y * img_h, lm.z * 3000] for lm in face_landmarks.landmark if lm.z != 0])
                    nose_3d_projection, _ = cv2.projectPoints(nose_3d, rot_vec, trans_vec, cam_matrix, dist_matrix)

                    p1 = (int(nose_3d_projection[0][0][0]), int(nose_3d_projection[0][0][1]))
                    p2 = (int(nose_3d_projection[0][0][0] + y * 10), int(nose_3d_projection[0][0][1] - x * 10))
                    
                    cv2.line(image, p1, p2, (255, 0, 0), 3)

                    # Add the text on the image
                    cv2.putText(image, text, (p1[0], p1[1] - 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.putText(image, "x: " + str(np.round(x, 2)), (p1[0], p1[1] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(image, "y: " + str(np.round(y, 2)), (p1[0], p1[1] - 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Draw the landmarks
                mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    landmark_drawing_spec=drawing_spec,
                    connection_drawing_spec=drawing_spec)

    end = time.time()
    totalTime = end - start

    fps = 1 / totalTime
    cv2.putText(image, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

    cv2.imshow('Head Pose Estimation', image)

    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
