import mediapipe as mp
import pandas as pd
import pickle
import cv2


# ------- Initialization -------- #
mp_drawing = mp.solutions.drawing_utils 
mp_pose = mp.solutions.pose 
pose = mp_pose.Pose()

with open('ridge_model_5_yoga_poses.pkl', 'rb') as f:
    model = pickle.load(f)

cap = cv2.VideoCapture(0)

while True:
    # Initialize Frame
    ret, frame = cap.read()

    if not ret:
        break

    # Convert the BGR image to RGB                          not required in current versions off openCV
    # rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame = frame

    # Get all
    results = pose.process(rgb_frame)
    # Extract landmarks if available
    if results.pose_landmarks:
        # 3D world coordinates with Hip as origin
        landmarks_world = results.pose_world_landmarks.landmark

        # required landmarks
        nose = landmarks_world[0]
        left_shoulder = landmarks_world[11]
        right_shoulder = landmarks_world[12]
        left_elbow = landmarks_world[13]
        right_elbow = landmarks_world[14]
        left_wrist = landmarks_world[15]
        right_wrist = landmarks_world[16]
        left_hip = landmarks_world[23]
        right_hip = landmarks_world[24]
        left_knee = landmarks_world[25]
        right_knee = landmarks_world[26]
        left_ankle = landmarks_world[27]
        right_ankle = landmarks_world[28]

    # input formation
    row = [
        nose.x, nose.y, nose.z, nose.visibility,
        left_shoulder.x, left_shoulder.y, left_shoulder.z, left_shoulder.visibility,
        right_shoulder.x, right_shoulder.y, right_shoulder.z, right_shoulder.visibility,
        left_elbow.x, left_elbow.y, left_elbow.z, left_elbow.visibility,
        right_elbow.x, right_elbow.y, right_elbow.z, right_elbow.visibility,
        left_wrist.x, left_wrist.y, left_wrist.z, left_wrist.visibility,
        right_wrist.x, right_wrist.y, right_wrist.z, right_wrist.visibility,
        left_hip.x, left_hip.y, left_hip.z, left_hip.visibility,
        right_hip.x, right_hip.y, right_hip.z, right_hip.visibility,
        left_knee.x, left_knee.y, left_knee.z, left_knee.visibility,
        right_knee.x, right_knee.y, right_knee.z, right_knee.visibility,
        left_ankle.x, left_ankle.y, left_ankle.z, left_ankle.visibility,
        right_ankle.x, right_ankle.y, right_ankle.z, right_ankle.visibility]
    
    if results.pose_landmarks:
        # Indices for face landmarks
        face_landmark_indices = list(range(0, 11))

        # Ensure we do not go out of bounds
        total_landmarks = len(results.pose_landmarks.landmark)
        
        # Removing face landmarks in vid feed
        for idx in face_landmark_indices:
            if idx < total_landmarks:
                results.pose_landmarks.landmark[idx].visibility = 0

    # Make skeleton wire frame
    mp_drawing.draw_landmarks(rgb_frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2) 
                                 )

    X = pd.DataFrame([row])
    body_language_class = model.predict(X)

    # ------ Screen outputs ---------- #
    print(body_language_class)
    cv2.rectangle(rgb_frame, (0,0), (250, 60), (245, 117, 16), -1)
            
    # Display Class
    cv2.putText(rgb_frame, 'CLASS', (95,12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    # cv2.putText(rgb_frame, body_language_class.split(' ')[0])
    cv2.putText(rgb_frame, str(body_language_class[0]), (90,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                        
    cv2.imshow('Raw Webcam Feed', rgb_frame)

    # exit on 'q'
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()