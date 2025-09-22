# https://medium.com/@nsidana123/real-time-pose-tracking-with-mediapipe-a-comprehensive-guide-for-fitness-applications-series-2-731b1b0b8f4d
#https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/index#models

# from PIL import Image, ImageDraw
import mediapipe as mp
# from mediapipe.tasks import python
# from mediapipe.tasks.python import vision
import cv2
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.3, min_tracking_confidence=0.1) as pose:
    while cap.isOpened():
        ret, Image = cap.read()
        Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
        Image.flags.writeable = False
        results = pose.process(Image)
        Image.flags.writeable = True
        Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(Image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                          mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=8, circle_radius=4),
                          mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=4, circle_radius=2))

        cv2.imshow("Mediapipe pose detection", Image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()


