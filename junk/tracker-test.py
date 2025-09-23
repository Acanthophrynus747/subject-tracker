import numpy as np
import cv2
import pyrealsense2 as rs
import time

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

#

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

class Hands_detection():
   
  def __init__(self, model_path) -> None:
    # Initialization of the image
    self.annotated_image = np.zeros((640,480,3), np.uint8)

    # Create an HandLandmarker object
    VisionRunningMode = mp.tasks.vision.RunningMode
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=model_path_full),
                                        running_mode=VisionRunningMode.LIVE_STREAM,
                                        result_callback=self.draw_hand_landmarks_on_live,
                                        num_hands=2,
                                        min_hand_detection_confidence=0.5,
                                        min_hand_presence_confidence=0.5,
                                        min_tracking_confidence=0.5)
    self.detector = vision.HandLandmarker.create_from_options(options)

    # Configure depth and color streams
    self.pipeline = rs.pipeline()
    config = rs.config() 

    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    self.pipeline.start(config)

  #

  def run(self):
    try:
      self.detect_body()
    finally:
      # Stop streaming
      self.pipeline.stop()

  #

  def draw_hand_landmarks_on_live(self, detection_result, rgb_image, _):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    rgb_image = rgb_image.numpy_view()
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
      hand_landmarks = hand_landmarks_list[idx]
      handedness = handedness_list[idx]

      # Draw the hand landmarks.
      hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
      hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
      ])
      solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

      # Get the top left corner of the detected hand's bounding box.
      height, width, _ = annotated_image.shape
      x_coordinates = [landmark.x for landmark in hand_landmarks]
      y_coordinates = [landmark.y for landmark in hand_landmarks]
      text_x = int(min(x_coordinates) * width)
      text_y = int(min(y_coordinates) * height) - MARGIN

      # Draw handedness (left or right hand) on the image.
      cv2.putText(annotated_image, f"{handedness[0].category_name}",
                  (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                  FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    self.annotated_image = annotated_image
    return

  #
  
  def detect_body(self):
    while True:
      # Wait for a coherent pair of frames: depth and color
      frames = self.pipeline.wait_for_frames()
      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()
      if not depth_frame or not color_frame:
          continue     

      color_image = np.array(color_frame.get_data())
      timestamp = int(round(time.time()*1000))

      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
      detection_result = self.detector.detect_async(mp_image, timestamp)

      cv2.imshow('Hand detection', self.annotated_image)
      cv2.waitKey(1)

      if cv2.waitKey(1) & 0xFF == ord('q'):
          break
