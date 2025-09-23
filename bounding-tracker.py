# https://medium.com/@nsidana123/real-time-pose-tracking-with-mediapipe-a-comprehensive-guide-for-fitness-applications-series-2-731b1b0b8f4d
#https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/index#models
# https://www.geeksforgeeks.org/python/python-opencv-cv2-rectangle-method/
# https://stackoverflow.com/questions/77350434/detecting-hands-or-body-using-running-mode-video-live-stream-mediapipe
import numpy as np
import cv2
import pyrealsense2 as rs
import time

from PIL import Image, ImageDraw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2, os
import numpy as np

cwd = os.getcwd()

model_path = cwd + '/efficientdet_lite2.tflite'

BaseOptions = mp.tasks.BaseOptions
ObjectDetector = mp.tasks.vision.ObjectDetector
ObjectDetectorOptions = mp.tasks.vision.ObjectDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

cap = cv2.VideoCapture(0)

class Detection():

    def __init__(self, model_path) -> None:
        self.annotated_image = np.zeros((640, 480, 3), np.uint8)

        options = vision.ObjectDetectorOptions(
            base_options = BaseOptions(model_asset_path = model_path),
            max_results = 5,
            running_mode = VisionRunningMode.LIVE_STREAM,
            result_callback = self.draw_result
            )
        self.detector = ObjectDetector.create_from_options(options)

        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        self.pipeline.start(config)

        
def run(self):
    try:
        self.detectObjects()
    finally:
        # stop streaming
        self.pipeline.stop()
    
def draw_result(self, detection_result, rgb_image, _):
    detection_list = detection_result.detections
    rgb_image = rgb_image.numpy_view()
    annotated_image = np.copy(rgb_image)

    #loop through detections
    for i in range(len(detection_list)):
        detection = detection_list[i]

        box = detection.bounding_box

        box_x = box.origin_x
        box_y = box.origin_y
        box_width = box.width
        box_height = box.height

        #name these better
        left = box_x
        bottom = box_y #actually the upper part
        right = left + box_width
        top = bottom + box_height #actually the bottom

        cv2.rectangle(annotated_image, (left,bottom), (right, top), (255,255,255), 5)
    
    self.annotated_image = annotated_image
    return

def detectObjects(self):
    while True:
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

def main():

    #main entry point for the node
    detection = Detection()

    try:
        #run control loop
        while True:
            ret, frame = cap.read()
            detection.run()
    except KeyboardInterrupt:
        exit()


if __name__ == "main":
    main()


"""
import time
import cv2
import numpy as np
import mediapipe as mp
# for visualizing results
from mediapipe.framework.formats import landmark_pb2


class landmarker_and_result():
   def __init__(self):
      self.result = mp.tasks.vision.HandLandmarkerResult
      self.landmarker = mp.tasks.vision.HandLandmarker
      self.createLandmarker()
   
   def createLandmarker(self):
      # callback function
      def update_result(result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
         self.result = result

      # HandLandmarkerOptions (details here: https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/python#live-stream)
      options = mp.tasks.vision.HandLandmarkerOptions( 
         base_options = mp.tasks.BaseOptions(model_asset_path="hand_landmarker.task"), # path to model
         running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM, # running on a live stream
         num_hands = 2, # track both hands
         min_hand_detection_confidence = 0.3, # lower than value to get predictions more often
         min_hand_presence_confidence = 0.3, # lower than value to get predictions more often
         min_tracking_confidence = 0.3, # lower than value to get predictions more often
         result_callback=update_result)
      
      # initialize landmarker
      self.landmarker = self.landmarker.create_from_options(options)
   
   def detect_async(self, frame):
      # convert np frame to mp image
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # detect landmarks
      self.landmarker.detect_async(image = mp_image, timestamp_ms = int(time.time() * 1000))

   def close(self):
      # close landmarker
      self.landmarker.close()

def draw_landmarks_on_image(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Courtesy of https://github.com/googlesamples/mediapipe/blob/main/examples/hand_landmarker/python/hand_landmarker.ipynb"""
   try:
      if detection_result.hand_landmarks == []:
         return rgb_image
      else:
         hand_landmarks_list = detection_result.hand_landmarks
         handedness_list = detection_result.handedness
         annotated_image = np.copy(rgb_image)

         # Loop through the detected hands to visualize.
         for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
               landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks])
            mp.solutions.drawing_utils.draw_landmarks(
               annotated_image,
               hand_landmarks_proto,
               mp.solutions.hands.HAND_CONNECTIONS,
               mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
               mp.solutions.drawing_styles.get_default_hand_connections_style())

         return annotated_image
   except:
      return rgb_image

def count_fingers_raised(rgb_image, detection_result: mp.tasks.vision.HandLandmarkerResult):
   """Iterate through each hand, checking if fingers (and thumb) are raised.
   Hand landmark enumeration (and weird naming convention) comes from
   https://developers.google.com/mediapipe/solutions/vision/hand_landmarker."""
   try:
      # Get Data
      hand_landmarks_list = detection_result.hand_landmarks
      # Counter
      numRaised = 0
      # for each hand...
      for idx in range(len(hand_landmarks_list)):
         # hand landmarks is a list of landmarks where each entry in the list has an x, y, and z in normalized image coordinates
         hand_landmarks = hand_landmarks_list[idx]
         # for each fingertip... (hand_landmarks 4, 8, 12, and 16)
         for i in range(8,21,4):
            # make sure finger is higher in image the 3 proceeding values (2 finger segments and knuckle)
            tip_y = hand_landmarks[i].y
            dip_y = hand_landmarks[i-1].y
            pip_y = hand_landmarks[i-2].y
            mcp_y = hand_landmarks[i-3].y
            if tip_y < min(dip_y,pip_y,mcp_y):
               numRaised += 1
         # for the thumb
         # use direction vector from wrist to base of thumb to determine "raised"
         tip_x = hand_landmarks[4].x
         dip_x = hand_landmarks[3].x
         pip_x = hand_landmarks[2].x
         mcp_x = hand_landmarks[1].x
         palm_x = hand_landmarks[0].x
         if mcp_x > palm_x:
            if tip_x > max(dip_x,pip_x,mcp_x):
               numRaised += 1
         else:
            if tip_x < min(dip_x,pip_x,mcp_x):
               numRaised += 1
         
         
      # display number of fingers raised on the image
      annotated_image = np.copy(rgb_image)
      height, width, _ = annotated_image.shape
      text_x = int(hand_landmarks[0].x * width) - 100
      text_y = int(hand_landmarks[0].y * height) + 50
      cv2.putText(img = annotated_image, text = str(numRaised) + " Fingers Raised",
                        org = (text_x, text_y), fontFace = cv2.FONT_HERSHEY_DUPLEX,
                        fontScale = 1, color = (0,0,255), thickness = 2, lineType = cv2.LINE_4)
      return annotated_image
   except:
      return rgb_image

def main():
   # access webcam
   cap = cv2.VideoCapture(0)

   # create landmarker
   hand_landmarker = landmarker_and_result()

   while True:
      # pull frame
      ret, frame = cap.read()
      # mirror frame
      frame = cv2.flip(frame, 1)
      # update landmarker results
      hand_landmarker.detect_async(frame)
      # draw landmarks on frame
      frame = draw_landmarks_on_image(frame,hand_landmarker.result)
      # count number of fingers raised
      frame = count_fingers_raised(frame,hand_landmarker.result)
      # display image
      cv2.imshow('frame',frame)
      if cv2.waitKey(1) == ord('q'):
         break
   
   # release everything
   hand_landmarker.close()
   cap.release()
   cv2.destroyAllWindows()

if __name__ == "__main__":
   main()

"""



















# cap = cv2.VideoCapture(0)

# with ObjectDetector.create_from_options(options) as detector:
#     while cap.isOpened():
#         ret, Image = cap.read()
#         Image = cv2.cvtColor(Image, cv2.COLOR_BGR2RGB)
#         Image.flags.writeable = False
#         Image.flags.writeable = True
#         Image = cv2.cvtColor(Image, cv2.COLOR_RGB2BGR)

#         detection_result = detector.detect(cap)
#         detection = detection_result.detections[0]
#         box = detection.bounding_box

#         box_x = box.origin_x
#         box_y = box.origin_y
#         box_width = box.width
#         box_height = box.height

#         #name these better
#         left = box_x
#         bottom = box_y #actually the upper part
#         right = left + box_width
#         top = bottom + box_height #actually the bottom

#         cv2.rectangle(cap, (left,bottom), (right, top), (255,255,255), 5)

#         cv2.imshow("Mediapipe subject detection", Image)

#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break


# cap.release()
# cv2.destroyAllWindows()




