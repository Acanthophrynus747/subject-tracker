# https://medium.com/@nsidana123/real-time-pose-tracking-with-mediapipe-a-comprehensive-guide-for-fitness-applications-series-2-731b1b0b8f4d
#https://ai.google.dev/edge/mediapipe/solutions/vision/object_detector/index#models

from PIL import Image, ImageDraw
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

