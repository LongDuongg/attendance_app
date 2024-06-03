from tensorflow import keras
from keras.models import load_model
from ultralytics import YOLO
import os

cwd = os.getcwd()
print(cwd)


FACE_DETECTION_MODEL = load_model("src/recognition_models/Face_Detection.h5")

FACE_CLS_MODEL = YOLO("src/recognition_models/yolov8-face-cls.pt")
