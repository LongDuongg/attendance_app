from tensorflow import keras
from keras.models import load_model
from ultralytics import YOLO
import os

cwd = os.getcwd()
print(cwd)

# FACE_RECOGNITION_MODEL = load_model("src/recognition_models/Face_Recognition_new.keras")

FACE_DETECTION_MODEL = load_model("src/recognition_models/Quoc.h5")
# FACE_DETECTION_MODEL = YOLO("src/recognition_models/yolov8-face.pt")

# FACE_CLS_MODEL = load_model("src/recognition_models/Face_Recognition_new.keras")
# FACE_CLS_MODEL = YOLO("src/recognition_models/yolov8-face-cls.pt")
