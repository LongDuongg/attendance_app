import cv2
import os
import numpy as np
from recognition_models import FACE_CLS_MODEL, FACE_DETECTION_MODEL

# FACE_DETECTION_MODEL.summary()
# FACE_CLS_MODEL.summary()


def load_images_from_folder(folder_path):
    images = []

    for img_path in os.listdir(folder_path):
        images.append(cv2.imread(os.path.join(folder_path, img_path)))

    return images


def test_yolov8_model():
    images = load_images_from_folder("src/recognition_models/test_images")

    results = FACE_CLS_MODEL.predict(images)

    for result in results:
        print(result)



def test_long_model():
    images = load_images_from_folder("src/recognition_models/test_images")

    for img in images:
        img = cv2.resize(img, (120,120))
        img = np.expand_dims(img,axis=0)
        
        result = FACE_CLS_MODEL.predict(img) 
        
        print('####'*10)
        print(result)
        print('Prediction is: ', np.argmax(result))



test_long_model()
