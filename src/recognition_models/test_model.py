import cv2
import os
from recognition_models import FACE_CLS_MODEL, FACE_DETECTION_MODEL

# FACE_DETECTION_MODEL.summary()


def load_images_from_folder(folder_path):
    images = []

    for img_path in os.listdir(folder_path):
        images.append(cv2.imread(os.path.join(folder_path, img_path)))

    return images


images = load_images_from_folder("src/recognition_models/test_images")

results = FACE_CLS_MODEL.predict(images)

for result in results:
    print(result)
