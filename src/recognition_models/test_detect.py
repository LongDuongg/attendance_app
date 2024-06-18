import cv2
import os
import numpy as np
import tensorflow as tf
from recognition_models import FACE_DETECTION_MODEL
from utils import crop_square
from ultralytics.utils.plotting import Annotator


def performFaceDetectionYolov8(vid):
    facetracker = FACE_DETECTION_MODEL

    size = 450
    img_size = [size, size]

    while True:
        result, frame = vid.read()

        # crop and resize frame to size 450
        frame = crop_square(frame, size)

        if result is False:
            break

        results = facetracker.predict(frame)

        for r in results:
            annotator = Annotator(frame)
            boxes = r.boxes

            for box in boxes:
                conf = box.conf[0].item()
                print(conf)

                if conf > 0.5:
                    b = box.xyxy[
                        0
                    ]  # get box coordinates in (left, top, right, bottom) format
                    c = box.cls
                    annotator.box_label(b, facetracker.names[int(c)])

        frame = annotator.result()

        cv2.imshow("My Face Detection Project", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


def performFaceDetection(vid):
    facetracker = FACE_DETECTION_MODEL

    size = 450
    img_size = [size, size]

    while True:
        result, frame = vid.read()
        if result is False:
            break

        # crop and resize frame to size 450
        frame = crop_square(frame, size)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = facetracker.predict(np.expand_dims(resized / 255, 0))
        print(yhat)
        sample_coords = yhat[1][0]

        if yhat[0] > 0.5:
            # Control the main rectangle
            cv2.rectangle(
                frame,
                tuple(np.multiply(sample_coords[:2], img_size).astype(int)),
                tuple(np.multiply(sample_coords[2:], img_size).astype(int)),
                (255, 0, 0),
                2,
            )

        cv2.imshow("My Face Detection Project", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


def accessCamera(IP_Stream):
    return cv2.VideoCapture(IP_Stream)


# video_stream = accessCamera(0)
# performFaceDetectionYolov8(video_stream)
