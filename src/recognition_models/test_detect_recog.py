import cv2
import os
import numpy as np
import tensorflow as tf
from recognition_models import FACE_DETECTION_MODEL, FACE_CLS_MODEL
from utils import crop_square
from ultralytics.utils.plotting import Annotator

NAME_DICT = {0: "Long", 1: "Phuc", 2: "Quoc"}


def performFaceRecognitionYolov8(vid):
    facetracker = FACE_DETECTION_MODEL
    face_recognition = FACE_CLS_MODEL

    size = 640

    while True:
        result, frame = vid.read()
        print("hereee")

        # crop and resize frame to size 640
        frame = crop_square(frame, size)

        if result is False:
            break

        results = facetracker.predict(frame)

        for r in results:
            annotator = Annotator(frame)
            boxes = r.boxes

            for box in boxes:
                detect_conf = box.conf[0].item()
                print(detect_conf)

                if detect_conf > 0.1:
                    b = box.xyxy[
                        0
                    ]  # get box coordinates in (left, top, right, bottom) format
                    # c = box.cls
                    # annotator.box_label(b, facetracker.names[int(c)])

                    x1, y1, x2, y2 = b
                    cropped = frame[int(y1) : int(y2), int(x1) : int(x2)]

                    pred = face_recognition.predict(cropped, verbose=False)
                    pred_name = pred[0].probs.top1
                    pred_conf = pred[0].probs.top1conf.item()
                    print("aaa", pred_name, pred_conf)
                    text_name = (
                        NAME_DICT[int(pred_name)] if pred_conf > 0.6 else "Unknown"
                    )
                    annotator.box_label(b, text_name)

        frame = annotator.result()

        cv2.imshow("My Face Recognition Project", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


def performFaceRecognition(vid):
    facetracker = FACE_DETECTION_MODEL
    face_recognition = FACE_CLS_MODEL

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
                tuple(np.multiply(sample_coords[:2], [450, 450]).astype(int)),
                tuple(np.multiply(sample_coords[2:], [450, 450]).astype(int)),
                (255, 0, 0),
                2,
            )

            resize = cv2.resize(
                frame[
                    np.multiply(sample_coords[1], 450)
                    .astype(int) : np.multiply(sample_coords[3], 450)
                    .astype(int),
                    np.multiply(sample_coords[0], 450)
                    .astype(int) : np.multiply(sample_coords[2], 450)
                    .astype(int),
                ],
                (120, 120),
            )

            name = face_recognition.predict(np.expand_dims(resize / 255, 0), verbose=0)
            print(name)

            # Control the text rendered
            cv2.putText(
                frame,
                # NAME_DICT[np.argmax(name)],
                np.argmax(name),
                tuple(
                    np.add(
                        np.multiply(sample_coords[:2], [450, 450]).astype(int), [0, -5]
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("My Face Detection Project", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()
    cv2.destroyAllWindows()


def accessCamera(IP_Stream):
    return cv2.VideoCapture(IP_Stream)


video_stream = accessCamera(0)
# performFaceRecognition(video_stream)
performFaceRecognitionYolov8(video_stream)
