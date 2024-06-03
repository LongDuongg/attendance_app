from flask import Blueprint, render_template, Response

import pickle
import cv2
import numpy as np

import tensorflow as tf
from recognition_models.recognition_models import FACE_DETECTION_MODEL, FACE_CLS_MODEL


video_stream_Controller = Blueprint(
    "video_stream_Controller", __name__, template_folder="../Views"
)


def accessCamera():
    return cv2.VideoCapture(0)


# map name with number
with open("src/ResultsMap.pkl", "rb") as file:
    result_map = pickle.load(file)


NAME_DICT = {0: "Long", 1: "Phuc", 2: "Quoc"}


# load model
face_tracker = FACE_DETECTION_MODEL
face_recognition = FACE_CLS_MODEL


def stream():
    vid = accessCamera()

    scale = 0.25
    size = 450
    img_size = [size, size]

    while True:
        print("while")
        result, frame = vid.read()
        if result is False:
            break

        # frame = frame[50:500, 50:500, :]
        frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = tf.image.resize(rgb, (120, 120))

        yhat = face_tracker.predict(np.expand_dims(resized / 255, 0))
        sample_coords = yhat[1][0]
        print(yhat)

        if yhat[0] > 0.5:
            # Control the main rectangle
            cv2.rectangle(
                frame,
                tuple(np.multiply(sample_coords[:2], img_size).astype(int)),
                tuple(np.multiply(sample_coords[2:], img_size).astype(int)),
                (255, 0, 0),
                2,
            )

            # capture error if frame is broken
            try:
                cropped = cv2.resize(
                    frame[
                        np.multiply(sample_coords[1], size)
                        .astype(int) : np.multiply(sample_coords[3], size)
                        .astype(int),
                        np.multiply(sample_coords[0], size)
                        .astype(int) : np.multiply(sample_coords[2], size)
                        .astype(int),
                    ],
                    (120, 120),
                )

                print(type(cropped))
                print(cropped)
                # name = face_recognition.predict(np.expand_dims(resized / 255, 0))
                # pred_name = name
                name = face_recognition.predict(cropped, conf=0.8)
                # name = "long"
                # print(name[0].probs)
                print(name[0].probs.top1conf.item())
                pred_name = name[0].probs.top1
                # conf is of tensor() type so do .item()
                pred_conf = name[0].probs.top1conf.item()
                text = NAME_DICT[pred_name] if pred_conf > 0.9 else "Unknown"

                # Control the text rendered
                cv2.putText(
                    frame,
                    # result_map[np.argmax(pred_name)],
                    text,
                    tuple(
                        np.add(
                            np.multiply(sample_coords[:2], img_size).astype(int),
                            [0, -5],
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            except Exception as e:
                print(str(e))

        ret, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")


@video_stream_Controller.route("/index")
def index():
    return render_template("index.html")


@video_stream_Controller.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
