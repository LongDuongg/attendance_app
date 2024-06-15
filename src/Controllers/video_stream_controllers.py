from flask import Blueprint, render_template, Response

import pickle
import cv2
import numpy as np
import pandas as pd
from datetime import date

import tensorflow as tf
from recognition_models.recognition_models import FACE_DETECTION_MODEL, FACE_CLS_MODEL
from Models.mock_data import STUDENTS


video_stream_controller = Blueprint(
    "video_stream_controller", __name__, template_folder="../Views"
)


def accessCamera():
    return cv2.VideoCapture(0)


# map name with number
with open("src/ResultsMap.pkl", "rb") as file:
    result_map = pickle.load(file)


NAME_DICT = {0: "Long", 1: "Phuc", 2: "Quoc"}
UNKNOWN = "Unknown"


# load model
face_tracker = FACE_DETECTION_MODEL
face_recognition = FACE_CLS_MODEL
# face_recognition = FACE_RECOGNITION_MODEL


def add_attendance(name):
    if name == UNKNOWN:
        return

    found_student = [s for s in STUDENTS if s["id_name"] == name][0]
    time = date.today().strftime("%d/%m/%Y %H:%M:%S")

    df = pd.read_csv("Attendance.csv", dtype=str, encoding="utf-8")
    found_ids = list(df["id"])

    if found_student["id"] not in found_ids:
        with open("Attendance.csv", "a", encoding='utf-8') as f:
            line = f"\n{found_student['name']},{found_student['id']},{found_student['class']},\"{found_student['address']}\",{found_student['image']},{time}"
            print(line)
            f.write(
                line
            )


def process_frame(frame):
    scale = 0.25
    size = 450
    img_size = [size, size]

    #for windows
    frame = frame[50:500, 50:500, :]
    
    #for MacOS
    # frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)

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

            # print(type(cropped))
            # print(cropped)
            # name = face_recognition.predict(np.expand_dims(resized / 255, 0))
            # pred_name = name
            name = face_recognition.predict(cropped)
            # name = "long"
            # print(name[0].probs)
            # print(name[0].probs.top1conf.item())
            pred_name = name[0].probs.top1
            # conf is of tensor() type so do .item()
            pred_conf = name[0].probs.top1conf.item()
            # text = NAME_DICT[pred_name] if pred_conf > 0.5 else UNKNOWN
            text = NAME_DICT[pred_name]

            add_attendance(text)

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
        return buffer.tobytes()


def stream():
    vid = accessCamera()

    counter = 0

    while True:
        result, frame = vid.read()
        if result is False:
            break

        # process every 5 frames
        if counter % 10 == 0:
            p_frame = process_frame(frame)
            if p_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + p_frame + b"\r\n"
                )

        counter += 1


@video_stream_controller.route("/index")
def index():
    return render_template("index.html", data=STUDENTS)


@video_stream_controller.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
