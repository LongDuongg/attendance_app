from flask import Blueprint, render_template, Response

import pickle
import cv2
import numpy as np
import pandas as pd
from datetime import date

import tensorflow as tf
from recognition_models.recognition_models import (
    FACE_DETECTION_MODEL,
    FACE_CLS_MODEL,
)
from recognition_models.utils import crop_square
from Models.mock_data import STUDENTS
from ultralytics.utils.plotting import Annotator

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


def add_attendance(name):
    if name == UNKNOWN:
        return

    found_student = [s for s in STUDENTS if s["id_name"] == name][0]
    time = date.today().strftime("%d/%m/%Y %H:%M:%S")

    df = pd.read_csv("Attendance.csv", dtype=str, encoding="utf-8")
    found_ids = list(df["id"])

    if found_student["id"] not in found_ids:
        with open("Attendance.csv", "a", encoding="utf-8") as f:
            line = f"\n{found_student['name']},{found_student['id']},{found_student['class']},\"{found_student['address']}\",{found_student['image']},{time}"
            print(line)
            f.write(line)


def process_frame_yolov8(frame):
    size = 480
    # crop and resize frame to size 640
    frame = crop_square(frame, size)

    results = face_tracker.predict(frame)

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
                print(pred_name, pred_conf)
                text = NAME_DICT[int(pred_name)] if pred_conf > 0.6 else "Unknown"
                
                add_attendance(text)
                
                annotator.box_label(b, text)

    frame = annotator.result()
    ret, buffer = cv2.imencode(".jpg", frame)
    return buffer.tobytes()


def process_frame_long(frame):
    size = 450
    crop_size = 120
    img_size = [size, size]

    # crop and resize frame to size 450
    frame = crop_square(frame, size)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (crop_size, crop_size))

    # np.expand_dims(img_tensor, axis=0): (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    # resized / 255: imshow expects values in the range [0, 1]
    yhat = face_tracker.predict(np.expand_dims(resized / 255, 0))
    sample_coords = yhat[1][0]
    # print(yhat)

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
                (crop_size, crop_size),
            )

            result = face_recognition.predict(
                np.expand_dims(cropped / 255, 0), verbose=0
            )
            # print("pred", result)
            pred_conf = np.max(result)
            pred_name = np.argmax(result)
            print(pred_name, pred_conf)

            text = NAME_DICT[pred_name] if pred_conf > 0.6 else UNKNOWN
            # text = NAME_DICT[pred_name]

            add_attendance(text)

            # Control the text rendered
            cv2.putText(
                frame,
                text,
                tuple(
                    np.add(
                        np.multiply(sample_coords[:2], [size, size]).astype(int),
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
        if counter % 5 == 0:
            # p_frame = process_frame_long(frame)
            p_frame = process_frame_yolov8(frame)
            if p_frame:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n" + p_frame + b"\r\n"
                )

        counter += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vid.release()


@video_stream_controller.route("/index")
def index():
    return render_template("index.html", data=STUDENTS)


@video_stream_controller.route("/video")
def video():
    return Response(stream(), mimetype="multipart/x-mixed-replace; boundary=frame")
