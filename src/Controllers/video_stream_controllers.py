from flask import Blueprint, render_template, Response

import pickle
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras.models import load_model

 
video_stream_Controller = Blueprint("video_stream_Controller", __name__, template_folder='../Views')

def accessCamera() :
  return cv2.VideoCapture(0)

def stream() :
  vid = accessCamera()
  # facetracker = load_model('Face_Detection.h5')
  # face_recognition = load_model('Face_Recognition.keras')
  
  # with open("ResultsMap.pkl", 'rb') as file:
  #  result_map = pickle.load(file)
   
  while True:
    result, frame = vid.read() 
    if result is False:
      break

    # frame = frame[50 : 500, 50 : 500,:]
  
    # rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # resized = tf.image.resize(rgb, (120,120))
    
    # yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    # sample_coords = yhat[1][0]
    
    # name = face_recognition.predict(np.expand_dims(resized/255, 0), verbose=0)
    
    # if yhat[0] > 0.5 :
    #   # Control the main rectangle
    #   cv2.rectangle(
    #     frame,
    #     tuple(np.multiply(sample_coords[:2], [450,450]).astype(int)),
    #     tuple(np.multiply(sample_coords[2:], [450,450]).astype(int)),
    #     (255,0,0),
    #     2 
    #   )
      
    #   # Control the text rendered
    #   cv2.putText(
    #     frame, 
    #     result_map[np.argmax(name)],
    #     tuple(np.add(np.multiply(sample_coords[:2], [450,450]).astype(int),[0, -5])),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     1,
    #     (255,255,255),
    #     2,
    #     cv2.LINE_AA
    #   )
      
    ret, buffer = cv2.imencode('.jpg', frame)
    frame = buffer.tobytes()
    
    yield(
      b'--frame\r\n'
      b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n'
    )


@video_stream_Controller.route('/index')
def index() :
  return render_template('index.html')

@video_stream_Controller.route('/video')
def video() :
  return Response(stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

  
  