import cv2
import os
import numpy as np
import tensorflow as tf
from recognition_models import FACE_DETECTION_MODEL

def crop_square(img, size, interpolation=cv2.INTER_AREA):
    h, w = img.shape[:2]
    min_size = np.amin([h,w])

    # Centralize and crop
    crop_img = img[int(h/2-min_size/2):int(h/2+min_size/2), int(w/2-min_size/2):int(w/2+min_size/2)]
    resized = cv2.resize(crop_img, (size, size), interpolation=interpolation)

    return resized


def performFaceDetection(vid) :
  facetracker = FACE_DETECTION_MODEL

  size = 450
  img_size = [size, size]
  
  while True :
    result, frame = vid.read()
    if result is False:
      break
    
    # crop and resize frame to size 450
    frame = crop_square(frame, size)
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resized = tf.image.resize(rgb, (120,120))
    
    yhat = facetracker.predict(np.expand_dims(resized/255, 0))
    print(yhat)
    sample_coords = yhat[1][0]
  
    if yhat[0] > 0.5 :
      # Control the main rectangle
      cv2.rectangle(
        frame,
        tuple(np.multiply(sample_coords[:2], img_size).astype(int)),
        tuple(np.multiply(sample_coords[2:], img_size).astype(int)),
        (255,0,0),
        2 
      )
    
    cv2.imshow("My Face Detection Project", frame) 
    if cv2.waitKey(1) & 0xFF == ord("q"):
      break
    
  vid.release()
  cv2.destroyAllWindows()
  
def accessCamera(IP_Stream) :
  return cv2.VideoCapture(IP_Stream)

video_stream = accessCamera(0)
performFaceDetection(video_stream)
