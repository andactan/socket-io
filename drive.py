import argparse
import base64
from datetime import datetime
import os
import shutil
import matplotlib.pyplot as plt

import cv2
import numpy as np
import socketio
import eventlet
import eventlet.wsgi
from flask import Flask
from io import BytesIO
from PIL import Image
from keras.models import load_model

from detect import yolo_net_out_to_car_boxes, draw_box

sio = socketio.Server()
app = Flask(__name__)
prev_image_array = None

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
set_speed = 9
controller.set_desired(set_speed)

@sio.on('connect')
def open(sid, environ):
    print("Opened -- ", sid)

@sio.on('telemetry')
def telemetry(sid, data):
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]
    # The current image from the center camera of the car
    imgString = data["image"]
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    image_array = np.asarray(image)

    model = load_model('my_model.h5')
    image_crop = image_array[300:650, 500:, :]
    resized = cv2.resize(image_crop, (448, 448))

    batch = np.transpose(resized, (2, 0, 1))
    batch = 2 * (batch / 255.) - 1
    batch = np.expand_dims(batch, axis=0)
    out = model.predict(batch)

    boxes = yolo_net_out_to_car_boxes(out[0], threshold=0.17)
    image_boxed = draw_box(boxes, image_array, [[500, 1280], [300, 650]])[1][0][1]
    if (image_boxed - 640 > 0):
        print('steer right')

    else:
        print('steer left')

    timestamp = datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-3]
    image_folder = "/images4"
    image_filename = os.path.join(image_folder, timestamp)
    Image.fromarray(image_boxed).save('{}.jpg'.format(image_filename))
    print(data)
    data.pop('image', None)
    send_control(data)

def send_control(data):
    pass

if __name__ == '__main__':
    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)

