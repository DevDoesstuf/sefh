import numpy as np
import cv2
import tensorflow as tf
from djitellopy import Tello
from PIL import Image, ImageEnhance
import time

# load the pre-trained mobilenet model
model = tf.keras.models.load_model('mobilenet_model.h5')

# connect to tello
tello = Tello()
tello.connect()
tello.streamon()

time.sleep(2)
tello.takeoff()

def capture_image():
    # capture image from tello's camera
    frame = tello.get_frame_read().frame
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    
    # apply contrast enhancement
    contrast_enhancer = ImageEnhance.Contrast(image)
    image_contrasted = contrast_enhancer.enhance(1.5)
    
    # apply color (saturation) enhancement
    color_enhancer = ImageEnhance.Color(image_contrasted)
    image_final = color_enhancer.enhance(1.5)
    
    return np.array(image_final)

# process the image for mobilenet model
def process_image(image):
    image = cv2.resize(image, (224, 224))
    image = np.expand_dims(image, axis=0) / 255.0
    prediction = model.predict(image)
    return np.argmax(prediction) == 1  # assuming class 1 corresponds to nitrogen deficiency

# moves the drone forward by a specified distance
def move_forward(distance):
    print(f"moving forward {distance} meters")
    tello.move_forward(distance * 100)
    time.sleep(2)

# adjusts the drone's tilt
def adjust_tilt(angle):
    print(f"tilting {angle} degrees")
    if angle == 15:
        tello.move_down(20)
    elif angle == 0:
        tello.move_up(20)
    time.sleep(1)

# checks battery status and returns true if low
def monitor_battery():
    return tello.get_battery() < 20

# lands the drone safely
def land_drone():
    print("battery low. landing...")
    tello.land()
    time.sleep(5)

# covers the area in a rectangular pattern
def cover_area(length, width):
    for row in range(int(length)):
        for col in range(int(width)):
            image = capture_image()
            if process_image(image):
                adjust_tilt(15)
                move_forward(1)
                adjust_tilt(0)
            else:
                move_forward(1)
            if monitor_battery():
                land_drone()
                return
        if row < length - 1:
            tello.rotate_clockwise(90)  # turn right
            move_forward(2)  # move forward 2 meters
            tello.rotate_clockwise(90)  # turn right again
    
# start covering the designated area
cover_area(length=10, width=5)

# land the drone after completion
land_drone()
