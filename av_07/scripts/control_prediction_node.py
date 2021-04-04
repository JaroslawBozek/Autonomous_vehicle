# !/usr/bin/env python
import rospy

from av_msgs.msg import States
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from prius_msgs.msg import Control
from av_msgs.msg import Mode
from simple_pid import PID
from cv_bridge import CvBridge, CvBridgeError
import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import backend as K

def allow_memory_growth():
  gpus = tf.config.experimental.list_physical_devices('GPU')
  if gpus:
    try:
      # Currently, memory growth needs to be the same across GPUs
      for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
      logical_gpus = tf.config.experimental.list_logical_devices('GPU')
      print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
      # Memory growth must be set before GPUs have been initialized
      print(e)

allow_memory_growth()


def img_preprocess(img):
  img = img[380:600, :, :]
  img = cv2.resize(img, (200, 200))

  return img



class Translator:

    def __init__(self):
        self.session = tf.compat.v1.keras.backend.get_session()
        self.model = tf.keras.models.load_model('../cnn_models/model.model')
        self.image_sub = rospy.Subscriber("prius/front_camera/image_raw",Image,self.callback)
        self.vel_sub = rospy.Subscriber("prius/states", States, self.callback2)
        self.mode_sub = rospy.Subscriber('prius/mode', Mode, self.callback3)
        self.pub = rospy.Publisher('prius', Control, queue_size=1)
        self.last_published_time = rospy.get_rostime()
        self.last_published = None
        self.timer = rospy.Timer(rospy.Duration(1. / 20.), self.timer_callback)

        self.autodrive = 0
        self.output_vel = 0
        self.curr_vel = 0
        self.pred_vel = 0

        self.bridge = CvBridge()
    def predict(self, x):
        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)
            out = self.model.predict(x)
        return out
    def timer_callback(self, event):
        if self.last_published and self.last_published_time < rospy.get_rostime() + rospy.Duration(1.0 / 20.):
            self.callback(self.last_published)

    def callback(self, message):

        if self.autodrive == True:
            try:
                image = self.bridge.imgmsg_to_cv2(message, "bgr8")
            except CvBridgeError as e:
                print(e)

            preprocessed_image = img_preprocess(image)
            preprocessed_image = preprocessed_image[np.newaxis, :]

            prediction = self.predict(preprocessed_image)

            command = Control()
            command.steer = prediction[0][1]
            self.pred_vel = prediction[0][0] - abs(prediction[0][1])
            command.throttle = self.output_vel
            self.pub.publish(command)

    def callback2(self, message):

        if self.autodrive == True:
            self.curr_vel = message.velocity
            pid_vel = PID(.1, .1, 0)
            pid_vel.sample_time = .2
            pid_vel.setpoint = self.pred_vel*135
            self.output_vel = pid_vel(self.curr_vel)


    def callback3(self, message):
        self.autodrive = message.selfdriving
if __name__ == '__main__':
    rospy.init_node('control_prediction')
    t = Translator()
    rospy.spin()