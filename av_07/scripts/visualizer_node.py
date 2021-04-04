#!/usr/bin/env python
from __future__ import print_function


import rospy
import cv2
from sensor_msgs.msg import Image
from av_msgs.msg import Mode
from av_msgs.msg import States
from prius_msgs.msg import Control
from cv_bridge import CvBridge, CvBridgeError

class image_converter:

  def __init__(self):
    self.varShiftGears = 0
    self.varSelfdriving = 0
    self.varCollect = 0
    self.varVelocity = 0
    self.varSteer = 0
    self.sub_prius = rospy.Subscriber("prius", Control, self.callback_prius)
    self.sub_prius_mode = rospy.Subscriber('prius/mode', Mode, self.callback_prius_mode)
    self.sub_prius_states = rospy.Subscriber('prius/states', States, self.callback_prius_states)
    self.image_pub = rospy.Publisher("prius/visualization",Image)

    self.bridge = CvBridge()
    self.image_sub = rospy.Subscriber("prius/front_camera/image_raw",Image,self.callback)

  def callback(self,data):
    try:
      cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
    except CvBridgeError as e:
      print(e)

    (rows,cols,channels) = cv_image.shape
    cv2.putText(cv_image, "LinV: {:.2f} km/h, Steer: {:.2f}".format(self.varVelocity, self.varSteer), (30, rows-30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(cv_image, "Gear: " + str(self.varShiftGears), (30, rows-80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(cv_image, "Collect: " + str(self.varCollect) + ", SelfDriving: " + str(self.varSelfdriving), (30, rows - 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imshow("Image window", cv_image)
    cv2.waitKey(3)

    try:
      self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
    except CvBridgeError as e:
      print(e)

  def callback_prius(self, message):
    self.varShiftGears = message.shift_gears
    self.varSteer = message.steer
  def callback_prius_mode(self, message):
    self.varSelfdriving = message.selfdriving
    self.varCollect = message.collect
  def callback_prius_states(self, message):
    self.varVelocity = message.velocity
if __name__ == '__main__':
    rospy.init_node('visualizer_node')
    ic = image_converter()
    rospy.spin()