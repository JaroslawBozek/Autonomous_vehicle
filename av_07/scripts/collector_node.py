# !/usr/bin/env python

# Copyright 2017 Open Source Robotics Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import rospy
from prius_msgs.msg import Control
from av_msgs.msg import Mode
from av_msgs.msg import States
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import csv
import cv2
import time

class Collector:

    def __init__(self):
        self.i = 0
        self.varVelocity = 0
        self.varSteer = 0
        self.varCollect = 0
        self.image_sub = rospy.Subscriber("prius/front_camera/image_raw",Image,self.callback)
        self.sub_prius = rospy.Subscriber('prius', Control, self.callback_prius)
        self.sub_prius_mode = rospy.Subscriber('prius/mode', Mode, self.callback_prius_mode)
        self.sub_prius_states = rospy.Subscriber('prius/states', States, self.callback_prius_states)

        self.last_published_time = rospy.get_rostime()
        self.last_published = None
        self.timer = rospy.Timer(rospy.Duration(1. / 20.), self.timer_callback)

        self.bridge = CvBridge()
    def timer_callback(self, event):
        if self.last_published and self.last_published_time < rospy.get_rostime() + rospy.Duration(1.0 / 20.):
            self.callback(self.last_published)

    def callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

        time.sleep(0.04)

        if self.varCollect == 1:
            cv2.imwrite('../data/images/images/' + str(self.i) + '.jpg', cv_image)
            with open("../data/data.csv", "a") as f:
                cr = csv.writer(f, delimiter=";", lineterminator="\n")
                cr.writerow([str(self.i), str(self.varVelocity), str(self.varSteer)])
            self.i = self.i + 1


    def callback_prius(self, message):
        self.varSteer = message.steer
    def callback_prius_mode(self, message):
        self.varCollect = message.collect
    def callback_prius_states(self, message):
        self.varVelocity = message.velocity


if __name__ == '__main__':
    rospy.init_node('collector_node')
    c = Collector()
    rospy.spin()
