# !/usr/bin/env python
import rospy

from av_msgs.msg import States
from nav_msgs.msg import Odometry




class Translator:

    def __init__(self):
        self.sub = rospy.Subscriber('base_pose_ground_truth', Odometry, self.callback)
        self.pub = rospy.Publisher('prius/states', States, queue_size=1)
        self.last_published_time = rospy.get_rostime()
        self.last_published = None
        self.timer = rospy.Timer(rospy.Duration(1. / 20.), self.timer_callback)

    def timer_callback(self, event):
        if self.last_published and self.last_published_time < rospy.get_rostime() + rospy.Duration(1.0 / 20.):
            self.callback(self.last_published)

    def callback(self, message):
        command = States()
        command.header = message.header

        var = (message.twist.twist.linear.x ** 2 + message.twist.twist.linear.y ** 2 + message.twist.twist.linear.z ** 2) ** 0.5
        command.velocity = var * 3.6
        # command.steer = message.twist.twist.angular.z

        self.pub.publish(command)

if __name__ == '__main__':
    rospy.init_node('states_node')
    t = Translator()
    rospy.spin()