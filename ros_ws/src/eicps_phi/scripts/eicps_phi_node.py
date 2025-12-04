#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import numpy as np
import sys
import os

# 动态添加 core 路径
current_dir = os.path.dirname(os.path.abspath(__file__))
core_path = os.path.join(current_dir, '../../../../core')
sys.path.append(core_path)

from phi_interface.turtlebot_core import EICPS_Interface
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32

class EICPS_Node:
    def __init__(self):
        rospy.init_node('eicps_phi_node', anonymous=True)
        self.eicps = EICPS_Interface()
        
        # ROS 参数
        self.eicps.d_min = rospy.get_param('~d_min', 0.20)
        self.eicps.v_max = rospy.get_param('~v_max', 0.26)
        
        self.min_scan_dist = 99.9
        
        # 通信接口
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/cmd_vel_ai', Twist, self.ai_cmd_cb)
        self.pub_safe = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.pub_h = rospy.Publisher('/eicps/debug/h_val', Float32, queue_size=1)

        rospy.loginfo("EICPS Phi Node Running...")

    def scan_cb(self, msg):
        ranges = np.array(msg.ranges)
        valid = ranges[np.isfinite(ranges)]
        if len(valid) > 0:
            self.min_scan_dist = np.min(valid)
        else:
            self.min_scan_dist = 99.9

    def ai_cmd_cb(self, msg):
        u_ai = np.array([msg.linear.x, msg.angular.z])
        
        # Φ1: HNN
        u_dyn = self.eicps.phi_hnn_projection(u_ai)
        
        # Φ2 + Φ3: CBF + PDT
        u_safe, status = self.eicps.solve_safety_filter(None, u_dyn, self.min_scan_dist)
        
        # Debug
        self.pub_h.publish(self.min_scan_dist - self.eicps.d_min)
        
        # Publish
        t = Twist()
        t.linear.x = u_safe[0]
        t.angular.z = u_safe[1]
        self.pub_safe.publish(t)

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        EICPS_Node().run()
    except rospy.ROSInterruptException:
        pass
