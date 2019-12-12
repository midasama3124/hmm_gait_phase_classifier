#!/usr/bin/env python
import rospy
import rospkg
import pickle
import entry_data as ed
import numpy as np
import sys
from sklearn.preprocessing import normalize
from entry_data import DataEntry, fullEntry
from pomegranate import*
from pomegranate.hmm import log as ln
from pomegranate import HiddenMarkovModel as HMM
from pomegranate import MultivariateGaussianDistribution as MGD
from scipy import io as scio
from scipy import linalg
from std_msgs.msg import Int8
from exo_msgs.msg import IMUData

"""Real-time HMM class"""
class DefaultGaitPhase():
    def __init__(self, verbose=False):
        """Variable initialization"""
        self.verbose = verbose
        self.rate = 200.0      # Hz
        self.gait_phase_perc = [0.16, 0.54, 0.6, 1.0]
        self.cycle_duration = 2.3    # seconds
        """ROS init"""
        rospy.init_node('default_gait_phase', anonymous=True)
        rospack = rospkg.RosPack()
        self.packpath = rospack.get_path('exo_gait_phase_det')
        self.init_pubs()

    """Init ROS publishers"""
    def init_pubs(self):
        self.phase_pub = rospy.Publisher("/gait_phase", Int8, queue_size = 1, latch = False)

def main():
    # if(len(sys.argv)<2):
    #     print("Missing patient's name.")
    #     exit()
    # else:
        # patient = sys.argv[1]
    DefGaitPhase = DefaultGaitPhase()
    rate = rospy.Rate(DefGaitPhase.rate) # Hz
    gait_phase_msg = Int8()
    gait_phase_msg.data = 0
    smp = 0

    rospy.logwarn("Publishing...")
    while not rospy.is_shutdown():
        # rospy.logwarn("Elapsed time: {} seconds".format(smp/DefGaitPhase.rate))
        if smp/DefGaitPhase.rate > DefGaitPhase.cycle_duration*DefGaitPhase.gait_phase_perc[gait_phase_msg.data]:
            gait_phase_msg.data += 1
        if gait_phase_msg.data == 4:
            gait_phase_msg.data = 0
            smp = 0
        DefGaitPhase.phase_pub.publish(gait_phase_msg)
        rate.sleep()
        smp += 1

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
