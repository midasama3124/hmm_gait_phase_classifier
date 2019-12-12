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
from sensor_msgs.msg import Imu

"""Real-time HMM class"""
class RealTimeHMM():
    def __init__(self, patient, n_trials=3, leave_one_out=1, verbose=False):
        """Variable initialization"""
        self.patient = patient
        self.n_trials = n_trials
        self.n_samples = 0
        self.n_features = 2      # Raw data and 1st-derivative
        self.leave_one_out = leave_one_out
        self.verbose = verbose
        self.rate = 200      # Hz
        self.model_name = "Gait"
        self.states = ['s1', 's2', 's3', 's4']
        self.n_states = len(self.states)
        self.state2phase = {"s1": "hs", "s2": "ff", "s3": "ho", "s4": "sp"}
        """ROS init"""
        rospy.init_node('real_time_HMM', anonymous=True)
        rospack = rospkg.RosPack()
        self.packpath = rospack.get_path('exo_gait_phase_det')
        self.init_pubs()
        """Local data loading and feature extraction"""
        self.load_data()

    """Init ROS publishers"""
    def init_pubs(self):
        self.imu_pub = rospy.Publisher("/imu_data", Imu, queue_size = 1, latch = False)
        self.label_pub = rospy.Publisher("/label", Int8, queue_size = 1, latch = False)

    """Local data loading and feature extraction"""
    def load_data(self):
        """Data loading"""
        # datapath = self.packpath + "/log/mat_files/"
        datapath = self.packpath + "/log/mat_files/simulation/"
        # self.gyro_y = []
        # self.labels = []
        try:
            data = scio.loadmat(datapath + self.patient + "_proc_data" + str(self.leave_one_out) + ".mat")
            self.accel_x = data["accel_x"][0]
            self.accel_y = data["accel_y"][0]
            self.accel_z = data["accel_z"][0]
            self.gyro_x = data["gyro_x"][0]
            self.gyro_y = data["gyro_y"][0]
            self.gyro_z = data["gyro_z"][0]
            self.labels = data["labels"][0]
        except:
            self.rate = 100
            data = scio.loadmat(datapath + self.patient + "_rec_data.mat")
            for i in range(len(data["gyro_y"])):
                self.accel_x.append(data["accel_x"][i][0])
                self.accel_y.append(data["accel_y"][i][0])
                self.accel_z.append(data["accel_z"][i][0])
                self.gyro_x.append(data["gyro_x"][i][0])
                self.gyro_y.append(data["gyro_y"][i][0])
                self.gyro_z.append(data["gyro_z"][i][0])

        self.n_samples = len(self.gyro_y)
        rospy.logwarn("Data loaded")


def main():
    if(len(sys.argv)<2):
        print("Missing patient's name.")
        exit()
    else:
        patient = sys.argv[1]
        RtHMM = RealTimeHMM(patient)
        rate = rospy.Rate(RtHMM.rate) # Hz
        imu_msg = Imu()
        label_msg = Int8()
        smp = 0
        n_smp = RtHMM.n_samples

        rospy.logwarn("Publishing...")
        start_time = time.time()
        while not rospy.is_shutdown() and smp < n_smp:
            # imu_msg.time_stamp = int(round((time.time() - start_time)*1000.0))
            imu_msg.angular_velocity.x = RtHMM.gyro_x[smp]
            imu_msg.angular_velocity.y = RtHMM.gyro_y[smp]
            imu_msg.angular_velocity.z = RtHMM.gyro_z[smp]
            imu_msg.linear_acceleration.x = RtHMM.accel_x[smp]
            imu_msg.linear_acceleration.y = RtHMM.accel_y[smp]
            imu_msg.linear_acceleration.z = RtHMM.accel_z[smp]
            RtHMM.imu_pub.publish(imu_msg)
            if len(RtHMM.labels) > 0:
                label_msg.data = RtHMM.labels[smp]
                RtHMM.label_pub.publish(label_msg)
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
