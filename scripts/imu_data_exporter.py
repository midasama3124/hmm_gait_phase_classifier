#!/usr/bin/python
import rospy
import rospkg
from exo_msgs.msg import IMUData
import time
import datetime
from os.path import expanduser

class IMU_Data_Exporter(object):
    def __init__(self):
        rospy.init_node("export_imu_data",anonymous=True)
        rospy.Subscriber("/imu_data", IMUData, self.callback)
        self.start_time = time.time()
        self.patient = rospy.get_param("gait_phase_det/patient")
        # Reading and logging imu_data
        now = datetime.datetime.now()
        rospack = rospkg.RosPack()
        packpath = rospack.get_path('exo_gait_phase_det')
        self.f = open(packpath + '/log/IMU_data/' + self.patient + '_'+str(now.month)+'-'+str(now.day)+'-'+str(now.hour)+'-'+str(now.minute)+'-'+str(now.second)+'-'+'.csv','wb')
        self.f.write('time_stamp,gyro_x,gyro_y,gyro_z,accel_x,accel_y,accel_z,quat_x,quat_y,quat_z,quat_w\n')
        # self.f.write('0,0,0,0,0,0,0\n')

    def callback(self, msg):
        # Time stamp
        time_stamp = int(round((time.time() - self.start_time)*1000.0))
        rospy.loginfo("Time: {} seconds".format( time_stamp/1000.0 ))

        # Reading and logging imu_data
        self.f.write('{},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}\n'.format(time_stamp,msg.gyro_x,msg.gyro_y,msg.gyro_z,msg.accel_x,msg.accel_y,msg.accel_z,msg.quat_x,msg.quat_y,msg.quat_z,msg.quat_w))

if __name__ == "__main__":
    time.sleep(5)
    record = raw_input("Do you want to start recording IMU data? [Y/n] ")
    if record.lower() in ['y', 'yes']:
        exp = IMU_Data_Exporter()
        rospy.spin()
    exp.f.close()
