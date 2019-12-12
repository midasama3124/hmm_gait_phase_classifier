#include <stdint.h>
#include "hmm_gait_phase_classifier/feature_extractor.hpp"
#include "hmm_gait_phase_classifier/gait_cycle_classifier.hpp"
#include "ros/ros.h"
#include <std_msgs/Int8.h>
#include <sensor_msgs/Imu.h>
#include <cstdlib>
#include <iostream>
#include <string>

using namespace std;

int32_t time_stamp;
double accel_y, gyro_y;
string node_name = "threshold_detection_node";

//instantiate the classifier
Gait_cycle_classifier *classifer = new Gait_cycle_classifier();
//used to receive the information of newly-recognized states
State_recognized_info latest_state_info;

void callback(const sensor_msgs::Imu::ConstPtr &msg){
  accel_y = msg->linear_acceleration.y;
  gyro_y = msg->angular_velocity.y;
  time_stamp = msg->header.stamp.nsec;

  //print received data out
  // cout << "time_stamp: " << time_stamp << ", accel_y: " << accel_y << ", gyro_y: " << gyro_y << endl;

  // if a new state(phase) was recognized, print out the state and the time it was recognized at
  /* Constants for original IMU class */
  // if (classifer->intake_data(4*accel_y, -6*gyro_y, time_stamp, latest_state_info)){
  /* Constants for IMU class with gyro/16.0 and accel/100.0 */
  // if (classifer->intake_data(400*accel_y, -100*gyro_y, time_stamp, latest_state_info)){      // Failing, working previously
  if (classifer->intake_data(400*accel_y, 500*gyro_y, time_stamp, latest_state_info)){
  // The gyroscope data needs to be inverted as the IMU from the original
  // package is oriented differently
  // if (classifer->intake_data(accel_y, -gyro_y, time_stamp, latest_state_info)){
  	ROS_INFO_STREAM("Gait phase: " << get_state_string(latest_state_info.recognized_state) << " at time: " << latest_state_info.time_recognized);
  }
}

int main(int argc, char** argv){
  cout << "Initializing the classifier..." << endl;

  // ROS init
  ros::init(argc, argv, node_name);
  // NodeHandle is the main access point to communications with the ROS system.
	ros::NodeHandle n;

  // ROS publisher. Topic: gait_phase_detection
  ros::Publisher chatter_pub = n.advertise<std_msgs::Int8>("gait_phase", 1000);
  // ROS subscriber. Topic: imu_data
  ros::Subscriber sub = n.subscribe("imu_data", 1000, callback);

  // Running at 100Hz
	ros::Rate loop_rate(100);
	ros::spinOnce();

  while (ros::ok()) {
    // Publish GaitPhase message
    std_msgs::Int8 gait_phase;
    if (get_state_string(latest_state_info.recognized_state) == "Heel strike"){
      gait_phase.data = 0;
    }
    else if (get_state_string(latest_state_info.recognized_state) == "Flat foot"){
      gait_phase.data = 1;
    }
    else if (get_state_string(latest_state_info.recognized_state) == "Mid-stance"){
      gait_phase.data = 1;
      // gait_phase.data = 2;
    }
    else if (get_state_string(latest_state_info.recognized_state) == "Heel-off"){
      gait_phase.data = 2;
      // gait_phase.data = 3;
    }
    else if (get_state_string(latest_state_info.recognized_state) == "Toe-off"){
      gait_phase.data = 3;
      // gait_phase.data = 4;
    }
    else if (get_state_string(latest_state_info.recognized_state) == "Mid-swing"){
      gait_phase.data = 3;
      // gait_phase.data = 5;
    }
    // Invalid State!!
    else gait_phase.data = 4;
    // else gait_phase.data = 6;
    // cout << "Publishing gait_phases topic..." << endl;
    chatter_pub.publish(gait_phase);

    ros::spinOnce();
    loop_rate.sleep();
  }
	return 0;
}
