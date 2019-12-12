#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "hmm_gait_phase_classifier/feature_extractor.hpp"
#include "hmm_gait_phase_classifier/gait_cycle_classifier.hpp"
#include "ros/ros.h"
#include <ros/package.h>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <iomanip>
#include <fstream>
#include <vector>
#include <sys/types.h>
#include <sys/stat.h>

using namespace std;

int32_t time_stamp;
double accel_y, gyro_y;
string node_name = "offline_threshold_detection_node";

int main(int argc, char** argv){
  cout << "Initializing the classifier..." << endl;

  // ROS init
  ros::init(argc, argv, node_name);
  // NodeHandle is the main access point to communications with the ROS system.
	ros::NodeHandle n;

  // Data loading
  string patient, PackPath = ros::package::getPath("hmm_gait_phase_classifier");;
  n.getParam("gait_phase_det/patient", patient);
	string data_path = PackPath + "/log/IMU_data/" + patient + ".csv";
  ROS_INFO_STREAM("Path for " + patient + "'s data: " << data_path);
	ifstream data_file(data_path);

	if(!data_file.is_open()) ROS_ERROR("Error: Open data file");

  string line;
	double gyro_y[50000];
	double accel_y[50000];
	double time_stamp[50000];
	int n_data = 0;

	while(data_file.good()){
		getline(data_file,line,'\n');
    vector<float> data;
    stringstream ss(line);
    float i;

    while(ss >> i){
      data.push_back(i);
      if (ss.peek() == ',') ss.ignore();
    }

    // Print line content
    // for (i=0; i< data.size(); i++)
    //   cout << data.at(i) << endl;
    // cout << endl;

    // gyro_y[n_data] = atof(line.c_str());   // Convert string to double
    if (data.size() > 0){
      time_stamp[n_data] = data.at(0);   // Convert string to double
      gyro_y[n_data] = data.at(2);   // Convert string to double
      accel_y[n_data] = data.at(5);   // Convert string to double
    }
		n_data++;
	}

	// Visualize data
	// for (int i=0; i<n_data; i++)
	// 	cout << gyro_y[i] << endl;

	data_file.close();

  /* Threshold-based classifier */
  // Instantiate the classifier
  Gait_cycle_classifier *classifier = new Gait_cycle_classifier();
  //used to receive the information of newly-recognized states
  State_recognized_info latest_state_info;

	// Export gait phase in a .csv file
	ofstream labels;
	labels.open(string(PackPath + "/log/IMU_data/" + patient + "_labels.csv").c_str());

	/* If a new state(phase) was recognized, print out the state and the time it was recognized at
	* Constants for original IMU class */
	for (int i=0; i<n_data; i++){
		// if (classifier->intake_data(4*accel_y[i], -6*gyro_y[i], time_stamp[i], latest_state_info)){
    /* Constants for IMU class with gyro/16.0 and accel/100.0
    * The gyroscope data needs to be inverted as the IMU from the original
    * package is oriented differently
    */
    // if (classifier->intake_data(400*accel_y[i], -100*gyro_y[i], time_stamp[i], latest_state_info)){    // Failing, working previously
		if (classifier->intake_data(400*accel_y[i], 500*gyro_y[i], time_stamp[i], latest_state_info)){
			cout << "Gait phase: " << get_state_string(latest_state_info.recognized_state) << " at time: " << latest_state_info.time_recognized/1000.0 << endl;
		}
		if (get_state_string(latest_state_info.recognized_state) == "Heel strike"){
			labels << "0\n";
		}
		else if (get_state_string(latest_state_info.recognized_state) == "Flat foot" || get_state_string(latest_state_info.recognized_state) == "Mid-stance"){
			labels << "1\n";
		}
		else if (get_state_string(latest_state_info.recognized_state) == "Heel-off"){
			labels << "2\n";
		}
		else if (get_state_string(latest_state_info.recognized_state) == "Toe-off" || get_state_string(latest_state_info.recognized_state) == "Mid-swing"){
			labels << "3\n";
		}
	}
	labels.close();
	cout << "csv file closed " << "(" << patient << ")." << endl;

	return 0;
}
