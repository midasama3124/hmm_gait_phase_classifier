# Online Gait Phase Detection

The AGoRA exoskeleton utilizes a threshold-based algorithm (written in C++) and a  Hidden Markov Model (HMM) based partitioning method with two training approaches (written in Python). These modules run on ROS (Robot Operative System) for its modular integration with the system control of this wearable device in the package exo_control.

## Threshold-based algorithm

The main node may be found in src/nodes/threshold_detection_node.cpp which implements the following utilities: src/util/feature_extractor.cpp and src/util/gait_cycle_classifier.cpp. 

## HMM-based algorithm

This segmentation method makes use of the pomegranate library to create a Gaussian HMM trained by means of the Baum-Welch algorithm and validated by using the Viterbi algorithm, which finds the most-likely sequence based on an already-trained model.

In this sense, this classification method implements two training approaches in order to assess intra/inter variability within both healthy and pathological subjects. As part of an approved experimental protocol, the recruited participants performed three walking trials on a treadmill at a self-selected speed. On the one hand, a subject-specific technique trains the model with data drawn from the first two walking trials and test its performance with the remaining one (scripts/HMM_sclassifier_intra-sub_train.py). On the other hand, an intra-subject approach uses data from healthy subjects to train the model and test with the last walking trial corresponding to the assessed subject (scripts/HMM_sclassifier_intra-sub_train.py).

## Usage

The mentioned ROS nodes may be executed by using the following command lines:

```bash
rosrun exo_control threshold_detection_node
```

```bash
rosrun exo_control <node_name>.py
```

for C++ and Python, respectively.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
