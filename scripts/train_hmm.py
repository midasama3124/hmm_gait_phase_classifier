#!/usr/bin/env python
import rospy
import rospkg
import pickle
import numpy as np
import sys, os
from sklearn.preprocessing import normalize
from pomegranate import*
from pomegranate.hmm import log as ln
from pomegranate import HiddenMarkovModel as HMM
from pomegranate import MultivariateGaussianDistribution as MGD
from scipy import io as scio
from scipy import linalg
from std_msgs.msg import Int8
from exo_msgs.msg import IMUData
from exo_msgs.srv import Trigger, TriggerResponse

"""HMM trainer class"""
class TrainHMM():
    def __init__(self):
        """ROS init"""
        rospy.init_node("hmm_trainer",anonymous=True, disable_signals=False)
        self.startTraining = False
        self.timeElapsed = False
        self.wasRecorded = False
        self.verbose = rospy.get_param("gait_phase_det/verbose")
        """Log init"""
        self.start_time = time.time()
        self.model_name = "Gait"
        self.states = ['s1', 's2', 's3', 's4']
        self.n_states = len(self.states)
        self.state2phase = {"s1": "hs", "s2": "ff", "s3": "ho", "s4": "sp"}
        self.mgds = {}
        self.dis_means = [[] for x in range(self.n_states)]
        self.dis_covars = [[] for x in range(self.n_states)]
        self.states_pos = {}
        for i in range(len(self.states)):
            self.states_pos[self.states[i]] = i
        self.start_prob = [1.0/self.n_states]*self.n_states
        self.model = HMM(name=self.model_name)
        self.gyro_y = []
        self.train_data = []
        self.train_time = rospy.get_param("gait_phase_det/training_time")
        self.patient = rospy.get_param("gait_phase_det/patient")
        rospack = rospkg.RosPack()
        self.packpath = rospack.get_path('exo_gait_phase_det')
        if not os.path.isfile(self.packpath + "/log/mat_files/" + self.patient + "_rec_data.mat"):
            rospy.logwarn("Initializing server...")
            self.init_servers_()
            self.init_subs_()
        else:
            self.wasRecorded = True
            self.load_data()

    def svr_handle_(self, req):
        self.startTraining = True
        self.start_time = time.time()
        return TriggerResponse(True, "Starting training...")

    """Init ROS servers"""
    def init_servers_(self):
        self.training_svr = rospy.Service('train_hmm', Trigger, self.svr_handle_)

    """Init ROS subcribers"""
    def init_subs_(self):
        rospy.Subscriber('/imu_data', IMUData, self.imu_callback)
        rospy.loginfo("Subscribed to IMU data")

    def imu_callback(self, data):
        if self.startTraining and not self.timeElapsed:
            elapsed_time = time.time() - self.start_time
            if ( elapsed_time < self.train_time ):
                self.gyro_y.append(data.gyro_y)
                rospy.loginfo("Elapsed time: {} seconds".format(elapsed_time))
            else:
                self.timeElapsed = True
<<<<<<< Updated upstream

    """Load previously recorded data"""
    def load_data(self):
        datapath = self.packpath + "/log/mat_files/"
        print datapath
        data = scio.loadmat(datapath + self.patient + "_rec_data.mat")
        for i in range(len(data["gyro_y"])):
            self.gyro_y.append(data["gyro_y"][i][0])
=======
>>>>>>> Stashed changes

    """Init HMM if no previous training"""
    def init_hmm(self):
        """Feature extraction"""
        '''First derivative'''
        fder_gyro_y = []
        fder_gyro_y.append(self.gyro_y[0])
        for i in range(1,len(self.gyro_y)-1):
            fder_gyro_y.append((self.gyro_y[i+1]-self.gyro_y[i-1])/2)
<<<<<<< Updated upstream
            fder_gyro_y.append(self.gyro_y[-1])
=======
        fder_gyro_y.append(self.gyro_y[-1])
>>>>>>> Stashed changes

        """Create training data"""
        for i in range(len(self.gyro_y)):
            f_ = []
            f_.append(self.gyro_y[i])
            f_.append(fder_gyro_y[i])
            self.train_data.append(f_)
        self.n_features = len(self.train_data[0])

        """Transition matrix (A)"""
        '''Left-right model'''
        self.trans_mat = np.array([(0.9, 0.1, 0, 0), (0, 0.9, 0.1, 0), (0, 0, 0.9, 0.1), (0.1, 0, 0, 0.9)])
        '''Right-left-right model'''
        # self.trans_mat = np.array([0.8, 0.1, 0, 0.1], [0.1, 0.8, 0.1, 0], [0, 0.1, 0.8, 0.1], [0.1, 0, 0.1, 0.8])
        if self.verbose: rospy.logwarn("TRANSITION MATRIX\n" + str(self.trans_mat))

        """Multivariate Gaussian Distributions for each hidden state"""
        '''MGDs loading'''
        try:
            for st in self.states:
                with open(self.packpath+'/log/HMM_models/generic_'+self.state2phase[st]+'.txt') as infile:
                    yaml_dis = yaml.safe_load(infile)
                    dis = MGD.from_yaml(yaml_dis)
                    self.mgds[st] = dis
                    '''Loading means and covariance matrix'''
                    self.dis_means[self.states_pos[st]] = self.mgds[st].parameters[0]
                    self.dis_covars[self.states_pos[st]] = self.mgds[st].parameters[1]
            rospy.loginfo("Generic MGCs were loaded.")
        except yaml.YAMLError as exc:
            rospy.logwarn("Not able to load distributions: " + exc)

        """Classifier initialization"""
        hmm_states = []
        for state_name in self.states:
            st = State(self.mgds[state_name], name=state_name)
            hmm_states.append(st)
        self.model.add_states(hmm_states)
        '''Initial transitions'''
        for i in range(self.n_states):
            self.model.add_transition(self.model.start, hmm_states[i], self.start_prob[i])
        '''Left-right model'''
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.model.add_transition(hmm_states[i], hmm_states[j], self.trans_mat[i][j])
        '''Finish model setup'''
        self.model.bake()

    """Train initialized model"""
    def train_hmm(self):
        rospy.logwarn("Training initialized model...")
        self.model.fit(self.train_data, algorithm='baum-welch', verbose=self.verbose)
        self.model.freeze_distributions()     # Freeze all model distributions, preventing update from ocurring
        '''Save Multivariate Gaussian Distributions into yaml file'''
        for st in self.model.states:
            if st.name != self.model_name+"-start" and st.name != self.model_name+"-end":
                dis = st.distribution
                dis_yaml = dis.to_yaml()
                try:
                    with open(self.packpath+'/log/HMM_models/'+self.patient+'_'+self.state2phase[st.name]+'.txt', 'w') as outfile:
                        yaml.dump(dis_yaml, outfile, default_flow_style=False)
                    rospy.loginfo(self.patient+"'s "+self.state2phase[st.name]+" distribution was saved.")
                except IOError:
                    rospy.logerror('It was not possible to write HMM model.')
        '''Save model (json script) into txt file'''
        model_json = self.model.to_json()
        try:
            with open(self.packpath+'/log/HMM_models/'+self.patient+'.txt', 'w') as outfile:
                json.dump(model_json, outfile)
            rospy.logwarn(self.patient+"'s HMM model was saved.")
        except IOError:
            rospy.logerr('It was not possible to write HMM model.')

def main():
    TrHMM = TrainHMM()
<<<<<<< Updated upstream
    if not TrHMM.wasRecorded:
        rospy.logwarn("Waiting client request to collect data for {} seconds...".format(TrHMM.train_time))
        while not TrHMM.startTraining: pass
        rospy.loginfo("Server was called")
        rospy.logwarn("Collecting data for {} seconds...".format(TrHMM.train_time))

    while not rospy.is_shutdown():
        if TrHMM.timeElapsed or TrHMM.wasRecorded:
=======
    rospy.logwarn("Waiting client request to collect data for {} seconds...".format(TrHMM.train_time))
    while not TrHMM.startTraining: pass
    rospy.loginfo("Server was called")

    rospy.logwarn("Collecting data for {} seconds...".format(TrHMM.train_time))

    while not rospy.is_shutdown():
        if TrHMM.timeElapsed:
>>>>>>> Stashed changes
            TrHMM.init_hmm()
            TrHMM.train_hmm()
            rospy.signal_shutdown("{}'s model was trained!".format(TrHMM.patient))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
