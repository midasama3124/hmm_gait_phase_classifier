#!/usr/bin/env python
import rospy
import rospkg
# import pickle
# import entry_data as ed
import numpy as np
import csv
import sys
import os
from scipy import io as scio
from sklearn.preprocessing import normalize
from entry_data import DataEntry, fullEntry
from pomegranate import*
from pomegranate import HiddenMarkovModel as HMM
from pomegranate import MultivariateGaussianDistribution as MGD
# ROS messages
from std_msgs.msg import Int32
from exo_msgs.msg import IMUData

"""Find path to ROS package"""
rospack = rospkg.RosPack()
packpath = get_path('exo_gait_phase_det')

"""Real-time HMM class"""
class RealTimeHMM:
    def __init__(self, patient):
        """Variable initialization"""
        self.patient = patient
        self.curr_state = -1                   # Current detected state
        self.rec_data = 0                      # Amount of received data
        self.proc_data = 0                     # Counter of processed data
        # self.win_len = 6                       # Window length
        # self.raw_win = np.zeros(win_len+2)     # Previous and next samples are needed for feature extraction
        # self.feat_win = np.zeros(win_len)
        # self.test_set = [[] for x in range(win_len)]
        self.test_set = []
        '''HMM variables'''
        self.n_classes = 4
        self.n_signals = 2      # Raw data and 1st-derivative
        self.startprob = [0.25, 0.25, 0.25, 0.25]
        self.state_names = ['hs', 'ff', 'ho', 'sw']
        self.state_dict = {}
        i = 0
        for name in self.state_names:
            self.state_dict[name] = i
            i += 1
        """Node initialization"""
        rospy.init_node('real-time_HMM', anonymous=True)
        self.init_subs()
        self.init_pubs()
        """HMM-training (if no model exists)"""
        try:
            '''HMM-model loading'''
            with open(packpath+'/log/HMM_models/'+patient+'.txt') as infile:
                json_model = json.load(infile)
                model = HMM.from_json(json_model)
                rospy.logwarn(patient+"'s HMM model was loaded.")
        except IOError:
            rospy.signal_shutdown("HMM model not trained yet!")
            # """Data loading"""
            # raw_data = []
            # with open(packpath+'/log/IMU_data/'+patient+'.csv') as csv_file:
            #     csv_reader = csv.reader(csv_file, delimiter=',')
            #     for row in csv_reader:
            #         raw_data.append(row[2])    # 2nd position corresponds to gyro_y data
            # """Construction of transition matrix"""
            # t = np.zeros((4, 4))        # Transition matrix
            # prev = -1
            # for i in range(0, len(labels[leave_one_out])):
            #     # data[i]._replace(label = correct_mapping[data[i].label])
            #     if prev == -1:
            #         prev = labels[leave_one_out][i]
            #     t[prev][labels[leave_one_out][i]] += 1.0
            #     prev = labels[leave_one_out][i]
            # t = normalize(t, axis=1, norm='l1')
            # if verbose: rospy.logwarn("TRANSITION MATRIX\n" + str(t))
            # """Construction of training dataset"""
            # class_data = [[] for x in range(4)]
            # full_data = []
            # for i in range(0,len(ff[leave_one_out])):
            #     full_data.append(ff[leave_one_out][i])
            # for i in range(0, len(full_data)):
            #     class_data[full_labels[i]].append(full_data[i])
            # """Multivariate Gaussian Distributions for each hidden state"""
            # class_means = [[[] for x in range(n_signals)] for i in range(n_classes)]
            # class_vars = [[[] for x in range(n_signals)] for i in range(n_classes)]
            # class_std = [[[] for x in range(n_signals)] for i in range(n_classes)]
            # class_cov = []
            # classifiers = []
            # for i in range(0, n_classes):
            #     cov = np.ma.cov(np.array(class_data[i]), rowvar=False)
            #     class_cov.append(cov)
            #     for j in range(0, n_signals):
            #         class_means[i][j] = np.array(class_data[i][:])[:, [j]].mean(axis=0)
            #         class_vars[i][j] = np.array(class_data[i][:])[:, [j]].var(axis=0)
            #         class_std[i][j] = np.array(class_data[i][:])[:, [j]].std(axis=0)
            # """Classifier initialization"""
            # distros = []
            # hmm_states = []
            # for i in range(0, n_classes):
            #     dis = MGD\
            #         (np.array(class_means[i]).flatten(),
            #          np.array(class_cov[i]))
            #     st = State(dis, name=state_names[i])
            #     distros.append(dis)
            #     hmm_states.append(st)
            # model = HMM(name="Gait")
            # model.add_states(hmm_states)
            # """Initial transitions"""
            # for i in range(0,n_classes):
            #     model.add_transition(model.start, hmm_states[i], startprob[i])
            # """Left-right model"""
            # for i in range(0, n_classes):
            #     for j in range(0, n_classes):
            #         model.add_transition(hmm_states[i], hmm_states[j], t[i][j])
            # model.bake()      # Model's final setup
            # """Training"""
            # x_train = []
            # for i in range(0,len(ff[leave_one_out-1])):
            #     x_train.append(ff[leave_one_out-1][i])
            # for i in range(0,len(ff[(leave_one_out+1) % n_trials])):
            #     x_train.append(ff[(leave_one_out+1) % n_trials][i])
            # x_train = list([x_train])
            # rospy.logwarn("Training...")
            # model.fit(x_train, algorithm='baum-welch', verbose=verbose)
            # # model.fit(seq, algorithm='viterbi', verbose='True')
            # """Save model (json script) into txt file"""
            # model_json = model.to_json()
            # try:
            #     with open(packpath+'/log/HMM_models/'+patient+'.txt', 'w') as outfile:
            #         json.dump(model_json, outfile)
            #     rospy.logwarn(patient+"'s HMM model was saved.")
            # except IOError:
            #     rospy.logwarn('It was not possible to write HMM model.')

    def init_pubs(self):
        self.phase_pub = rospy.Publisher('/phase', Int32, queue_size=10)

    def init_subs(self):
        rospy.Subscriber('/imu_data', IMUData, self.imu_callback())

    def load_local_data(self):
        n_trials = 3
        data = [[] for x in range(0,n_trials)]
        for i in range(0,n_trials):
            data[i] = scio.loadmat(datapath + prefix + "_proc_data" + str(i+1) + ".mat")
        accel_x = [[] for x in range(0, n_trials)]
        accel_y = [[] for x in range(0, n_trials)]
        accel_z = [[] for x in range(0, n_trials)]
        gyro_x = [[] for x in range(0, n_trials)]
        gyro_y = [[] for x in range(0, n_trials)]
        gyro_z = [[] for x in range(0, n_trials)]
        time_array = [[] for x in range(0, n_trials)]
        labels = [[] for x in range(0, n_trials)]
        fs_fsr = []
        for i in range(0, n_trials):
            # accel_x[i] = data[i]["accel_x"][0]
            # accel_y[i] = data[i]["accel_y"][0]
            # accel_z[i] = data[i]["accel_z"][0]
            gyro_x[i] = data[i]["gyro_x"][0]
            gyro_y[i] = data[i]["gyro_y"][0]
            gyro_z[i] = data[i]["gyro_z"][0]
            time_array[i] = data[i]["time"][0]
            labels[i] = data[i]["labels"][0]
            fs_fsr.append(data[i]["Fs_fsr"][0][0])

    """Feature extraction: 1st-derivative of angular velocity"""
    # def extract_feature(self, raw_win):
    #     fder_win = []
    #     for i in range(1,self.win_len+1):    # First and last samples not taken into account
    #         der = (raw_win[i-1]+raw_win[i+1])/2
    #         fder_win.append(der)
    #     return fder_win

    """Online approaches of Viterbi algorithm for decoding ML state sequence"""
    def online_viterbi(self, test_set):
        '''Sliding window'''
        class_labels = []
        # t = time.time()
        logp, path = model.viterbi(test_set)
        # logp, path = model.maximum_a_posteriori(test_set)
        for i in range(len(path)):
            name = path[i][1].name
            if name in self.state_dict.keys():
                class_labels.append(self.state_dict[name])
        # elapsed = time.time() - t
        # print elapsed
        if self.curr_state == -1:
            self.curr_state = class_labels[-1]
        else:
            if self.curr_state+1 in class_labels:
                self.curr_state += 1
                rospy.logwarn("Detected state: " + self.state_names[self.curr_state])

        '''Forward-only or backward-only'''
        # class_labels = []
        # win_len = 6
        # cont = 0
        # while cont+win_len < len(ff[leave_one_out]):
        #     if win_len == 1:
        #         dataset = [ff[leave_one_out][cont:cont+win_len]]
        #     else:
        #         dataset = ff[leave_one_out][cont:cont+win_len]
        #     lprob_mat = model.forward(dataset)
        #     # logp_mat = model.backward(dataset)
        #     for v in lprob_mat[1:]:
        #         # Variable initialization
        #         curr_state = 0
        #         det_state = -1
        #         max_prob = 0
        #         for lprob in v[1:-1]:     # First row is not valid, and first and last state are silent
        #             prob = np.exp(lprob)
        #             if prob > max_prob:
        #                 max_prob = prob
        #                 det_state = curr_state
        #             curr_state += 1
        #         class_labels.append(det_state)
        #     cont += win_len
        # labels[leave_one_out] = labels[leave_one_out][:-(len(labels[leave_one_out]) % win_len)]    # Remove additional elements
        # # print len(labels[leave_one_out])
        # # print len(class_labels)

        '''Forward-backward'''
        # class_labels = []
        # win_len = 6        # Window length
        # cont = 0
        # while cont+win_len < len(ff[leave_one_out]):
        #     # emis, lprob_mat = model.forward_backward(ff[leave_one_out][cont:cont+win_len])
        #     lprob_mat = model.predict_proba(ff[leave_one_out][cont:cont+win_len])
        #     # lprob_mat = model.predict_log_proba(ff[leave_one_out][cont:cont+win_len])
        #     for v in lprob_mat:
        #         max_prob = 0
        #         curr_state = 0
        #         det_state = -1
        #         for prob in v:
        #         # for lprob in v:
        #         #     prob = np.exp(lprob)
        #             if prob > max_prob:
        #                 max_prob = prob
        #                 curr_state = det_state
        #             curr_state += 1
        #         class_labels.append(state_dict[name])
        #     cont += win_len
        # labels[leave_one_out] = labels[leave_one_out][:-(len(labels[leave_one_out]) % win_len)]    # Remove additional elements

    def imu_callback(self, data):
        self.rec_data += 1
        self.raw_win.append(data.gyro_y)

        if self.rec_data >= 3:      # At least one previous and one subsequent data should have been received
            """Extract feature and append it to test dataset"""
            self.test_set.append([self.raw_win[self.proc_data+1], (self.raw_win[self.proc_data]+self.raw_win[self.proc_data+2])/2])    # First-derivate of angular velocity
            self.proc_data += 1     # One gyro data has been processed

def main():
    """Check system arguments"""
    if(len(sys.argv)<2):
        print("Missing the patient's name.")
        exit()
    else:
        patient = sys.argv[1]
    rospy.logwarn("Patient: {}".format(patient))

    """Check if patient's data exists"""
    if not os.path.isfile(packpath+"/log/IMU_data/"+patient+".csv"):
        rospy.logerr("Patient's IMU data has not been recollected")
        exit()

    rtHMM = RealTimeHMM()
    ros.spin()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
