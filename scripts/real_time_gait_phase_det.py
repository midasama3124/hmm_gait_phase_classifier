#!/usr/bin/env python
import rospy
import rospkg
import pickle
import numpy as np
import sys, os
from sklearn.preprocessing import normalize
from pomegranate import *
from pomegranate.hmm import log as ln
from pomegranate import HiddenMarkovModel as HMM
from pomegranate import MultivariateGaussianDistribution as MGD
from scipy import io as scio
from scipy import linalg
from std_msgs.msg import Int8
from sensor_msgs.msg import Imu
import csv

"""Supported decoder algorithms"""
DECODER_ALGORITHMS = frozenset(("fov", "bvsw"))   # FOV: Forward-only viterbi, BVSW: Bounded Variable Sliding Window Approach

"""Real-time HMM class"""
class RealTimeHMM():
    def __init__(self, n_trials=3, leave_one_out=1):
        """Variable initialization"""
        self.patient = rospy.get_param("gait_phase_det/patient")
        self.verbose = rospy.get_param("gait_phase_det/verbose")
        self.n_trials = n_trials
        self.n_features = 2      # Raw data and 1st-derivative
        self.leave_one_out = leave_one_out
        self.rec_data = 0.0       # Number of recorded IMU data
        self.proc_data = 0.0      # Number of extracted features
        self.win_size = 3
        self.raw_win = [None] * self.win_size
        # self.fder_win = [0] * self.win_size
        self.ff = [[] for x in range(self.n_trials)]      # Training and test dataset
        self.labels = [[] for x in range(self.n_trials)]  # Reference labels from local data
        self.first_eval = True
        self.model_loaded = False
        algorithm = rospy.get_param("gait_phase_det/algorithm")
        rospy.loginfo('Decoding algorithm: {}'.format(algorithm))
        if algorithm not in DECODER_ALGORITHMS:
            raise ValueError("Unknown decoder {!r}".format(algorithm))
        self.decode = {
            "fov": self._run_fov,
            "bvsw": self._run_bvsw
        }[algorithm]
        self.imu_callback = {
            "fov": self._fov_callback,
            "bvsw": self._bvsw_callback
        }[algorithm]
        """HMM variables"""
        ''' State list:
            s1: Heel Strike (HS)
            s2: Flat Foot   (FF)
            s3: Heel Off    (HO)
            s4: Swing Phase (SP)'''
        self.model_name = "Gait"
        self.has_model = False
        self.must_train = False
        self.states = ['s1', 's2', 's3', 's4']
        self.n_states = len(self.states)
        self.state2phase = {"s1": "hs", "s2": "ff", "s3": "ho", "s4": "sp"}
        self.train_data = []
        self.mgds = {}
        self.dis_means = [[] for x in range(self.n_states)]
        self.dis_covars = [[] for x in range(self.n_states)]
        self.start_prob = [1.0/self.n_states]*self.n_states
        self.trans_mat = np.array([(0.9, 0.1, 0, 0), (0, 0.9, 0.1, 0), (0, 0, 0.9, 0.1), (0.1, 0, 0, 0.9)])    # Left-right model
        # self.trans_mat = np.array([0.8, 0.1, 0, 0.1], [0.1, 0.8, 0.1, 0], [0, 0.1, 0.8, 0.1], [0.1, 0, 0.1, 0.8])    # Left-right-left model
        self.log_startprob = []
        self.log_transmat = np.empty((self.n_states, self.n_states))
        self.max_win_len = 11       # ms (120 ms: mean IC duration for healthy subjects walking at comfortable speed)
        self.viterbi_path = np.empty((self.max_win_len+1, self.n_states))
        self.backtrack = [[None for x in range(self.n_states)] for y in range(self.max_win_len+1)]
        self.global_path = []
        self.work_buffer = np.empty(self.n_states)
        self.boundary = 1
        self.buff_len = 0
        self.states_pos = {}
        for i in range(len(self.states)):
            self.states_pos[self.states[i]] = i
        self.last_state = -1
        self.curr_state = -1
        self.conv_point = 0
        self.conv_found = False
        self.smp_freq = 100.0   # Hz
        self.fp_thresh = 1/self.smp_freq*4    # Threshold corresponds to 8 samples
        self.time_passed = 0.0
        self.obs = [[None for x in range(self.n_features)] for y in range(self.max_win_len)]
        self.model = HMM(name=self.model_name)
        """ROS init"""
        rospy.init_node('real_time_HMM', anonymous=True)
        rospack = rospkg.RosPack()
        self.packpath = rospack.get_path('hmm_gait_phase_classifier')
        self.init_subs()
        self.init_pubs()
        """HMM-training (if no model exists)"""
        try:
            '''HMM-model loading'''
            with open(self.packpath+'/log/HMM_models/'+self.patient+'.txt') as infile:
                json_model = json.load(infile)
                self.model = HMM.from_json(json_model)
                rospy.logwarn(self.patient + "'s HMM model was loaded.")
                self.has_model = True
        except IOError:
            if os.path.isfile(self.packpath + "/log/mat_files/" + self.patient + "_proc_data1.mat"):
                """Training with data collected with FSR-based reference system"""
                self.data_ext = 'mat'
                self.must_train = True
            elif os.path.isfile(self.packpath + "/log/IMU_data/" + self.patient + "_labels.csv"):
                """Training with data collected with offline threshold-based gait phase detection method"""
                self.data_ext = 'csv'
                self.must_train = True
            else:
                rospy.logerr("Please collect data for training ({})!".format(self.patient))
        if self.must_train:
            rospy.logwarn("HMM model not trained yet for {}!".format(self.patient))
            rospy.logwarn("Training HMM with local data...")
            self.load_data()
            self.init_hmm()
            self.train_hmm()
            self.has_model = True
        if self.has_model:
            try:
                '''MGDs loading if model exists'''
                for st in self.states:
                    with open(self.packpath+'/log/HMM_models/'+self.patient+'_'+self.state2phase[st]+'.txt') as infile:
                        yaml_dis = yaml.safe_load(infile)
                        dis = MGD.from_yaml(yaml_dis)
                        self.mgds[st] = dis
                        rospy.logwarn(self.patient +"'s " + self.state2phase[st] + " MGC was loaded.")
                        '''Loading means and covariance matrix'''
                        self.dis_means[self.states_pos[st]] = self.mgds[st].parameters[0]
                        self.dis_covars[self.states_pos[st]] = self.mgds[st].parameters[1]
            except yaml.YAMLError as exc:
                rospy.logwarn("Not able to load distributions: " + exc)
            """Transition and initial (log) probabilities matrices upon training"""
            trans_mat = self.model.dense_transition_matrix()[:self.n_states,:self.n_states]
            if self.verbose: print '**TRANSITION MATRIX (post-training)**\n'+ str(trans_mat)
            for i in range(self.n_states):
                self.log_startprob.append(ln(self.start_prob[i]))
                for j in range(self.n_states):
                    self.log_transmat[i,j] = ln(trans_mat[i][j])
            self.model_loaded = True

    """Init ROS publishers"""
    def init_pubs(self):
        self.phase_pub = rospy.Publisher('/gait_phase', Int8, queue_size=100)

    """Init ROS subcribers"""
    def init_subs(self):
        rospy.Subscriber('/imu_data', Imu, self.imu_callback)

    """Callback function upon arrival of IMU data for forward-only decoding"""
    def _fov_callback(self, data):
        self.rec_data += 1.0
        self.raw_win.append(data.angular_velocity.y)
        self.raw_win.pop(0)       # Drop first element

        if self.rec_data >= self.win_size and self.model_loaded:      # At least one previous and one subsequent data should have been received
            """Extract feature and append it to test dataset"""
            fder = (self.raw_win[self.win_size/2 + 1] - self.raw_win[self.win_size/2 - 1])/2
            # peak_detector = self.raw_win[self.win_size/2]/max(self.raw_win)
            # self.fder_win.append(fder)
            # self.fder_win.pop(0)
            # sder = (self.fder_win[2] - self.fder_win[0])/2
            # test_set = [self.raw_win[self.win_size/2], self.raw_win[self.win_size/2 - 2], self.raw_win[self.win_size/2 - 1], self.raw_win[self.win_size/2 + 1], self.raw_win[self.win_size/2 + 2]]         # Temporally proximal features
            test_set = [self.raw_win[self.win_size/2], fder]         # Temporally proximal features
            '''Forward-only decoding approach'''
            state = self.decode(test_set)
            # rospy.loginfo("Decoded phase: {}".format(state))
            self.time_passed += 1/self.smp_freq
            if self.last_state == state:
                if (self.time_passed >= self.fp_thresh) and (self.curr_state == 3 and state == 0) or (state == self.curr_state + 1):
                    self.curr_state = state
                    self.phase_pub.publish(state)
                else:
                    # rospy.loginfo("Current phase: {}".format(self.state2phase[self.states[self.curr_state]]))
                    # self.phase_pub.publish((self.curr_state-1)%4)
                    self.phase_pub.publish(self.curr_state)
            else:
                self.last_state = state
                self.time_passed = 1/self.smp_freq
                # rospy.logwarn("Detected phase: {}".format(self.state2phase[self.states[self.last_state]]))

    """Callback function upon arrival of IMU data for BVSW"""
    def _bvsw_callback(self, data):
        self.rec_data += 1.0
        self.raw_win.append(data.angular_velocity.y)
        self.raw_win.pop(0)       # Drop first element

        if self.rec_data >= 3 and self.model_loaded:      # At least one previous and one subsequent data should have been received
            """Extract feature and append it to test dataset"""
            test_set = [self.raw_win[1], (self.raw_win[0]+self.raw_win[2])/2]    # First-derivate of angular velocity
            '''Bounded sliding window decoding approach'''
            self.obs.append(test_set)
            self.obs.pop(0)                     # This way, -1 element corresponds to last received features
            states = self.decode(test_set)
            if len(states) != 0:
                for st in states:
                    self.phase_pub.publish(st)
                    if self.curr_state != st:
                        rospy.logwarn("Detected phase: {}".format(self.state2phase[self.states[st]]))
                        self.curr_state = st
                self.proc_data += 1.0     # One gyro data has been processed

    """Local data loading from mat file and feature extraction"""
    def load_data(self):
        """Data loading"""
        '''Load mat file (Data processed offline in Matlab)'''
        if self.data_ext == 'mat':
            rospy.logwarn("Initializing parameters via FSR-based reference system")
            # subs = ['daniel', 'erika', 'felipe', 'jonathan', 'luis', 'nathalia', 'paula', 'pedro', 'tatiana']      # Healthy subjects
            # subs = ['carmen', 'carolina', 'catalina', 'claudia', 'emmanuel', 'fabian', 'gustavo']      # Pathological subjects
            # subs = ['daniel']      # Single subject
            datapath = self.packpath + "/log/mat_files/"
            gyro_y = [[] for x in range(self.n_trials)]
            time_array = [[] for x in range(self.n_trials)]
            # for patient in subs:
            for i in range(self.n_trials):
                data = scio.loadmat(datapath + self.patient + "_proc_data" + str(i+1) + ".mat")
                # data = scio.loadmat(datapath + patient + "_proc_data" + str(i+1) + ".mat")
                gyro_y[i] = data["gyro_y"][0]
                # gyro_y[i] += list(data["gyro_y"][0])
                time_array[i] = data["time"][0]
                # time_array[i] += list(data["time"][0])
                self.labels[i] = data["labels"][0]
                # self.labels[i] += list(data["labels"][0])

            """Feature extraction"""
            '''First derivative'''
            fder_gyro_y = []
            for i in range(self.n_trials):
                der = []
                der.append(gyro_y[i][0])
                for j in range(1,len(gyro_y[i])-1):
                    der.append((gyro_y[i][j+1]-gyro_y[i][j-1])/2)
                der.append(gyro_y[i][-1])
                fder_gyro_y.append(der)

            '''Second derivative'''
            # sder_gyro_y = []
            # for i in range(self.n_trials):
            #     der = []
            #     der.append(fder_gyro_y[i][0])
            #     for j in range(1,len(fder_gyro_y[i])-1):
            #         der.append((fder_gyro_y[i][j+1]-fder_gyro_y[i][j-1])/2)
            #     der.append(fder_gyro_y[i][-1])
            #     sder_gyro_y.append(der)

            '''Peak detector'''
            # peak_detector = []
            # for i in range(self.n_trials):
            #     win = []
            #     for j in range(len(gyro_y[i])):
            #         if (j - self.win_size/2) < 0:
            #             win.append(gyro_y[i][j] / self._max(gyro_y[i][0:j + self.win_size/2]))
            #         elif (j + self.win_size/2) > (len(gyro_y[i])-1):
            #             win.append(gyro_y[i][j] / self._max(gyro_y[i][j - self.win_size/2:len(gyro_y[i])-1]))
            #         else:
            #             win.append(gyro_y[i][j] / self._max(gyro_y[i][j - self.win_size/2:j + self.win_size/2]))
            #     peak_detector.append(win)

            """Create training data"""
            for j in range(self.n_trials):
                # for k in range(self.win_size/2,len(gyro_y[j])-1-self.win_size/2):
                for k in range(len(gyro_y[j])):
                    f_ = []
                    f_.append(gyro_y[j][k])
                    '''Approximate differentials'''
                    f_.append(fder_gyro_y[j][k])
                    # f_.append(sder_gyro_y[j][k])
                    '''Temporally proximal feature'''
                    # f_.append(gyro_y[j][k-1])
                    # f_.append(gyro_y[j][k-2])
                    # f_.append(gyro_y[j][k+1])
                    # f_.append(gyro_y[j][k+2])
                    '''Peak detector'''
                    # f_.append(peak_detector[j][k])
                    self.ff[j].append(f_)
            self.ff = np.array(self.ff)
            self.n_features = len(self.ff[0][0])

            for i in range(len(self.ff[self.leave_one_out-1])):
                self.train_data.append(self.ff[self.leave_one_out-1][i])
            for i in range(len(self.ff[(self.leave_one_out+1) % self.n_trials])):
                self.train_data.append(self.ff[(self.leave_one_out+1) % self.n_trials][i])
            self.train_data = [self.train_data]       # Keep this line or training won't work

        """Parameter initialization"""
        '''Kmeans clustering'''
        # n_components = 4       # No. components of MGD
        # init = 'kmeans++'      # "kmeans||", "first-k", "random"
        # n_init = 1
        # weights = None
        # max_kmeans_iterations = 1
        # batch_size = None
        # batches_per_epoch = None
        #
        # X_concat = x_train
        #
        # # X_concat = numpy.concatenate(x_train)
        # # if X_concat.ndim == 1:
        # #     X_concat = X_concat.reshape(X_concat.shape[0], 1)
        # # n, d = X_concat.shape
        #
        # rospy.logwarn("K-means clustering...")
        # clf = Kmeans(n_components, init=init, n_init=n_init)
        # clf.fit(X_concat, weights, max_iterations=max_kmeans_iterations,
        #     batch_size=batch_size, batches_per_epoch=batches_per_epoch)
        # y = clf.predict(X_concat)
        #
        # rospy.logwarn("Creating distributions...")
        # class_data = [[] for x in range(self.n_states)]
        # for i in range(len(x_train)):
        #     class_data[y[i]].append(x_train[i])
        # if self.verbose:
        #     sum = 0
        #     print "Kmeans clusters:"
        #     for i in range(self.n_states):
        #         temp = len(class_data[i])
        #         sum += temp
        #         print temp
        #     print sum, len(x_train)

        """Offline threshold-based classification"""
        if self.data_ext == 'csv':
            rospy.logwarn("Initializing parameters via threshold-based gait phase detection method")
            self.leave_one_out = 0
            gyro_y = []
            datapath = self.packpath + '/log/IMU_data/'
            with open(datapath+'{}.csv'.format(self.patient), 'r') as imu_file:
                reader = csv.DictReader(imu_file)
                for row in reader:
                    gyro_y.append(float(dict(row)['gyro_y']))
            '''Feature extraction: 1st derivative'''
            for i in range(1, len(gyro_y)-1):
                feature_vect = [gyro_y[i], (gyro_y[i+1]-gyro_y[i-1])/2]
                self.train_data.append(feature_vect)
                self.ff[self.leave_one_out].append(feature_vect)
            '''Reference labels'''
            with open(datapath+'{}_labels.csv'.format(self.patient), 'r') as labels_file:
                reader = csv.reader(labels_file)
                for row in reader:
                    self.labels[self.leave_one_out].append(int(row[0]))
            self.train_data = [self.train_data]

    """Init HMM if no previous training"""
    def init_hmm(self):
        if self.data_ext == 'mat':
            rospy.logwarn("-------Leaving trial {} out-------".format(self.leave_one_out+1))

        """Transition matrix (A)"""
        '''Transition matrix from reference labels'''
        # prev = -1
        # for i in range(len(self.labels[self.leave_one_out])):
        #     if prev == -1:
        #         prev = self.labels[self.leave_one_out][i]
        #     self.trans_mat[prev][self.labels[self.leave_one_out][i]] += 1.0
        #     prev = self.labels[self.leave_one_out][i]
        # self.trans_mat = normalize(self.trans_mat, axis=1, norm='l1')
        if self.verbose: rospy.logwarn("**TRANSITION MATRIX (pre-training)**\n" + str(self.trans_mat) + '\nMatrix type: {}'.format(type(self.trans_mat)))

        class_data = [[] for x in range(self.n_states)]
        for i in range(len(self.ff[self.leave_one_out])):
            class_data[self.labels[self.leave_one_out][i]].append(self.ff[self.leave_one_out][i])

        """Multivariate Gaussian Distributions for each hidden state"""
        class_means = [[[] for x in range(self.n_features)] for i in range(self.n_states)]
        # class_vars = [[[] for x in range(self.n_features)] for i in range(self.n_states)]
        # class_std = [[[] for x in range(self.n_features)] for i in range(self.n_states)]
        class_cov = []
        for i in range(self.n_states):
            cov = np.ma.cov(np.array(class_data[i]), rowvar=False)
            class_cov.append(cov)
            for j in range(self.n_features):
                class_means[i][j] = np.array(class_data[i][:])[:, [j]].mean(axis=0)
                # class_vars[i][j] = np.array(class_data[i][:])[:, [j]].var(axis=0)
                # class_std[i][j] = np.array(class_data[i][:])[:, [j]].std(axis=0)
        """Classifier initialization"""
        distros = []
        hmm_states = []
        for i in range(self.n_states):
            dis = MGD(np.array(class_means[i]).flatten(), np.array(class_cov[i]))
            st = State(dis, name=self.states[i])
            distros.append(dis)
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
        if self.verbose: print "**HMM model:\n{}**".format(self.model)

        """Save Multivariate Gaussian Distributions into yaml file"""
        for st in self.model.states:
            if st.name != self.model_name+"-start" and st.name != self.model_name+"-end":
                dis = st.distribution
                dis_yaml = dis.to_yaml()
                try:
                    with open(self.packpath+'/log/HMM_models/'+self.patient+'_'+self.state2phase[st.name]+'.txt', 'w') as outfile:
                        yaml.dump(dis_yaml, outfile, default_flow_style=False)
                    rospy.logwarn(self.patient+"'s "+self.state2phase[st.name]+" distribution was saved.")
                except IOError:
                    rospy.logwarn('It was not possible to write GMM distribution.')
        """Save model (json script) into txt file"""
        model_json = self.model.to_json()
        try:
            with open(self.packpath+'/log/HMM_models/'+self.patient+'.txt', 'w') as outfile:
                json.dump(model_json, outfile)
            rospy.logwarn(self.patient+"'s HMM model was saved.")
        except IOError:
            rospy.logwarn('It was not possible to write HMM model.')

    """Compute the log probability under a multivariate Gaussian distribution.
    Parameters
    ----------
        X : array_like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row corresponds to a
            single data point.
        means : array_like, shape (n_components, n_features)
            List of n_features-dimensional mean vectors for n_components Gaussians.
            Each row corresponds to a single mean vector.
        covars : array_like
            List of n_components covariance parameters for each Gaussian. The shape
            is (n_components, n_features, n_features) if 'full'
        Returns
    ----------
        lpr : array_like, shape (n_samples, n_components)
            Array containing the log probabilities of each data point in
            X under each of the n_components multivariate Gaussian distributions."""
    def log_multivariate_normal_density(self, X, min_covar=1.e-7):
        """Log probability for full covariance matrices."""
        n_samples, n_dim = X.shape
        nmix = len(self.dis_means)
        log_prob = np.empty((n_samples, nmix))
        for c, (mu, cv) in enumerate(zip(self.dis_means, self.dis_covars)):
            try:
                cv_chol = linalg.cholesky(cv, lower=True)
            except linalg.LinAlgError:
                # The model is most probably stuck in a component with too
                # few observations, we need to reinitialize this components
                try:
                    cv_chol = linalg.cholesky(cv + min_covar * np.eye(n_dim),
                                              lower=True)
                except linalg.LinAlgError:
                    raise ValueError("'covars' must be symmetric, "
                                     "positive-definite")

            cv_log_det = 2 * np.sum(np.log(np.diagonal(cv_chol)))
            cv_sol = linalg.solve_triangular(cv_chol, (X - mu).T, lower=True).T
            log_prob[:, c] = - .5 * (np.sum(cv_sol ** 2, axis=1) +
                                     n_dim * np.log(2 * np.pi) + cv_log_det)
        return log_prob

    """Find argument (pos) that has the maximum value in array-like object"""
    def _argmax(self, X):
        X_max = float("-inf")
        pos = 0
        for i in range(X.shape[0]):
            if X[i] > X_max:
                X_max = X[i]
                pos = i
        return pos

    """Find max value in array-like object"""
    def _max(self, X):
        return X[self._argmax(X)]

    """Backtracking process to decode most-likely state sequence"""
    def _optim_backtrack(self, k):
        opt = []
        self.last_state = where_from = self._argmax(self.viterbi_path[k])
        opt.append(where_from)
        for lp in range(k-1, -1, -1):
            opt.insert(0, self.backtrack[lp + 1][where_from])
            where_from = self.backtrack[lp + 1][where_from]
        self.global_path.extend(opt)
        return opt

    '''Forward-only decoding approach'''
    def _run_fov(self, test_set):
        # Probability distribution of state given observation
        framelogprob = self.log_multivariate_normal_density(np.array([test_set]))
        if self.first_eval:
            for i in range(self.n_states):
                self.viterbi_path[0, i] = self.log_startprob[i] + framelogprob[0, i]
            self.first_eval = False
            return self._argmax(self.viterbi_path[0])
        else:       # Recursion
            for i in range(self.n_states):
                for j in range(self.n_states):
                    self.work_buffer[j] = (self.log_transmat[j, i] + self.viterbi_path[0, j])
                self.viterbi_path[1, i] = self._max(self.work_buffer) + framelogprob[0, i]
            self.viterbi_path[0] = self.viterbi_path[1]       # Prepare for next feature vector
            return self._argmax(self.viterbi_path[1])

    '''Bounded sliding variable window approach'''
    def _run_bvsw(self, test_set):
        framelogprob = self.log_multivariate_normal_density(np.array([test_set]))
        if self.first_eval:
            for i in range(self.n_states):
                self.viterbi_path[0,i] = self.log_startprob[i] + framelogprob[0,i]
                self.backtrack[0][i] = None
            self.first_eval = False
            return []
        else:
            '''Find likelihood probability and backpointer'''
            for j in range(self.n_states):
                for i in range(self.n_states):
                    self.work_buffer[i] = self.viterbi_path[self.boundary - 1][i] + self.log_transmat[i, j]
                self.viterbi_path[self.boundary][j] = self._max(self.work_buffer) + framelogprob[0][j]
                self.backtrack[self.boundary][j] = self._argmax(self.work_buffer)
            '''Backtracking local paths'''
            local_paths = [[] for x in range(self.n_states)]
            for j in range(self.n_states):
                where_from = j
                for smp in range(self.boundary-1, -1, -1):
                    local_paths[j].insert(0, self.backtrack[smp+1][where_from])
                    where_from = self.backtrack[smp+1][where_from]
            # if self.verbose:
            #     print "\n{}, {}".format(t, b)
            #     for path in local_paths:
            #         print path
            '''Given all local paths, find fusion point'''
            tmp = [None] * self.n_states
            for k in range(len(local_paths[0])-1, 0, -1):
                for st in range(self.n_states):
                    tmp[st] = local_paths[st][k]
                if tmp.count(tmp[0]) == len(tmp):      # All local paths point to only one state?
                    self.conv_found = True
                    self.conv_point = k
                    # if self.verbose: print "Found, {}".format(k)
                    break
            '''Find local path if fusion point was found'''
            if self.boundary < self.max_win_len and self.conv_found:
                self.buff_len += self.conv_point
                opt = self._optim_backtrack(self.conv_point)
                self.conv_found = False
                # if self.verbose: print "\nOpt1: " + str(opt) + ", {}".format(len(self.global_path))
                '''Reinitialize local variables'''
                for i in range(self.n_states):
                    if i == self.last_state:
                        self.log_startprob[i] = ln(1.0)
                    else:
                        self.log_startprob[i] = ln(0.0)
                    self.viterbi_path[0][i] = self.log_startprob[i] + framelogprob[0][i]
                    self.backtrack[0][i] = None
                for smp in range(1, self.boundary-self.conv_point+1):
                    for j in range(self.n_states):
                        for i in range(self.n_states):
                            self.work_buffer[i] = self.viterbi_path[smp - 1][i] + self.log_transmat[i, j]
                        self.viterbi_path[smp][j] = self._max(self.work_buffer) + self.log_multivariate_normal_density(np.array([self.obs[self.conv_point-self.boundary+smp-1]]))[0,j]
                        self.backtrack[smp][j] = self._argmax(self.work_buffer)
                self.boundary -= self.conv_point-1
                return opt
            elif self.boundary >= self.max_win_len:
                '''Bounding threshold was exceeded'''
                self.buff_len += self.max_win_len
                opt = self._optim_backtrack(self.boundary-1)
                # if self.verbose: print "\nOpt2: " + str(opt) + ", {}".format(len(self.global_path))
                '''Reinitialize local variables'''
                self.boundary = 1
                for i in range(self.n_states):
                    if i == self.last_state:
                        self.log_startprob[i] = ln(1.0)
                    else:
                        self.log_startprob[i] = ln(0.0)
                    self.viterbi_path[0][i] = self.log_startprob[i] + framelogprob[0][i]
                    self.backtrack[0][i] = None
                return opt
            else:
                self.boundary += 1
                return []


def main():
    # if(len(sys.argv)<2):
    #     print("Missing patient's name.")
    #     exit()
    # else:
    #     patient = sys.argv[1]

    RtHMM = RealTimeHMM()
    if RtHMM.model_loaded:
        rospy.logwarn("Spinning...")
        rospy.spin()
        # if RtHMM.rec_data != 0:
        #     print "Rata of processed data: " + str(RtHMM.proc_data/RtHMM.rec_data)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
