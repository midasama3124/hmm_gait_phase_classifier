#!/usr/bin/env python
import rospy
import rospkg
import pickle
import numpy as np
import sys
from sklearn.preprocessing import normalize
from pomegranate import*
from pomegranate.hmm import log as ln
from pomegranate import HiddenMarkovModel as HMM
from pomegranate import MultivariateGaussianDistribution as MGD
from scipy import io as scio
from scipy import linalg
from itertools import groupby
from std_msgs.msg import Int8
from exo_msgs.msg import IMUData


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
        self.raw_win = []       # Window of raw IMU data
        self.rec_data = 0       # Number of recorded IMU data
        self.proc_data = 0      # Number of extracted features
        self.ff = [[] for x in range(self.n_trials)]      # Training and test dataset
        self.labels = [[] for x in range(self.n_trials)]  # Reference labels from local data
        self.first_eval = True
        self.conv_found = False
        self.conv_point = 0
        self.last_eval = False
        """HMM variables"""
        ''' State list:
            s1: Heel Strike (HS)
            s2: Flat Foot   (FF)
            s3: Heel Off    (HO)
            s4: Swing Phase (SP)'''
        self.model_name = "Gait"
        self.states = ['s1', 's2', 's3', 's4']
        self.n_states = len(self.states)
        self.state2phase = {"s1": "hs", "s2": "ff", "s3": "ho", "s4": "sp"}
        self.mgds = {}
        self.dis_means = [[] for x in range(self.n_states)]
        self.dis_covars = [[] for x in range(self.n_states)]
        self.start_prob = [1.0/self.n_states]*self.n_states
        self.trans_mat = np.zeros([self.n_states, self.n_states])
        self.log_startprob = []
        self.log_transmat = np.empty((self.n_states, self.n_states))
        self.states_pos = {}
        for i in range(len(self.states)):
            self.states_pos[self.states[i]] = i
        self.global_path = []
        self.V = []      # Viterbi path
        self.backtrack = []
        self.work_buffer = []
        self.backtrck_buffer = []
        self.max_win_len = 22       # ms (120 ms: mean IC duration for healthy subjects walking at comfortable speed)
        self.last_state = -1
        self.model = HMM(name=self.model_name)
        """ROS init"""
        rospy.init_node('real_time_HMM', anonymous=True)
        rospack = rospkg.RosPack()
        self.packpath = rospack.get_path('exo_gait_phase_det')
        self.init_subs()
        self.init_pubs()
        """Local data loading and feature extraction"""
        self.load_data()
        """HMM-training (if no model exists)"""
        try:
            '''HMM-model loading'''
            with open(self.packpath+'/log/HMM_models/'+patient+'.txt') as infile:
                json_model = json.load(infile)
                self.model = HMM.from_json(json_model)
                rospy.logwarn(patient + "'s HMM model was loaded.")
        except IOError:
            rospy.logwarn("HMM model not trained yet!")
            self.init_hmm()
            self.train_hmm()
        try:
            '''MGDs loading'''
            for st in self.states:
                with open(self.packpath+'/log/HMM_models/'+patient+'_'+self.state2phase[st]+'.txt') as infile:
                    yaml_dis = yaml.safe_load(infile)
                    dis = MGD.from_yaml(yaml_dis)
                    self.mgds[st] = dis
                    rospy.logwarn(patient + "'s "+self.state2phase[st]+" MGC was loaded.")
                    '''Loading means and covariance matrix'''
                    self.dis_means[self.states_pos[st]] = self.mgds[st].parameters[0]
                    self.dis_covars[self.states_pos[st]] = self.mgds[st].parameters[1]
        except yaml.YAMLError as exc:
            rospy.logwarn("Not able to load distributions: " + exc)
        self.test_hmm()
        # self.eval_perform()

    """Init ROS publishers"""
    def init_pubs(self):
        self.phase_pub = rospy.Publisher('/phase', Int8, queue_size=100)

    """Init ROS subcribers"""
    def init_subs(self):
        rospy.Subscriber('/imu_data', IMUData, self.imu_callback)

    """Callback function upon arrival of IMU data"""
    def imu_callback(self, data):
        self.rec_data += 1
        self.raw_win.append(data.gyro_y)

        if self.rec_data >= 3:      # At least one previous and one subsequent data should have been received
            """Extract feature and append it to test dataset"""
            self.test_set.append([self.raw_win[self.proc_data+1], (self.raw_win[self.proc_data]+self.raw_win[self.proc_data+2])/2])    # First-derivate of angular velocity
            self.proc_data += 1     # One gyro data has been processed

    """Local data loading and feature extraction"""
    def load_data(self):
        """Data loading"""
        datapath = self.packpath + "/log/mat_files/"
        data = [[] for x in range(self.n_trials)]
        for i in range(self.n_trials):
            data[i] = scio.loadmat(datapath + self.patient + "_proc_data" + str(i+1) + ".mat")
        gyro_y = [[] for x in range(self.n_trials)]
        time_array = [[] for x in range(self.n_trials)]
        for i in range(self.n_trials):
            gyro_y[i] = data[i]["gyro_y"][0]
            time_array[i] = data[i]["time"][0]
            self.labels[i] = data[i]["labels"][0]

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

        """Create training and test data"""
        for j in range(self.n_trials):
            for k in range(len(time_array[j])):
                f_ = []
                f_.append(gyro_y[j][k])
                f_.append(fder_gyro_y[j][k])
                self.ff[j].append(f_)
        self.ff = np.array(self.ff)
        type(self.ff[self.leave_one_out])
        self.n_features = len(self.ff[0][0])

    """Init HMM if no previous training"""
    def init_hmm(self):
        rospy.logwarn("-------Leaving trial {} out-------".format(self.leave_one_out+1))

        """Transition matrix (A)"""
        '''Transition matrix from reference labels'''
        prev = -1
        for i in range(len(self.labels[self.leave_one_out])):
            if prev == -1:
                prev = self.labels[self.leave_one_out][i]
            self.trans_mat[prev][self.labels[self.leave_one_out][i]] += 1.0
            prev = self.labels[self.leave_one_out][i]
        self.trans_mat = normalize(self.trans_mat, axis=1, norm='l1')
        '''Left-right model'''
        # self.trans_mat = np.array([(0.9, 0.1, 0, 0), (0, 0.9, 0.1, 0), (0, 0, 0.9, 0.1), (0.1, 0, 0, 0.9)])
        '''Right-left-right model'''
        # self.trans_mat = np.array([0.8, 0.1, 0, 0.1], [0.1, 0.8, 0.1, 0], [0, 0.1, 0.8, 0.1], [0.1, 0, 0.1, 0.8])
        if self.verbose: rospy.logwarn("TRANSITION MATRIX\n" + str(self.trans_mat))

        class_data = [[] for x in range(self.n_states)]
        for i in range(len(self.ff[self.leave_one_out])):
            class_data[self.labels[self.leave_one_out][i]].append(self.ff[self.leave_one_out][i])
        """Multivariate Gaussian Distributions for each hidden state"""
        class_means = [[[] for x in range(self.n_features)] for i in range(self.n_states)]
        class_vars = [[[] for x in range(self.n_features)] for i in range(self.n_states)]
        class_std = [[[] for x in range(self.n_features)] for i in range(self.n_states)]
        class_cov = []
        for i in range(self.n_states):
            cov = np.ma.cov(np.array(class_data[i]), rowvar=False)
            class_cov.append(cov)
            for j in range(self.n_features):
                class_means[i][j] = np.array(class_data[i][:])[:, [j]].mean(axis=0)
                class_vars[i][j] = np.array(class_data[i][:])[:, [j]].var(axis=0)
                class_std[i][j] = np.array(class_data[i][:])[:, [j]].std(axis=0)
        """Classifier initialization"""
        distros = []
        hmm_states = []
        for i in range(self.n_states):
            dis = MGD\
                (np.array(class_means[i]).flatten(),
                 np.array(class_cov[i]))
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
        x_train = []
        for i in range(len(self.ff[self.leave_one_out-1])):
            x_train.append(self.ff[self.leave_one_out-1][i])
        for i in range(len(self.ff[(self.leave_one_out+1) % self.n_trials])):
            x_train.append(self.ff[(self.leave_one_out+1) % self.n_trials][i])
        x_train = list([x_train])
        rospy.logwarn("Training initialized model...")
        self.model.fit(x_train, algorithm='baum-welch', verbose=self.verbose)
        self.model.freeze_distributions()     # Freeze all model distributions, preventing update from ocurring
        '''Save Multivariate Gaussian Distributions into yaml file'''
        for st in self.model.states:
            if st.name != self.model_name+"-start" and st.name != self.model_name+"-end":
                dis = st.distribution
                dis_yaml = dis.to_yaml()
                try:
                    with open(self.packpath+'/log/HMM_models/'+self.patient+'_'+self.state2phase[st.name]+'.txt', 'w') as outfile:
                        yaml.dump(dis_yaml, outfile, default_flow_style=False)
                    rospy.logwarn(self.patient+"'s "+self.state2phase[st.name]+" distribution was saved.")
                except IOError:
                    rospy.logwarn('It was not possible to write HMM model.')
        '''Save model (json script) into txt file'''
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
    -------
    lpr : array_like, shape (n_samples, n_components)
        Array containing the log probabilities of each data point in
        X under each of the n_components multivariate Gaussian distributions."""
    def log_multivariate_normal_density(self, X, means, covars, min_covar=1.e-7):
        """Log probability for full covariance matrices."""
        n_samples, n_dim = X.shape
        nmix = len(means)
        log_prob = np.empty((n_samples, nmix))
        for c, (mu, cv) in enumerate(zip(means, covars)):
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

    """Find most-likely sequence"""
    def test_hmm(self):
        """Transition matrix and initial (log) probabilities upon training"""
        trans_mat = self.model.dense_transition_matrix()[:self.n_states,:self.n_states]
        for i in range(self.n_states):
            self.log_startprob.append(ln(self.start_prob[i]))
            for j in range(self.n_states):
                self.log_transmat[i,j] = ln(trans_mat[i][j])
        if self.verbose: print log_transmat
        """HMM decoding approaches"""
        '''Offline decoding approach (pomegranate library)'''
        # logp, path = self.model.viterbi(self.ff[self.leave_one_out])
        # # print logp
        # # print path
        # for i in range(len(self.labels[self.leave_one_out])):
        #     path_phase = path[i][1].name
        #     for state in range(self.n_states):
        #         if path_phase == self.states[state]:
        #             self.global_path.append(state)
        # self.labels[self.leave_one_out] = list(self.labels[self.leave_one_out][1:])
        '''Manual offline viterbi'''
        # where_from = -1
        # n_samples = len(self.ff[self.leave_one_out])
        # self.global_path = np.empty(n_samples, dtype=np.int32)
        # viterbi_lattice = np.zeros((n_samples, self.n_states))
        # work_buffer = np.empty(self.n_states)
        # '''Probability distribution of state given observation'''
        # framelogprob = self.log_multivariate_normal_density(np.array(self.ff[self.leave_one_out]), self.dis_means, self.dis_covars)
        #
        # for i in range(self.n_states):
        #     viterbi_lattice[0, i] = self.log_startprob[i] + framelogprob[0, i]
        # # Induction
        # for t in range(1, n_samples):
        #     for i in range(self.n_states):
        #         for j in range(self.n_states):
        #             work_buffer[j] = (self.log_transmat[j, i] + viterbi_lattice[t - 1, j])
        #         viterbi_lattice[t, i] = self._max(work_buffer) + framelogprob[t, i]
        # # Observation traceback
        # self.global_path[n_samples - 1] = where_from = self._argmax(viterbi_lattice[n_samples - 1])
        # logprob = viterbi_lattice[n_samples - 1, where_from]
        #
        # for t in range(n_samples - 2, -1, -1):
        #     for i in range(self.n_states):
        #         work_buffer[i] = viterbi_lattice[t, i] + self.log_transmat[i, where_from]
        #     self.global_path[t] = where_from = self._argmax(work_buffer)
        '''Offline bounded variable sliding window (BVSW)'''
        buffer_len = 35000
        verbose = False
        self.conv_point = 0
        self.n_samples = len(self.ff[self.leave_one_out])
        print self.n_samples
        self.global_path = []
        self.V = np.empty((self.max_win_len+1, self.n_states))
        self.backtrack = [[None for x in range(self.n_states)] for y in range(self.max_win_len+1)]
        self.work_buffer = np.empty(self.n_states)
        framelogprob = self.log_multivariate_normal_density(np.array(self.ff[self.leave_one_out]), self.dis_means, self.dis_covars)

        n_buffer = 0
        for i in range(self.n_states):
            self.V[0][i] = self.log_startprob[i] + framelogprob[0][i]
            self.backtrack[0][i] = None
        t = b = 1
        # while t < buffer_len:
        while t < self.n_samples:
            '''Find emission distribution'''
            # if t+b >= buffer_len:
            if t+b >= self.n_samples:
                self.last_eval = True
                b = self.n_samples - t
            '''Find likelihood probability and backpointer'''
            for j in range(self.n_states):
                for i in range(self.n_states):
                    self.work_buffer[i] = self.V[b - 1][i] + self.log_transmat[i, j]
                self.V[b][j] = self._max(self.work_buffer) + framelogprob[t+b-1][j]
                self.backtrack[b][j] = self._argmax(self.work_buffer)
            '''Backtracking local paths'''
            local_paths = [[] for x in range(self.n_states)]
            for j in range(self.n_states):
                where_from = j
                for smp in range(b-1, -1, -1):
                    local_paths[j].insert(0, self.backtrack[smp+1][where_from])
                    where_from = self.backtrack[smp+1][where_from]
            if verbose:
                print
                print t, b
                for path in local_paths:
                    print path
            '''Given all local paths, find fusion point'''
            tmp = [None] * self.n_states
            for k in range(len(local_paths[0])-1, 0, -1):
                for st in range(self.n_states):
                    tmp[st] = local_paths[st][k]
                if tmp.count(tmp[0]) == len(tmp):      # All local paths point to only one state?
                    self.conv_found = True
                    self.conv_point = k
                    if verbose:
                        print "Found"
                        print k
                    break
            '''Find local path if fusion point was found'''
            if b < self.max_win_len and self.conv_found:
                n_buffer += self.conv_point
                opt = []
                self.last_state = where_from = self._argmax(self.V[self.conv_point])
                opt.append(where_from)
                for lp in range(self.conv_point-1, -1, -1):
                    opt.insert(0, self.backtrack[lp + 1][where_from])
                    where_from = self.backtrack[lp + 1][where_from]
                self.global_path.extend(opt)
                self.conv_found = False
                if verbose: print "\nOpt1: " + str(opt) + ", {}".format(len(self.global_path))
                '''Reinitialize local variables'''
                t += self.conv_point + 1
                # b = 1
                for i in range(self.n_states):
                    if i == self.last_state:
                        self.log_startprob[i] = ln(1.0)
                    else:
                        self.log_startprob[i] = ln(0.0)
                    self.V[0][i] = self.log_startprob[i] + framelogprob[t-1][i]
                    self.backtrack[0][i] = None
                ''''''
                for smp in range(1, b-self.conv_point+1):
                    for j in range(self.n_states):
                        for i in range(self.n_states):
                            self.work_buffer[i] = self.V[smp - 1][i] + self.log_transmat[i, j]
                        self.V[smp][j] = self._max(self.work_buffer) + framelogprob[t][j]
                        self.backtrack[smp][j] = self._argmax(self.work_buffer)
                b -= self.conv_point-1
            elif (b >= self.max_win_len or self.last_eval):
                '''Bounding threshold was exceeded'''
                n_buffer += b
                opt = []
                self.last_state = where_from = self._argmax(self.V[b-1])
                opt.append(where_from)
                for lp in range(b-2, -1, -1):
                    opt.insert(0, self.backtrack[lp + 1][where_from])
                    where_from = self.backtrack[lp + 1][where_from]
                self.global_path.extend(opt)
                if self.last_eval:
                    break
                if verbose: print "\nOpt2: " + str(opt) + ", {}".format(len(self.global_path))
                '''Reinitialize local variables'''
                t += self.max_win_len
                b = 1
                for i in range(self.n_states):
                    if i == self.last_state:
                        self.log_startprob[i] = ln(1.0)
                    else:
                        self.log_startprob[i] = ln(0.0)
                    self.V[0][i] = self.log_startprob[i] + framelogprob[t-1][i]
                    self.backtrack[0][i] = None
            else:
                b += 1
        # self.labels[self.leave_one_out] = self.labels[self.leave_one_out][1:]
        print
        print "Global path length: {}".format(len(self.global_path))
        self.global_path = [x[0] for x in groupby(self.global_path)]
        print self.global_path
        '''Offline forward-only approach'''
        # self.n_samples = len(self.ff[self.leave_one_out])
        # self.global_path = np.empty(self.n_samples, dtype=np.int32)
        # viterbi_lattice = np.zeros((self.n_samples, self.n_states))
        # work_buffer = np.empty(self.n_states)
        # # Probability distribution of state given observation
        # framelogprob = self.log_multivariate_normal_density(np.array(self.ff[self.leave_one_out]), self.dis_means, self.dis_covars)
        #
        # for i in range(self.n_states):
        #     viterbi_lattice[0, i] = self.log_startprob[i] + framelogprob[0, i]
        # self.global_path[0] = self._argmax(viterbi_lattice[0])
        # # Recursion
        # for t in range(1, self.n_samples):
        #     for i in range(self.n_states):
        #         for j in range(self.n_states):
        #             work_buffer[j] = self.log_transmat[j,i] + viterbi_lattice[t - 1, j]
        #         viterbi_lattice[t, i] = self._max(work_buffer) + framelogprob[t, i]
        #     self.global_path[t] = self._argmax(viterbi_lattice[t])
        ''''''
        # for t in range(1, self.n_samples):
        #     for i in range(self.n_states):
        #         viterbi_lattice[t, i] = self._max(self.log_transmat[i] + viterbi_lattice[t - 1] + framelogprob[t, i])
        #     self.global_path[t] = self._argmax(viterbi_lattice[t])
        '''Print model attributes'''
        # self.global_path = [x[0] for x in groupby(self.global_path)]
        # print "Global length: {}".format(len(self.global_path))
        # print self.global_path[2000:2005]
        # print len(self.labels[self.leave_one_out])
        # print self.labels[self.leave_one_out][2000:2005]
        # for st in self.global_path:
        #     print st

    """Short-time Viterbi algorithm
        **Parameters:
        obs:     Observation sequence
        trans_p: Transition matrix (A)
        emit_p:  Emission matrix (B)
        **Returns:
        None:    No fusion/convergence point found
        Int []:  Decoding of samples prior to fusion point"""
    def st_viterbi(self, buffer_len, emit_p):
        '''Initial probs of Viterbi path'''
        if self.first_eval:
            for i in range(self.n_states):
                self.V[0][i] = self.log_startprob[i] + self.log_multivariate_normal_density(np.array(self.ff[self.leave_one_out]), self.dis_means, self.dis_covars)[0][i]
        elif self.conv_found:
            for i in range(self.n_states):
                if i == self.last_state:
                    self.V[0][i] = ln(1)      # Prior probability set to 1 in last state found by convergence point
                else:
                    self.V[0][i] = ln(0)
            self.conv_found = False
        # Induction of last sample
        smp = buffer_len-1
        for i in range(self.n_states):
            for j in range(self.n_states):
                self.work_buffer[j] = self.log_transmat[j, i] + self.V[smp - 1][j]
                self.backtrck_buffer[j] = self.V[smp - 1][j] + self.log_transmat[j,i]
            self.V[smp][i] = self._max(self.work_buffer) + emit_p[i]
            self.backtrack[smp][i] = self._argmax(self.backtrck_buffer)
        # print len(self.backtrack[smp])
        '''Return suboptimal path if bound exceeded'''
        if buffer_len >= self.max_win_len or self.last_eval == True:
            print "overflow"
            return self.optim_backtrack(buffer_len-1)
        '''Look for fusion/convergence point'''
        prev_loc_path = list(dict.fromkeys(self.backtrack[smp]))
        print prev_loc_path
        # if len(prev_loc_path) == 1:
        #     return self.optim_backtrack(smp)
        for t in range(smp, 0, -1):
            curr_loc_path = []
            for i in prev_loc_path:
                curr_loc_path.append(self.backtrack[t][i])
            print curr_loc_path
            curr_loc_path = list(dict.fromkeys(curr_loc_path))
            print curr_loc_path
            if len(curr_loc_path) == 1:
                opt = self.optim_backtrack(t-1)    # Partial likelihood up to fusion point
                print "FP"
                return opt
            else:
                prev_loc_path = curr_loc_path
        return None

    """Optimal backtracking upon determination of fusion point
       **Parameters:
       -conv_pos: time """
    def optim_backtrack(self, conv_pos):
        opt = []
        where_from = -1
        if self.first_eval:
            last_backtrck = -1
            self.first_eval = False
        else:
            last_backtrck = 0
        '''Highest probability at convergence point'''
        self.last_state = where_from = self._argmax(np.array(self.V[conv_pos]))
        '''Follow backtrack until first observation'''
        for t in range(conv_pos, last_backtrck, -1):
            for i in range(self.n_states):
                self.work_buffer[i] = self.V[t][i] + self.log_transmat[i, where_from]
            where_from = self._argmax(self.work_buffer)
            opt.insert(0, where_from)
        # if len(opt) > 2:
        #     print "True", str(len(opt))
        return opt

    """Performance evaluation"""
    def eval_perform(self):
        '''Vars init'''
        sum = 0.0
        true_pos = 0.0
        false_pos = 0.0
        true_neg = 0.0
        false_neg = 0.0
        tol = 6e-2       # Tolerance window of 60 ms
        fs_fsr = 200     # Hz
        tol_window = int((tol/2) / (1/float(fs_fsr)))
        if self.verbose: print "Tolerance win: " + str(tol_window)

        '''Calculate results'''
        rospy.logwarn("Calculating results...")
        for phase in range(self.n_states):
            for i in range(len(self.labels[self.leave_one_out])):
                """Tolerance window"""
                if i >= tol_window and i < len(self.labels[self.leave_one_out])-tol_window:
                    win = []
                    for j in range(i-tol_window,i+tol_window+1):
                        win.append(self.labels[self.leave_one_out][j])
                    if self.global_path[i] == phase:
                        if self.global_path[i] in win:
                            true_pos += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_path[i] + ", true_pos"
                        else:
                            false_pos += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_path[i] + ", false_pos"
                    else:
                        if phase != self.labels[self.leave_one_out][i]:
                        # if phase not in win:
                            true_neg += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_path[i] + ", true_neg"
                        else:
                            false_neg += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_path[i] + ", false_neg"
                else:
                    if self.global_path[i] == phase:
                        if self.global_path[i] == self.labels[self.leave_one_out][i]:
                            true_pos += 1.0
                        else:
                            false_pos += 1.0
                    else:
                        if phase != self.labels[self.leave_one_out][i]:
                            true_neg += 1.0
                        else:
                            false_neg += 1.0

        '''Accuracy'''
        if (true_neg+true_pos+false_neg+false_pos) != 0.0:
            acc = (true_neg + true_pos)/(true_neg + true_pos + false_neg + false_pos)
        else:
            acc = 0.0
        '''Sensitivity or True Positive Rate'''
        if true_pos+false_neg != 0:
            tpr = true_pos / (true_pos+false_neg)
        else:
            tpr = 0.0
        '''Specificity or True Negative Rate'''
        if false_pos+true_neg != 0:
            tnr = true_neg / (false_pos+true_neg)
        else:
            tnr = 0.0
        rospy.logwarn("Accuracy: {}%".format(acc*100.0))
        rospy.logwarn("Sensitivity: {}%".format(tpr*100.0))
        rospy.logwarn("Specificity: {}%".format(tnr*100.0))
        '''Goodness index'''
        G = np.sqrt((1-tpr)**2 + (1-tnr)**2)
        if G <= 0.25:
            rospy.logwarn("Optimum classifier (G = {} <= 0.25)".format(G))
        elif G > 0.25 and G <= 0.7:
            rospy.logwarn("Good classifier (0.25 < G = {} <= 0.7)".format(G))
        elif G == 0.7:
            rospy.logwarn("Random classifier (G = 0.7)")
        else:
            rospy.logwarn("Bad classifier (G = {} > 0.7)".format(G))
        # print self.log_transmat
        # print self.log_startprob


def main():
    if(len(sys.argv)<2):
        print("Missing patient's name.")
        exit()
    else:
        patient = sys.argv[1]
        RtHMM = RealTimeHMM(patient)
        rate = rospy.Rate(50)   # Hz
        last = -1
        prev = -1
        cont = 1
        threshold = 20    # Threshold of 110 ms
        prt = False

        # while not rospy.is_shutdown():
        #     for label in RtHMM.global_path:
        #         if label != prev:
        #             print label
        #             prev = label
        #         rate.sleep()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
