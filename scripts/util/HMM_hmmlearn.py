#!/usr/bin/env python
import rospy
import rospkg
import pickle
import numpy as np
import sys
from sklearn.preprocessing import normalize
from sklearn.datasets.samples_generator import make_spd_matrix
from sklearn.utils import check_random_state
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import GMMHMM
from sklearn.mixture import GaussianMixture
from pomegranate import MultivariateGaussianDistribution as MGD
from scipy import io as scio
from std_msgs.msg import Int8
from exo_msgs.msg import IMUData

"""Real-time HMM class"""
class RealTimeHMM():
    def __init__(self, patient, n_trials=3, leave_one_out=1, verbose=False):
        """Variable initialization"""
        self.patient = patient
        self.n_trials = n_trials
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
        self.last_eval = False
        """HMM variables"""
        ''' State list:
            s1: Heel Strike (HS)
            s2: Flat Foot   (FF)
            s3: Heel Off    (HO)
            s4: Swing Phase (SP)'''
        self.states = ['s1', 's2', 's3', 's4']
        self.state2phase = {"s1": "hs", "s2": "ff", "s3": "ho", "s4": "sp"}
        self.gauss_distros = []
        self.n_states = len(self.states)
        self.start_prob = np.array([0.25, 0.25, 0.25, 0.25])
        self.trans_mat = np.zeros([self.n_states, self.n_states])
        self.states_pos = {}
        for i in range(len(self.states)):
            self.states_pos[self.states[i]] = i
        self.global_opt = []
        self.V = [{}]      # Viterbi path
        self.max_win_len = 11       # ms (120 ms: mean IC duration for healthy subjects walking at comfortable speed)
        self.covariance_type = "full"
        '''GaussianHMM'''
        self.model = GaussianHMM(n_components=self.n_states, covariance_type=self.covariance_type)
        # self.model = GaussianHMM(n_components=self.n_states, covariance_type=self.covariance_type, params="cm", init_params="cm")
        '''GMMHMM'''
        # self.model = GMMHMM(n_components=self.n_states, covariance_type=self.covariance_type, covars_prior=1.0, params="mct", init_params="cm")
        """ROS init"""
        rospy.init_node('real_time_HMM', anonymous=True)
        rospack = rospkg.RosPack()
        self.packpath = rospack.get_path('exo_gait_phase_det')
        print self.packpath
        self.init_subs()
        self.init_pubs()
        """Local data loading and feature extraction"""
        self.load_data()
        """HMM-training (if no model exists)"""
        # try:
        #     '''HMM-model loading'''
        #     with open(self.packpath+'/log/HMM_models_hmmlearn/'+patient+'.pkl', "rb") as infile:
        #         self.model = pickle.load(infile)
        #         rospy.logwarn(patient + "'s HMM model was loaded.")
        # except IOError:
            # rospy.logwarn("HMM model not trained yet!")
        # WARNING: Indent the two following lines when uncomment previous ones
        self.init_hmm()
        self.train_hmm()
        self.test_hmm()
        self.eval_perform()

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
        self.n_features = len(self.ff[0][0])

    """Create random covariance matrix"""
    def make_covar_matrix(self, covariance_type, n_components, n_features, random_state=None):
        mincv = 0.1
        prng = check_random_state(random_state)
        if covariance_type == 'spherical':
            return (mincv + mincv * prng.random_sample((n_components,))) ** 2
        elif covariance_type == 'tied':
            return (make_spd_matrix(n_features) + mincv * np.eye(n_features))
        elif covariance_type == 'diag':
            return (mincv + mincv * prng.random_sample((n_components, n_features))) ** 2
        elif covariance_type == 'full':
            return np.array([
                (make_spd_matrix(n_features, random_state=prng)
                 + mincv * np.eye(n_features))
                for x in range(n_components)
            ])

    """Init HMM if no previous training"""
    def init_hmm(self):
        rospy.logwarn("-------Leaving trial {} out-------".format(self.leave_one_out+1))
        class_data = [[] for x in range(self.n_states)]
        for i in range(len(self.ff[self.leave_one_out])):
            class_data[self.labels[self.leave_one_out][i]].append(self.ff[self.leave_one_out][i])

        """Transition matrix (A)"""
        """As in online example"""
        # for i in range(self.n_states):
        #     if i == self.n_states - 1:
        #         self.trans_mat[i, i] = 1.0
        #     else:
        #         self.trans_mat[i, i] = self.trans_mat[i, i + 1] = 0.5
        '''Drawn from reference labels'''
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
        if self.verbose: rospy.logwarn("TRANSITION MATRIX\n" + str(self.trans_mat) + "\n")

        """Means and covariance matrix"""
        class_means = [[[] for x in range(self.n_features)] for i in range(self.n_states)]
        class_cov = []
        # class_cov = self.make_covar_matrix(self.covariance_type, self.n_states, self.n_features)
        for i in range(self.n_states):
            cov = np.cov(np.array(class_data[i]), rowvar=False)
            class_cov.append(cov)
            for j in range(self.n_features):
                class_means[i][j] = np.array(class_data[i][:])[:, [j]].mean(axis=0)[0]
                # class_means[i][j] = np.array(class_data[i][:])[:, [j]].mean(axis=0)
        # print len(class_means[0])
        # print class_means
        # print "-----"
        # print len(class_cov[0])
        # print class_cov
        '''Gaussian Mixture Model for each hidden state'''
        # for st in range(self.n_states):
        #     # dis = GaussianMixture(covariance_type=self.covariance_type).fit(class_data[st])
        #     dis = GaussianMixture(covariance_type=self.covariance_type)
        #     dis.means_ = np.array([class_means[st]])
        #     dis.covariances_ = np.array([class_cov[st]])
        #     # dis = MGD(np.array(class_means[st]).flatten(), np.array(class_cov[st]))
        #     # dis.covars_ = self.make_covar_matrix(self.covariance_type, 1, self.n_features)
        #     self.gauss_distros.append(dis)
        # # for dis in gauss_distros:
        # #     print dis
        #     # print str(dis.means_) + "\n"
        #     # print str(dis.covariances_) + "\n"
        """Classifier initialization"""
        self.model.startprob_ = self.start_prob.copy()
        self.model.transmat_ = self.trans_mat.copy()
        '''GaussianHMM'''
        self.model.means_ = class_means
        self.model.covars_ = class_cov
        # # self.model._check()     # Check mathematical correctness of covariances and means
        '''GMMHMM'''
        # self.model.gmms_ = self.gauss_distros
        # print self.model.gmms_
        '''Print model atributes'''
        print self.model.startprob_
        print self.model.transmat_
        # print self.model.means_
        # print self.model.covars_

    """Train initialized model"""
    def train_hmm(self):
        X1 = self.ff[self.leave_one_out-1]
        X2 = self.ff[(self.leave_one_out+1) % self.n_trials]
        x_train = np.concatenate([X1, X2])
        lengths = [len(X1), len(X2)]
        rospy.logwarn("Training initialized model...")
        self.model.fit(x_train, lengths)
        self.model._check()
        '''Print model atributes'''
        print "\n" + str(self.model.startprob_)
        print self.model.transmat_
        print self.model.means_
        print self.model.covars_
        '''Save model into pickle file'''
        try:
            with open(self.packpath+'/log/HMM_models_hmmlearn/'+self.patient+'.pkl', "wb") as outfile:
                pickle.dump(self.model, outfile)
            rospy.logwarn(self.patient+"'s HMM model was saved.")
        except IOError:
            rospy.logwarn('It was not possible to write HMM model.')

    """Find most-likely sequence"""
    def test_hmm(self):
        '''Offline decoding (viterbi algorithm)'''
        x_test = self.ff[self.leave_one_out]
        score, self.global_opt = self.model.decode(x_test, algorithm="viterbi")
        self.global_opt = list(self.global_opt)
        assert np.isfinite(score), "Viterbi path did not converge"
        # model_score = self.model.score(x_test)
        # print model_score
        '''Bounded variable sliding window'''
        # t = 0
        # buffer_len = 6
        # global_opt = []
        # cont = 0
        # n_buffer = 0
        # while t < len(self.ff[self.leave_one_out]):
        #     if t+buffer_len >= len(self.ff[self.leave_one_out]):
        #         obs = self.ff[self.leave_one_out][t:]
        #         self.last_eval = True
        #     else:
        #         obs = self.ff[self.leave_one_out][t : t+buffer_len]
        #     emiss_prob = self.model.predict_proba(obs)
        #     opt = self.st_viterbi(obs, trans_prob, emiss_prob)
        #     if opt:
        #         for det_st in opt:
        #             global_opt.append(det_st)
        #         t += buffer_len              # start next decoding from last convergence point
        #         cont += buffer_len
        #         n_buffer += 1
        #         buffer_len = 4               # restart buffer window length
        #         self.conv_found = True
        #         self.first_eval = False
        #     else:
        #         self.conv_found = False
        #         buffer_len += 1
        '''Forward-only decoding approach'''
        # emiss_prob = {}
        # for smp in range(len(self.ff[self.leave_one_out])):
        #     obs = self.ff[self.leave_one_out][smp]
        #     for st in self.states:
        #         # emiss_prob[st] = self.gauss_dis[st].probability(obs)
        #         emiss_prob[st] = self.gauss_dis[st].log_probability(obs)
        #     self.fo_viterbi(smp, trans_prob, emiss_prob)
        '''Print decoding results'''
        # print self.V[0]
        # print len(self.global_opt)
        # print self.global_opt[1000:1500]
        # print "------------------"
        # print self.labels[self.leave_one_out][1000:1500]
        # for st in range(self.n_states):
        #     print st + ": " + str(self.global_opt.count(st))
        # print cont
        # print cont/n_buffer
        # print len(self.ff[self.leave_one_out])
        # assert np.allclose(self.global_opt, self.labels[self.leave_one_out])

    """Forward-only Viterbi algorithm
    **Parameters:
    smp: Sample in observation dataset
    trans_p: Dense transition probability distribution (A)
    emit_p:  Emission probability distribution of single observation (B)
    **Returns:
    None:    No fusion/convergence point found
    Int []:  Decoding of samples prior to fusion point"""
    def fo_viterbi(self, smp, trans_p, emit_p):
        '''Initial probs of Viterbi path'''
        if smp == 0:
            self.V = [{} for x in range(len(self.ff[self.leave_one_out]))]
            for st in self.states:
                self.V[0][st] = self.start_prob[self.states_pos[st]] * emit_p[st]
            max_prob = 0
            for st in self.states:
                loc_prob = self.V[0][st]
                if loc_prob >= max_prob:
                    max_prob = loc_prob
                    loc_state = st
            self.global_opt.append(st)
        else:
            for i in self.states:
                max_prob = 0
                for j in self.states:
                    loc_prob = self.V[smp-1][i] * trans_p[i][j]
                    if loc_prob > max_prob:
                        max_prob = loc_prob
                self.V[smp][i] = max_prob * emit_p[i]
            max_prob = 0
            for st in self.states:
                if self.V[smp][st] >= max_prob:
                    max_prob = self.V[smp][st]
                    loc_state = st
            self.global_opt.append(loc_state)

    """Short-time Viterbi algorithm
        **Parameters:
        obs:     Observation sequence
        trans_p: Transition matrix (A)
        emit_p:  Emission matrix (B)
        **Returns:
        None:    No fusion/convergence point found
        Int []:  Decoding of samples prior to fusion point"""
    def st_viterbi(self, obs, trans_p, emit_p):
        '''Initial probs of Viterbi path'''
        if self.first_eval:
            self.V = [{}]
            for st in self.states:
                self.V[0][st] = {"prob": self.start_prob[self.states_pos[st]] * emit_p[0][self.states_pos[st]], "backtrack": None}
        elif self.conv_found:
            self.V = [self.V[-1]]      # Keep probabilities of recently found convergence point
            self.conv_found = False
        else:
            self.V = [self.V[0]]       # Use probabilities of last found convergence point
        '''Run Viterbi when t > 0'''
        for t in range(1, len(obs)):
            self.V.append({})
            for st in self.states:
                max_tr_prob = 0
                for k in self.states:
                    tr_prob = self.V[t-1][k]["prob"]*trans_p[k][st]
                    if tr_prob >= max_tr_prob:
                        max_tr_prob = tr_prob
                        prev_st_selected = k
                max_prob = max_tr_prob * emit_p[t][self.states_pos[st]]
                self.V[t][st] = {"prob": max_prob, "backtrack": prev_st_selected}
        '''Return suboptimal path if bound exceeded'''
        if len(obs) >= self.max_win_len or self.last_eval == True:
            return self.optim_backtrack(len(obs)-1)
        '''Print log probability matrix'''
        if self.verbose:
            for line in dptable():
                print line
        '''Backtracking fusion/convergence point'''
        opt = None
        '''Initial backtrackers'''
        backtracks = []
        for st in self.states:
            backtracks.append(self.V[-1][st]["backtrack"])
        backtracks = list(dict.fromkeys(backtracks))    # Remove duplicates
        if len(backtracks) == 1:    # Fusion point
            return self.optim_backtrack(len(obs)-1)     # Partial likelihood up to fusion point
        for t in range(len(self.V)-2, -1, -1):
            backtracks = [self.V[t][btrck]["backtrack"] for btrck in backtracks]
            backtracks = list(dict.fromkeys(backtracks))    # Remove duplicates
            if len(backtracks) == 1:    # Fusion point
                opt = self.optim_backtrack(t)     # Partial likelihood up to fusion point
                break
        return opt

    """Optimal backtracking upon determination of fusion point
       **Parameters:
       -conv_point: time """
    def optim_backtrack(self, conv_point):
        opt = []
        '''Highest probability at convergence point'''
        max_prob = max(value["prob"] for value in self.V[conv_point].values())
        prev = None
        '''Get backtrack of most probable state'''
        for st, data in self.V[conv_point].items():
            if data["prob"] == max_prob:
                prev = st
        '''Follow backtrack until first observation'''
        for t in range(conv_point-1, -1, -1):
            opt.insert(0, self.V[t+1][prev]["backtrack"])
            prev = self.V[t+1][prev]["backtrack"]
        return opt

    def dptable(self):
        # Print a table of steps from dictionary
        yield " ".join(("%12d" % i) for i in range(len(V)))
        for state in self.V[0]:
            yield "%.7s: " % state + " ".join("%.7s" % ("%f" % v[state]["prob"]) for v in self.V)

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
                    if self.global_opt[i] == phase:
                        if self.global_opt[i] in win:
                            true_pos += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_opt[i] + ", true_pos"
                        else:
                            false_pos += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_opt[i] + ", false_pos"
                    else:
                        if phase != self.labels[self.leave_one_out][i]:
                        # if phase not in win:
                            true_neg += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_opt[i] + ", true_neg"
                        else:
                            false_neg += 1.0
                            if self.verbose: print phase + ", " + self.state2phase[self.states[self.labels[self.leave_one_out][i]]] + ", " + self.global_opt[i] + ", false_neg"
                else:
                    if self.global_opt[i] == phase:
                        if self.global_opt[i] == self.labels[self.leave_one_out][i]:
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

def main():
    if(len(sys.argv)<2):
        print("Missing patient's name.")
        exit()
    else:
        patient = sys.argv[1]
        RtHMM = RealTimeHMM(patient)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
