#!/usr/bin/env python
import rospy
import rospkg
import pickle
import entry_data as ed
import numpy as np
import time
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import normalize
from sklearn.mixture import GaussianMixture
from entry_data import DataEntry, fullEntry
from pomegranate import*
from pomegranate import HiddenMarkovModel as HMM
from pomegranate import MultivariateGaussianDistribution as MGD
from scipy import io as scio
from std_msgs.msg import Int32
from hmmlearn import hmm

def create_training_data(data_, imu, meas):
    ff_ = []
    for k in range(0, len(data_)):
        f_ = []
        for jj in range(0, len(imu)):
            if imu[jj] == 1:
                for ii in range(0, len(meas)):
                    if meas[ii] == 1:
                        if ii == 0:
                            f_.append(data_[k][jj*13])
                            f_.append(data_[k][jj*13 + 1])
                            f_.append(data_[k][jj*13 + 2])
                            f_.append(data_[k][jj*13 + 3])
                        else:
                            f_.append(data_[k][jj*13 + ii*3 + 1])
                            f_.append(data_[k][jj*13 + ii*3 + 2])
                            f_.append(data_[k][jj*13 + ii*3 + 3])
        ff_.append(f_)
    return ff_

rospy.init_node('hmm_trainer')
param_vec = []
rospack = rospkg.RosPack()
if(len(sys.argv)<3):
    print("Missing the prefix or trial argument.")
    exit()
else:
    prefix = sys.argv[1]
    trial = sys.argv[2]
use_measurements = np.zeros(3)

# patient = rospy.get_param('~patient', 'None')
# if prefix == 'None':
#     rospy.logerr("No filename given ,exiting")
#     exit()

phase_pub = rospy.Publisher('/phase', Int32, queue_size=10)
datapath = rospack.get_path('exo_util') + "/log/mat_files/"
if prefix[-1] == 's':
    rospy.logwarn("Training %s' data", prefix)
else:
    rospy.logwarn("Training %s's data", prefix)

"""Data loading"""
data = scio.loadmat(datapath + prefix + "_proc_data" + str(trial) + ".mat")
accel_x = data["accel_x"][0]
accel_y = data["accel_y"][0]
accel_z = data["accel_z"][0]
gyro_x = data["gyro_x"][0]
gyro_y = data["gyro_y"][0]
gyro_z = data["gyro_z"][0]
time_array = data["time"][0]
labels = data["labels"][0]

t = np.zeros((4, 4))        # Transition matrix
prev = -1
for i in range(0, len(labels)):
    # data[i]._replace(label = correct_mapping[data[i].label])
    if prev == -1:
        prev = labels[i]
    t[prev][labels[i]] += 1.0
    prev = labels[i]
t = normalize(t, axis=1, norm='l1')
rospy.logwarn("TRANSITION MATRIX\n" + str(t))

n_classes = 4
class_data = [[] for x in range(4)]

"""Creating training data"""
ff = []
for k in range(0, len(time_array)):
    f_ = []
    # f_.append(accel_x[k])
    # f_.append(accel_y[k])
    # f_.append(accel_z[k])
    f_.append(gyro_x[k])
    f_.append(gyro_y[k])
    f_.append(gyro_z[k])
    ff.append(f_)
n_signals = len(ff[0])
n_classes = 4

for i in range(0, len(ff)):
    class_data[labels[i]].append(ff[i])

"""Multivariate Gaussian Distributions for each hidden state"""
class_means = [[[] for x in range(n_signals)] for i in range(n_classes)]
class_vars = [[[] for x in range(n_signals)] for i in range(n_classes)]
class_std = [[[] for x in range(n_signals)] for i in range(n_classes)]
class_cov = []
classifiers = []

for i in range(0, n_classes):
    cov = np.ma.cov(np.array(class_data[i]), rowvar=False)
    class_cov.append(cov)
    for j in range(0, n_signals):
        class_means[i][j] = np.array(class_data[i][:])[:, [j]].mean(axis=0)
        class_vars[i][j] = np.array(class_data[i][:])[:, [j]].var(axis=0)
        class_std[i][j] = np.array(class_data[i][:])[:, [j]].std(axis=0)

"""Classifier initialization"""
startprob = [0.25, 0.25, 0.25, 0.25]
distros = []
hmm_states = []
state_names = ['ff', 'ho', 'sw', 'hs']
for i in range(0, n_classes):
    dis = MGD\
        (np.array(class_means[i]).flatten(),
         np.array(class_cov[i]))
    st = State(dis, name=state_names[i])
    distros.append(dis)
    hmm_states.append(st)
model = HMM(name="Gait")

model.add_states(hmm_states)
"""Initial transitions"""
for i in range(0,n_classes):
    model.add_transition(model.start, hmm_states[i], startprob[i])
"""Left-right model"""
for i in range(0, n_classes):
    for j in range(0, n_classes):
        model.add_transition(hmm_states[i], hmm_states[j], t[i][j])

model.bake()

# print (model.name)
rospy.logwarn("N. observations: " + str(model.d))
# print (model.edges)
rospy.logwarn("N. hidden states: " + str(model.silent_start))
# print model

"""Training"""
limit = int(len(ff)*(8/10.0))    # 80% of data to test, 20% to train
# seq = list([ff[:limit]])
x_train = list([ff[limit:]])
print len(x_train)
rospy.logwarn("Training...")
# time_start = time.clock()
model.fit(x_train, algorithm='baum-welch', verbose='True')
# model.fit(x_train, algorithm='viterbi', verbose='True')
# time_elapsed = time.clock()- time_start
# rospy.logwarn("Training time: " + str(time_elapsed) + "s")

"""Find most-likely sequence"""
# logp, path = model.viterbi(ff[limit:])
x_test = ff[:limit]
logp, path = model.viterbi(x_test)
sum_ = 0.0
for i in range(0, len(path)):
    if path[i][1].name != 'Gait-start':
        # if path[i][1].name == state_names[labels[i+limit - 1]]:
        if path[i][1].name == state_names[labels[i-1]]:
            # print path[i][1].name + " = " + state_names[labels[i+limit - 1]]
            sum_ += 1.0
        # else:
        #     print path[i][1].name + " != " + state_names[labels[i+limit - 1]]
# effect = sum_/float(len(ff[limit:]))
effect = sum_/float(len(x_test))
rospy.logwarn("Effectiveness: {}%".format(effect*100))

"""HMM from hmmlearn library"""
# # trellis = np.zeros((4, len(x_test)))
# # backpt = np.ones((4, len(x_test)))
# model = hmm.GMMHMM(n_components=4, n_mix=4, covariance_type="diag", n_iter=100, verbose=True)
# model.startprob_ = startprob
# model.transmat_ = t
# model.means_ = class_means
# model.covar_ = class_cov
# model.fit(x_train)
# if model.monitor_.converged:
#     print "\nHMM:\n" + str(model)
# y_train_pred = model.predict(x_train)
# y_train = labels[limit:]
# train_accuracy = np.mean(y_train_pred.ravel() == np.array(y_train).ravel()) * 100
# rospy.logwarn("\nTraining accuracy: " + str(train_accuracy) + "%")
# y_test_pred = model.predict(x_test)
# y_test = labels[:limit]
# test_accuracy = np.mean(y_test_pred.ravel() == np.array(y_test).ravel()) * 100
# rospy.logwarn("\nTraining accuracy: " + str(test_accuracy) + "%")
#
# """Generate samples"""
# X, Z = model.sample(500)

"""HMM from scikit library"""
# skf = StratifiedKFold(n_splits=4)
# train_index, test_index = next(iter(skf.split(np.zeros(len(labels)), labels)))

"""Gaussian Mixture Model
Representation of a Gaussian mixture model probability distribution.
This class allows for easy evaluation of, sampling from, and maximum-likelihood
estimation of the parameters of a GMM distribution.
**Parameters:
    -n_components (int, defaults to 1):
        Number of mixture components.
    -covariance_type ("full" (default)):
        String describing the type of covariance parameters to use. Must be one of:
            'full': each component has its own general covariance matrix.
            'tied': all components share the same general covariance matrix.
            'diag': each component has its own diagonal covariance matrix.
            'spherical': each component has its own single variance.
    -max_iter (int, defaults to 100):
        The number of EM iterations to perform."""
# classifier = GaussianMixture(n_components=4, covariance_type='diag', max_iter=100).fit(x_train)

"""Predict the labels for the data samples in X using trained model."""
# y_train_pred = classifier.predict(x_train)
# train_accuracy = np.mean(y_train_pred.ravel() == np.array(y_train).ravel()) * 100
# print "Training accuracy: " + str(train_accuracy) + "%"
# y_test_pred = classifier.predict(x_test)
# test_accuracy = np.mean(y_test_pred.ravel() == np.array(y_test).ravel()) * 100
# print "Test accuracy: " + str(test_accuracy) + "%"

"""Predict posterior probability of each component given the data."""
# prob_x = classifier.predict_proba(x_test)
