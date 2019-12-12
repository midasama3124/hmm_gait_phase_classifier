import numpy as np
import matplotlib.pyplot as plt
import pickle
from hmmlearn import hmm

startprob = np.array([0.6, 0.3, 0.1, 0.0])
# The transition matrix, note that there are no transitions possible
# between component 1 and 3
transmat = np.array([[0.7, 0.2, 0.0, 0.1],
                     [0.3, 0.5, 0.2, 0.0],
                     [0.0, 0.3, 0.5, 0.2],
                     [0.2, 0.0, 0.2, 0.6]])
# The means of each component
means = np.array([[0.0,  0.0],
                  [0.0, 11.0],
                  [9.0, 10.0],
                  [11.0, -1.0]])
# The covariance of each component
covars = .5 * np.tile(np.identity(2), (4, 1, 1))

# Build an HMM instance and set parameters
model = hmm.GaussianHMM(n_components=4, covariance_type="full", algorithm="viterbi")

# Instead of fitting it from the data, we directly set the estimated
# parameters, the means and covariance of the components
model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars

# Data loading
try:
    with open("/home/miguel/catkin_ws/src/agora_exo/exo_gait_phase_det/log/HMM_models_hmmlearn/test.pkl", "rb") as file:
        X = pickle.load(file)
except IOError:
    X, Z = model.sample(100)
    try:
        with open("/home/miguel/catkin_ws/src/agora_exo/exo_gait_phase_det/log/HMM_models_hmmlearn/test.pkl", "wb") as file:
            pickle.dump(X,file)
    except IOError:
        print "X observations could not be saved"

# model.fit(X)
path = model.predict(X)
print path
