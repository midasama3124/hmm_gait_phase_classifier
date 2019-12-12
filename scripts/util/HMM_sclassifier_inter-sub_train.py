#!/usr/bin/env python
import rospy
import rospkg
import pickle
import entry_data as ed
import numpy as np
import sys
from sklearn.preprocessing import normalize
from entry_data import DataEntry, fullEntry
from pomegranate import*
from pomegranate import HiddenMarkovModel as HMM
from pomegranate import MultivariateGaussianDistribution as MGD
from scipy import io as scio
from std_msgs.msg import Int32

def main():
    rospy.init_node('hmm_trainer')
    phase_pub = rospy.Publisher('/phase', Int32, queue_size=10)
    rospack = rospkg.RosPack()
    packpath = rospack.get_path('exo_control')
    datapath = packpath + "/log/mat_files/"
    verbose = rospy.get_param('~verbose', False)

    """Print console output into text file"""
    sys.stdout = open(packpath + "/log/results/leave-one-out_cross_validation.txt", "w")

    """Data loading"""
    n_trials = 3
    n_sub = 9
    healthy_subs = ["daniel", "erika", "felipe", "jonathan", "luis", "nathalia", "paula", "pedro", "tatiana"]
    patients = ["andres", "carlos", "carmen", "carolina", "catalina", "claudia", "emmanuel", "fabian", "gustavo"]
    study_subs = [healthy_subs, patients]

    dataset = [{} for x in range(len(study_subs))]
    for i in range(len(study_subs)):
        for sub in study_subs[i]:
            dataset[i][sub] = {"gyro_y": [[] for x in range(n_trials)],
                               "fder_gyro_y": [[] for x in range(n_trials)],
                               "time": [[] for x in range(n_trials)],
                               "labels": [[] for x in range(n_trials)],
                               "Fs_fsr": 0.0}

    for group in dataset:
        for sub,data in group.iteritems():
            for trial in range(n_trials):
                mat_file = scio.loadmat(datapath + sub + "_proc_data" + str(trial+1) + ".mat")
                for signal in data:
                    if signal not in ["pathol","fder_gyro_y"]:
                        if signal == "Fs_fsr":
                            data[signal] = mat_file[signal][0][0]
                        else:
                            data[signal][trial] = mat_file[signal][0]
    del mat_file

    """Feature extraction"""
    """First derivative"""
    for group in dataset:
        for sub,data in group.iteritems():
            for trial in range(n_trials):
                der = []
                gyro_y = data["gyro_y"][trial]
                der.append(gyro_y[0])
                for i in range(1,len(gyro_y)-1):
                    der.append((gyro_y[i+1]-gyro_y[i-1])/2)
                der.append(gyro_y[-1])
                data["fder_gyro_y"][trial] = der
    del der, sub, data

    """Global variables of cHMM"""
    startprob = [0.25, 0.25, 0.25, 0.25]
    state_names = ['hs', 'ff', 'ho', 'sw']
    n_classes = 4
    n_signals = 2
    tol = 6e-2       # Tolerance window of 60 ms

    # for pathology in range(len(dataset)):
    #     if pathology == 0:
    #         rospy.logwarn("**Leave-one-out cross validation with HEALTHY subjects**")
    #         print "**Leave-one-out cross validation with HEALTHY subjects**"
    #     else:
    #         rospy.logwarn("**Leave-one-out cross validation with PATIENTS**")
    #         print "**Leave-one-out cross validation with PATIENTS**"
    if True:
        # for lou_sub,lou_data in dataset[pathology].iteritems():       # Iterate through leave-one-out subject's data
        for lou_sub,lou_data in dataset[0].iteritems():       # Iterate through leave-one-out subject's data
            rospy.logwarn("Leave " + lou_sub + " out:")
            print "Leave " + lou_sub + " out:"

            t = np.zeros((4, 4))        # Transition matrix
            prev = -1
            for trial in range(n_trials):
                for label in lou_data["labels"][trial]:
                    if prev == -1:
                        prev = label
                    t[prev][label] += 1.0
                    prev = label
            t = normalize(t, axis=1, norm='l1')
            if verbose: rospy.logwarn("TRANSITION MATRIX\n" + str(t))

            class_data = [[] for x in range(n_classes)]
            full_lou_data = []
            full_lou_labels = []
            for trial in range(n_trials):
                for sample in range(len(lou_data["gyro_y"][trial])):
                    d = [lou_data["gyro_y"][trial][sample], lou_data["fder_gyro_y"][trial][sample]]
                    l = lou_data["labels"][trial][sample]
                    full_lou_data.append(d)
                    full_lou_labels.append(l)
                    class_data[l].append(d)

            """Multivariate Gaussian Distributions for each hidden state"""
            class_means = [[[] for x in range(n_signals)] for i in range(n_classes)]
            class_vars = [[[] for x in range(n_signals)] for i in range(n_classes)]
            class_std = [[[] for x in range(n_signals)] for i in range(n_classes)]
            class_cov = []
            classifiers = []

            for state in range(n_classes):
                cov = np.ma.cov(np.array(class_data[state]), rowvar=False)
                class_cov.append(cov)
                for signal in range(n_signals):
                    class_means[state][signal] = np.array(class_data[state][:])[:, [signal]].mean(axis=0)
                    class_vars[state][signal] = np.array(class_data[state][:])[:, [signal]].var(axis=0)
                    class_std[state][signal] = np.array(class_data[state][:])[:, [signal]].std(axis=0)

            """Classifier initialization"""
            distros = []
            hmm_states = []
            for state in range(n_classes):
                dis = MGD\
                    (np.array(class_means[state]).flatten(),
                     np.array(class_cov[state]))
                st = State(dis, name=state_names[state])
                distros.append(dis)
                hmm_states.append(st)
            model = HMM(name="Gait")

            model.add_states(hmm_states)
            """Initial transitions"""
            for state in range(n_classes):
                model.add_transition(model.start, hmm_states[state], startprob[state])
            """Left-right model"""
            for i in range(n_classes):
                for j in range(n_classes):
                    model.add_transition(hmm_states[i], hmm_states[j], t[i][j])

            model.bake()

            """Create training and test data"""
            x_train = []
            x_test = []
            test_gyro_y = lou_data["gyro_y"][-1]
            test_fder_gyro_y = lou_data["fder_gyro_y"][-1]
            """Create test data with n-th trial of leave-one-out subject"""
            for sample in range(len(test_gyro_y)):
                x_test.append([test_gyro_y[sample], test_fder_gyro_y[sample]])
            """Create training data with n-1 trials of all subjects (patients group)"""
            # if pathology == 1:
            #     for trial in range(n_trials-1):
            #         train_gyro_y = lou_data["gyro_y"][trial]
            #         train_fder_gyro_y = lou_data["fder_gyro_y"][trial]
            #         for sample in range(len(train_gyro_y)):
            #             x_train.append([train_gyro_y[sample], train_fder_gyro_y[sample]])

            """Create training data with n-1 trials of the rest of subjects (healthy group)"""
            # for train_sub,train_data in dataset[pathology].iteritems():
            for train_sub,train_data in dataset[0].iteritems():
                # if lou_sub != train_sub:
                if True:
                    for trial in range(n_trials-1):
                        train_gyro_y = train_data["gyro_y"][trial]
                        train_fder_gyro_y = train_data["fder_gyro_y"][trial]
                        for sample in range(len(train_gyro_y)):
                            x_train.append([train_gyro_y[sample], train_fder_gyro_y[sample]])
            x_train = list([x_train])

            """Training"""
            rospy.logwarn("Training HMM...")
            model.fit(x_train, algorithm='baum-welch', verbose=True)
            # model.fit(x_train, algorithm='viterbi', verbose='True')

            """Find most-likely sequence"""
            logp, path = model.viterbi(x_test)
            class_labels = []
            for i in range(len(lou_data["labels"][-1])):
                path_phase = path[i][1].name
                for state in range(n_classes):
                    if path_phase == state_names[state]:
                        class_labels.append(state)
            # Saving classifier labels into csv file
            np.savetxt(packpath+"/log/inter_labels/"+lou_sub+"_labels.csv", class_labels, delimiter=",", fmt='%s')
            rospy.logwarn("csv file with classifier labels was saved.")
            lou_data["labels"][-1] = lou_data["labels"][-1][1:]

            """Results"""
            sum = 0.0
            true_pos = 0.0
            false_pos = 0.0
            true_neg = 0.0
            false_neg = 0.0
            tol_window = int((tol/2) / (1/float(lou_data["Fs_fsr"])))

            rospy.logwarn("Calculating results...")
            time_error = [[] for x in range(n_classes)]
            for phase in range(n_classes):
                for i in range(len(lou_data["labels"][-1])):
                    """Tolerance window"""
                    if i >= tol_window and i < len(lou_data["labels"][-1])-tol_window:
                        win = []
                        for win_label in lou_data["labels"][-1][i-tol_window:i+tol_window+1]:
                            win.append(win_label)
                        if class_labels[i] == phase:
                            if class_labels[i] in win:
                                for k in range(len(win)):
                                    if win[k] == phase:
                                        time_error[phase].append((k-tol_window)/lou_data["Fs_fsr"])
                                        break
                                true_pos += 1.0
                                if verbose: print phase + ", " + lou_data["labels"][-1][i] + ", " + class_labels[i] + ", true_pos"
                            else:
                                false_pos += 1.0
                                if verbose: print phase + ", " + lou_data["labels"][-1][i] + ", " + class_labels[i] + ", false_pos"
                        else:
                            if phase != lou_data["labels"][-1][i]:
                            # if phase not in win:
                                true_neg += 1.0
                                if verbose: print phase + ", " + lou_data["labels"][-1][i] + ", " + class_labels[i] + ", true_neg"
                            else:
                                false_neg += 1.0
                                if verbose: print phase + ", " + lou_data["labels"][-1][i] + ", " + class_labels[i] + ", false_neg"
                    else:
                        if class_labels[i] == phase:
                            if class_labels[i] == lou_data["labels"][-1][i]:
                                true_pos += 1.0
                            else:
                                false_pos += 1.0
                        else:
                            if phase != lou_data["labels"][-1][i]:
                                true_neg += 1.0
                            else:
                                false_neg += 1.0

            rospy.logwarn("Timing error")
            print "Timing error"
            for phase in range(n_classes):
                rospy.logwarn("(" + state_names[phase] + ")")
                print "(" + state_names[phase] + ")"
                if len(time_error[phase]) > 0:
                    rospy.logwarn(str(np.mean(time_error[phase])) + " + " + str(np.std(time_error[phase])))
                    print str(np.mean(time_error[phase])) + " + " + str(np.std(time_error[phase]))
                else:
                    rospy.logwarn("0.06 + 0")
                    print "0.06 + 0"

            """Calculate mean time (MT) of stride and each gait phase and Coefficient of Variation (CoV)"""
            rospy.logwarn("Mean time (MT) and Coefficient of Variance (CoV)")
            print "Mean time (MT) and Coefficient of Variance (CoV)"
            n_group = 0
            for label_group in [class_labels, lou_data["labels"][-1]]:
                if n_group == 0:
                    rospy.logwarn("Results for HMM:")
                    print "Results for HMM:"
                else:
                    rospy.logwarn("Results for FSR:")
                    print "Results for FSR:"
                curr_label = -1
                count = 0
                n_phases = 0
                stride_samples = 0
                phases_time = [[] for x in range(n_classes)]
                stride_time = []
                for label in label_group:
                # for label in class_labels:
                    if curr_label != label:
                        n_phases += 1
                        stride_samples += count
                        if label == 0:  # Gait start: HS
                            if n_phases == 4:   # If a whole gait cycle has past
                                stride_time.append(stride_samples/lou_data["Fs_fsr"])
                            n_phases = 0
                            stride_samples = 0
                        phases_time[label-1].append(count/lou_data["Fs_fsr"])
                        curr_label = label
                        count = 1
                    else:
                        count += 1.0
                for phase in range(n_classes):
                    mean_time = np.mean(phases_time[phase])
                    phase_std = np.std(phases_time[phase])
                    rospy.logwarn("(" + state_names[phase] + ")")
                    print "(" + state_names[phase] + ")"
                    rospy.logwarn("Mean time: " + str(mean_time) + " + " + str(phase_std))
                    print "Mean time: " + str(mean_time) + " + " + str(phase_std)
                    rospy.logwarn("CoV: " + str(phase_std/mean_time*100.0))
                    print("CoV: " + str(phase_std/mean_time*100.0))
                mean_time = np.mean(stride_time)
                phase_std = np.std(stride_time)
                rospy.logwarn("(Stride)")
                print "(Stride)"
                rospy.logwarn("Mean time: " + str(mean_time) + " + " + str(phase_std))
                print "Mean time: " + str(mean_time) + " + " + str(phase_std)
                rospy.logwarn("CoV: " + str(phase_std/mean_time*100.0))
                print("CoV: " + str(phase_std/mean_time*100.0))
                n_group += 1

            """Accuracy"""
            if (true_neg+true_pos+false_neg+false_pos) != 0.0:
                acc = (true_neg + true_pos)/(true_neg + true_pos + false_neg + false_pos)
            else:
                acc = 0.0
            """Sensitivity or True Positive Rate"""
            if true_pos+false_neg != 0:
                tpr = true_pos / (true_pos+false_neg)
            else:
                tpr = 0.0
            """Specificity or True Negative Rate"""
            if false_pos+true_neg != 0:
                tnr = true_neg / (false_pos+true_neg)
            else:
                tnr = 0.0
            rospy.logwarn("Accuracy: {}%".format(acc*100.0))
            print("Accuracy: {}%".format(acc*100.0))
            rospy.logwarn("Sensitivity: {}%".format(tpr*100.0))
            print("Sensitivity: {}%".format(tpr*100.0))
            rospy.logwarn("Specificity: {}%".format(tnr*100.0))
            print("Specificity: {}%".format(tnr*100.0))
            """Goodness index"""
            G = np.sqrt((1-tpr)**2 + (1-tnr)**2)
            if G <= 0.25:
                rospy.logwarn("Optimum classifier (G = {} <= 0.25)".format(G))
                print("Optimum classifier (G = {} <= 0.25)".format(G))
            elif G > 0.25 and G <= 0.7:
                rospy.logwarn("Good classifier (0.25 < G = {} <= 0.7)".format(G))
                print("Good classifier (0.25 < G = {} <= 0.7)".format(G))
            elif G == 0.7:
                rospy.logwarn("Random classifier (G = 0.7)")
                print("Random classifier (G = 0.7)")
            else:
                rospy.logwarn("Bad classifier (G = {} > 0.7)".format(G))
                print("Bad classifier (G = {} > 0.7)".format(G))

    del test_gyro_y, test_fder_gyro_y, train_gyro_y, train_fder_gyro_y, d, l


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
