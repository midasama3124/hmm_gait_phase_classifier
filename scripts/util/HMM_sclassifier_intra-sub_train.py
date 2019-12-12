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

def main():
    rospy.init_node('hmm_trainer')
    param_vec = []
    rospack = rospkg.RosPack()
    if(len(sys.argv)<2):
        print("Missing the prefix argument.")
        exit()
    else:
        prefix = sys.argv[1]
    use_measurements = np.zeros(3)

    # patient = rospy.get_param('~patient', 'None')
    # if prefix == 'None':
    #     rospy.logerr("No filename given ,exiting")
    #     exit()

    phase_pub = rospy.Publisher('/phase', Int32, queue_size=10)
    packpath = rospack.get_path('exo_gait_phase_det')
    datapath = packpath + "/log/mat_files/"
    rospy.logwarn("Patient: {}".format(prefix))
    print("Patient: {}".format(prefix))
    verbose = rospy.get_param('~verbose', False)

    """Print console output into text file"""
    # sys.stdout = open(packpath + "/log/results/intra-sub_" + prefix + ".txt", "w")

    """Data loading"""
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

    """Feature extraction"""
    """First derivative"""
    # fder_gyro_x = []
    # for i in range(n_trials):
    #     der = []
    #     der.append(gyro_x[i][0])
    #     for j in range(1,len(gyro_x[i])-1):
    #         der.append((gyro_x[i][j+1]-gyro_x[i][j-1])/2)
    #     der.append(gyro_x[i][-1])
    #     fder_gyro_x.append(der)

    fder_gyro_y = []
    for i in range(n_trials):
        der = []
        der.append(gyro_y[i][0])
        for j in range(1,len(gyro_y[i])-1):
            der.append((gyro_y[i][j+1]-gyro_y[i][j-1])/2)
        der.append(gyro_y[i][-1])
        fder_gyro_y.append(der)

    # fder_gyro_z = []
    # for i in range(n_trials):
    #     der = []
    #     der.append(gyro_z[i][0])
    #     for j in range(1,len(gyro_z[i])-1):
    #         der.append((gyro_z[i][j+1]-gyro_z[i][j-1])/2)
    #     der.append(gyro_z[i][-1])
    #     fder_gyro_z.append(der)

    """Second derivative"""
    # sder_gyro_x = []
    # for i in range(n_trials):
    #     der = []
    #     der.append(fder_gyro_x[i][0])
    #     for j in range(1,len(fder_gyro_x[i])-1):
    #         der.append((fder_gyro_x[i][j+1]-fder_gyro_x[i][j-1])/2)
    #     der.append(fder_gyro_x[i][-1])
    #     sder_gyro_x.append(der)
    #
    # sder_gyro_y = []
    # for i in range(n_trials):
    #     der = []
    #     der.append(fder_gyro_y[i][0])
    #     for j in range(1,len(fder_gyro_y[i])-1):
    #         der.append((fder_gyro_y[i][j+1]-fder_gyro_y[i][j-1])/2)
    #     der.append(fder_gyro_y[i][-1])
    #     sder_gyro_y.append(der)
    #
    # sder_gyro_z = []
    # for i in range(n_trials):
    #     der = []
    #     der.append(fder_gyro_z[i][0])
    #     for j in range(1,len(fder_gyro_z[i])-1):
    #         der.append((fder_gyro_z[i][j+1]-fder_gyro_z[i][j-1])/2)
    #     der.append(fder_gyro_z[i][-1])
    #     sder_gyro_z.append(der)

    """Peak detector"""
    # window_wid = 15        # Window width should be odd
    # search_ratio = window_wid/2
    # pdet_gyro_x = []
    # for i in range(n_trials):
    #     pdet = []
    #     for j in range(len(gyro_x[i])):
    #         if j <= search_ratio:
    #             win = gyro_x[i][:j+search_ratio+1]
    #         elif j >= len(gyro_x[i])-search_ratio-1:
    #             win = gyro_x[i][j-search_ratio:]
    #         else:
    #             win = gyro_x[i][j-search_ratio:j+search_ratio+1]
    #         pdet.append(gyro_x[i][j]/max(win))
    #     pdet_gyro_x.append(pdet)

    # print len(gyro_x)
    # print len(pdet_gyro_x)
    # for i in range(3):
    #     print len(gyro_x[i])
    #     print len(pdet_gyro_x[i])

    # pdet_gyro_y = []
    # for i in range(n_trials):
    #     pdet = []
    #     for j in range(len(gyro_y[i])):
    #         if j <= search_ratio:
    #             win = gyro_y[i][:j+search_ratio+1]
    #         elif j >= len(gyro_y[i])-search_ratio-1:
    #             win = gyro_y[i][j-search_ratio:]
    #         else:
    #             win = gyro_y[i][j-search_ratio:j+search_ratio+1]
    #         pdet.append(gyro_y[i][j]/max(win))
    #     pdet_gyro_y.append(pdet)
    #
    # pdet_gyro_z = []
    # for i in range(n_trials):
    #     pdet = []
    #     for j in range(len(gyro_z[i])):
    #         if j <= search_ratio:
    #             win = gyro_z[i][:j+search_ratio+1]
    #         elif j >= len(gyro_z[i])-search_ratio-1:
    #             win = gyro_z[i][j-search_ratio:]
    #         else:
    #             win = gyro_z[i][j-search_ratio:j+search_ratio+1]
    #         pdet.append(gyro_z[i][j]/max(win))
    #     pdet_gyro_z.append(pdet)

    """Create training and test data"""
    ff = [[] for x in range(0, n_trials)]
    for j in range(0, n_trials):
        for k in range(0, len(time_array[j])):
            f_ = []
            # f_.append(accel_x[j][k])
            # f_.append(accel_y[j][k])
            # f_.append(accel_z[j][k])
            # f_.append(gyro_x[j][k])
            # f_.append(fder_gyro_x[j][k])
            # f_.append(sder_gyro_x[j][k])
            # f_.append(pdet_gyro_x[j][k])
            f_.append(gyro_y[j][k])
            f_.append(fder_gyro_y[j][k])
            # f_.append(sder_gyro_y[j][k])
            # f_.append(pdet_gyro_y[j][k])
            # f_.append(gyro_z[j][k])
            # f_.append(fder_gyro_z[j][k])
            # f_.append(sder_gyro_z[j][k])
            # f_.append(pdet_gyro_z[j][k])
            ff[j].append(f_)
    n_signals = len(ff[0][0])

    """cHMM"""
    startprob = [0.25, 0.25, 0.25, 0.25]
    state_names = ['hs', 'ff', 'ho', 'sw']
    rospy.logwarn("""Intra-subject training""")
    print("""Intra-subject training""")
    # for leave_one_out in range(0, n_trials):
    for leave_one_out in range(1, 2):
        rospy.logwarn("-------TRIAL {}-------".format(leave_one_out+1))
        print("-------TRIAL {}-------".format(leave_one_out+1))

        """Transition matrix"""
        t = np.zeros((4, 4))        # Transition matrix
        prev = -1
        for i in range(0, len(labels[leave_one_out])):
            # data[i]._replace(label = correct_mapping[data[i].label])
            if prev == -1:
                prev = labels[leave_one_out][i]
            t[prev][labels[leave_one_out][i]] += 1.0
            prev = labels[leave_one_out][i]
        t = normalize(t, axis=1, norm='l1')
        if verbose: rospy.logwarn("TRANSITION MATRIX\n" + str(t))

        n_classes = 4
        class_data = [[] for x in range(n_classes)]
        full_data = []
        full_labels = []
        for i in range(len(ff[leave_one_out])):
            full_data.append(ff[leave_one_out][i])
            full_labels.append(labels[leave_one_out][i])
        # print full_data == ff[leave_one_out]
        # print full_labels == labels[leave_one_out]
        # print len(full_data) == len(full_labels)
        # for i in range(0,len(ff[leave_one_out-1])):
        #     full_data.append(ff[leave_one_out-1][i])
        #     full_labels.append(labels[leave_one_out-1][i])
        # for i in range(0,len(ff[(leave_one_out+1) % n_trials])):
        #     full_data.append(ff[(leave_one_out+1) % n_trials][i])
        #     full_labels.append(labels[(leave_one_out+1) % n_trials][i])

        # print len(full_data) == (len(ff[leave_one_out]) + len(ff[leave_one_out-1]) + len(ff[(leave_one_out+1) % n_trials]))
        # print full_data
        # print len(full_data)
        # print full_labels
        # print len(full_labels)

        for i in range(0, len(full_data)):
            class_data[full_labels[i]].append(full_data[i])

        """Multivariate Gaussian Distributions for each hidden state"""
        class_means = [[[] for x in range(n_signals)] for i in range(n_classes)]
        class_vars = [[[] for x in range(n_signals)] for i in range(n_classes)]
        class_std = [[[] for x in range(n_signals)] for i in range(n_classes)]
        class_cov = []
        classifiers = []

        for i in range(0, n_classes):
            # cov = np.ma.cov(np.array(class_data[i]), rowvar=False)
            cov = np.cov(np.array(class_data[i]), rowvar=False)
            class_cov.append(cov)
            for j in range(0, n_signals):
                class_means[i][j] = np.array(class_data[i][:])[:, [j]].mean(axis=0)
                class_vars[i][j] = np.array(class_data[i][:])[:, [j]].var(axis=0)
                class_std[i][j] = np.array(class_data[i][:])[:, [j]].std(axis=0)
        print "\n" + str(class_cov) + "\n"

        """Classifier initialization"""
        distros = []
        hmm_states = []
        for i in range(n_classes):
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
        # rospy.logwarn("N. observations: " + str(model.d))
        # print (model.edges)
        # rospy.logwarn("N. hidden states: " + str(model.silent_start))
        # print model

        """Training"""
        # limit = int(len(ff1)*(8/10.0))    # 80% of data to test, 20% to train
        # x_train = list([ff1[:limit]])
        # x_train = list([ff1,ff2])
        # x_train = list([ff2])
        x_train = []
        for i in range(0,len(ff[leave_one_out-1])):
            x_train.append(ff[leave_one_out-1][i])
        for i in range(0,len(ff[(leave_one_out+1) % n_trials])):
            x_train.append(ff[(leave_one_out+1) % n_trials][i])
        x_train = list([x_train])
        rospy.logwarn("Training...")
        model.fit(x_train, algorithm='baum-welch', verbose=verbose)
        # model.fit(list([ff[leave_one_out-1]]), algorithm='baum-welch', verbose=verbose)
        # model.fit(list([ff[(leave_one_out+1) % n_trials]]), algorithm='baum-welch', verbose=verbose)
        # model.fit(seq, algorithm='viterbi', verbose='True')

        """Find most-likely sequence"""
        # logp, path = model.viterbi(ff[limit:])
        logp, path = model.viterbi(ff[leave_one_out])
        # print logp
        # print path
        class_labels = []
        for i in range(len(labels[leave_one_out])):
            path_phase = path[i][1].name
            for state in range(n_classes):
                if path_phase == state_names[state]:
                    class_labels.append(state)
        labels[leave_one_out] = list(labels[leave_one_out][1:])
        # Saving classifier labels into csv file
        # np.savetxt(packpath+"/log/intra_labels/"+prefix+"_labels"+str(leave_one_out+1)+".csv", class_labels, delimiter=",", fmt='%s')
        # rospy.logwarn("csv file with classifier labels was saved.")

        sum = 0.0
        true_pos = 0.0
        false_pos = 0.0
        true_neg = 0.0
        false_neg = 0.0
        tol = 6e-2       # Tolerance window of 60 ms
        tol_window = int((tol/2) / (1/float(fs_fsr[leave_one_out])))
        print "FSR freq: " + str(fs_fsr[leave_one_out])
        print "Tolerance win: " + str(tol_window)
        # print tol_window
        # # print type(tol_window)
        # for i in range(0, len(labels[leave_one_out])):
        #     """Tolerance window"""
        #     if i > tol_window+1 and i < len(labels[leave_one_out])-tol_window:
        #         # curr_tol = time_array[leave_one_out][i+tol_window]-time_array[leave_one_out][i-tol_window]
        #         # print curr_tol
        #         win = []
        #         for j in range(i-tol_window,i+tol_window+1):
        #             win.append(state_names[labels[leave_one_out][j]])
        #         if path[i][1].name in win:
        #             sum += 1.0
        #     else:
        #         if path[i][1].name == labels[leave_one_out][i]:
        #             sum += 1.0

        """Performance Evaluation"""
        rospy.logwarn("Calculating results...")
        time_error = [[] for x in range(n_classes)]
        for phase in range(n_classes):
            for i in range(len(labels[leave_one_out])):
                """Tolerance window"""
                if i >= tol_window and i < len(labels[leave_one_out])-tol_window:
                    # curr_tol = time_array[leave_one_out][i+tol_window]-time_array[leave_one_out][i-tol_window]
                    # print curr_tol
                    win = []
                    for j in range(i-tol_window,i+tol_window+1):
                        win.append(labels[leave_one_out][j])
                    """Calculate time error with true positives"""
                    if class_labels[i] == phase:
                        if class_labels[i] in win:
                            for k in range(len(win)):
                                if win[k] == phase:
                                    time_error[phase].append((k-tol_window)/fs_fsr[leave_one_out])
                                    break
                            true_pos += 1.0
                            if verbose: print phase + ", " + state_names[labels[leave_one_out][i]] + ", " + class_labels[i] + ", true_pos"
                        else:
                            false_pos += 1.0
                            if verbose: print phase + ", " + state_names[labels[leave_one_out][i]] + ", " + class_labels[i] + ", false_pos"
                    else:
                        if phase != labels[leave_one_out][i]:
                        # if phase not in win:
                            true_neg += 1.0
                            if verbose: print phase + ", " + state_names[labels[leave_one_out][i]] + ", " + class_labels[i] + ", true_neg"
                        else:
                            false_neg += 1.0
                            if verbose: print phase + ", " + state_names[labels[leave_one_out][i]] + ", " + class_labels[i] + ", false_neg"
                else:
                    if class_labels[i] == phase:
                        if class_labels[i] == labels[leave_one_out][i]:
                            true_pos += 1.0
                        else:
                            false_pos += 1.0
                    else:
                        if phase != labels[leave_one_out][i]:
                            true_neg += 1.0
                        else:
                            false_neg += 1.0

        rospy.logwarn("Timing error")
        print("Timing error")
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
        print("Mean time (MT) and Coefficient of Variance (CoV)")
        n_group = 0
        for label_group in [class_labels, labels[leave_one_out]]:
            if n_group == 0:
                rospy.logwarn("Results for HMM:")
                print("Results for HMM:")
            else:
                rospy.logwarn("Results for FSR:")
                print("Results for FSR:")
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
                            stride_time.append(stride_samples/fs_fsr[leave_one_out])
                        n_phases = 0
                        stride_samples = 0
                    phases_time[label-1].append(count/fs_fsr[leave_one_out])
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
        # acc = sum/len(labels[leave_one_out])
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
        # rospy.logwarn("Accuracy: {}%".format(acc*100))
        rospy.logwarn("Accuracy: {}%".format(acc*100.0))
        # print("Accuracy: {}%".format(acc*100.0))
        rospy.logwarn("Sensitivity: {}%".format(tpr*100.0))
        # print("Sensitivity: {}%".format(tpr*100.0))
        rospy.logwarn("Specificity: {}%".format(tnr*100.0))
        # print("Specificity: {}%".format(tnr*100.0))
        """Goodness index"""
        G = np.sqrt((1-tpr)**2 + (1-tnr)**2)
        if G <= 0.25:
            rospy.logwarn("Optimum classifier (G = {} <= 0.25)".format(G))
            # print("Optimum classifier (G = {} <= 0.25)".format(G))
        elif G > 0.25 and G <= 0.7:
            rospy.logwarn("Good classifier (0.25 < G = {} <= 0.7)".format(G))
            # print("Good classifier (0.25 < G = {} <= 0.7)".format(G))
        elif G == 0.7:
            rospy.logwarn("Random classifier (G = 0.7)")
            # print("Random classifier (G = 0.7)")
        else:
            rospy.logwarn("Bad classifier (G = {} > 0.7)".format(G))
            # print("Bad classifier (G = {} > 0.7)".format(G))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
