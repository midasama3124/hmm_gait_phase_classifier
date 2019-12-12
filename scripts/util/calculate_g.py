import numpy as np
import csv
import os
import sys
from scipy import io as scio

def main():
    """Color output"""
    class bcolors:
        HEADER = '\033[95m'
        OKBLUE = '\033[94m'
        OKGREEN = '\033[92m'
        WARNING = '\033[93m'
        FAIL = '\033[91m'
        ENDC = '\033[0m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'

    """User input"""
    patient = raw_input("Input patient's name: ")
    trial = raw_input("Input trial number: ")
    file_path = "/home/miguel/catkin_ws/src/agora_exo/exo_control/src/nodes/csv_files/{}_labels{}.csv".format(patient,trial)
    class_labels = []

    """Print console output into text file"""
    sys.stdout = open("/home/miguel/catkin_ws/src/agora_exo/exo_control/log/results/threshold-based_{}_{}.txt".format(patient,trial), "w")

    """Import csv file with labels from threshold-based algorithm"""
    os.system("sudo chmod 777 {}".format(file_path))
    with open(file_path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')
        for row in spamreader:
            class_labels.append(int(row[0]))

    """Import patient's data"""
    data_path = "/home/miguel/catkin_ws/src/agora_exo/exo_control/log/mat_files/"
    data = scio.loadmat(data_path + patient + "_proc_data" + trial + ".mat")
    fs_fsr = data["Fs_fsr"][0][0]
    ref_labels = list(data["labels"][0])

    # print type(class_labels[0])
    # print type(ref_labels[0])
    # for i in range(0,len(class_labels)):
    #     print "{}, {}".format(class_labels[i], ref_labels[i])

    true_pos = 0.0
    false_pos = 0.0
    true_neg = 0.0
    false_neg = 0.0
    tol = 0.06       # Tolerance window of 60 ms
    tol_window = int((tol/2.0) * fs_fsr)
    # print "Tolerance window: " + str(tol_window) + " samples"
    # # print type(tol_window)
    # print "Frequency: " + str(fs_fsr) + " Hz"
    n_classes = 4
    verbose = False

    # print bcolors.WARNING + "Patient's name: {}".format(patient) + bcolors.ENDC
    print "Patient's name: {}".format(patient)
    print "Trial: {}".format(trial)

    # sum = 0
    # for i in range(0, len(class_labels)):
    #     """Tolerance window"""
    #     if i >= tol_window and i < len(class_labels)-tol_window:
    #         # curr_tol = time_array[leave_one_out][i+tol_window]-time_array[leave_one_out][i-tol_window]
    #         # print curr_tol
    #         win = []
    #         for j in range(i-tol_window,i+tol_window+1):
    #             win.append(ref_labels[j])
    #         if class_labels[i] in win:
    #             sum += 1.0
    #     else:
    #         if class_labels[i] == ref_labels[i]:
    #             sum += 1.0

    state_names = ['hs', 'ff', 'ho', 'sw']
    time_error = [[] for x in range(n_classes)]
    for phase in range(n_classes):
        for i in range(len(class_labels)):
            """Tolerance window"""
            if i >= tol_window and i < len(class_labels)-tol_window:
                # curr_tol = time_array[leave_one_out][i+tol_window]-time_array[leave_one_out][i-tol_window]
                # print curr_tol
                win = []
                for j in range(i-tol_window, i+tol_window+1):
                    win.append(ref_labels[j])
                if class_labels[i] == phase:
                    if class_labels[i] in win:
                        for k in range(len(win)):
                            if win[k] == phase:
                                time_error[phase].append((k-tol_window)/fs_fsr)
                                break
                        true_pos += 1.0
                        if verbose: print state_names[phase] + ", " + state_names[int(class_labels[i])] + ", " + state_names[ref_labels[i]] + ", true_pos"
                    else:
                        false_pos += 1.0
                        if verbose: print state_names[phase] + ", " + state_names[int(class_labels[i])] + ", " + state_names[ref_labels[i]] + ", false_pos"
                else:
                    # if ref_labels[i] not in win:
                    if phase != ref_labels[i]:
                        true_neg += 1.0
                        if verbose: print state_names[phase] + ", " + state_names[int(class_labels[i])] + ", " + state_names[ref_labels[i]] + ", true_neg"
                    else:
                        false_neg += 1.0
                        if verbose: print state_names[phase] + ", " + state_names[int(class_labels[i])] + ", " + state_names[ref_labels[i]] + ", false_neg"
            else:
                if class_labels[i] == phase:
                    if class_labels[i] == ref_labels[i]:
                        true_pos += 1.0
                    else:
                        false_pos += 1.0
                else:
                    if phase != ref_labels[i]:
                        true_neg += 1.0
                    else:
                        false_neg += 1.0

    print("Timing error")
    for phase in range(n_classes):
        print "(" + state_names[phase] + ")"
        if len(time_error[phase]) > 0:
            print str(np.mean(time_error[phase])) + " + " + str(np.std(time_error[phase]))
        else:
            print "0.06 + 0"

    """Calculate mean time (MT) of stride and each gait phase and Coefficient of Variation (CoV)"""
    print("Mean time (MT) and Coefficient of Variance (CoV)")
    n_group = 0
    for label_group in [class_labels, ref_labels]:
        if n_group == 0:
            print("Results for threshold-based algorithm:")
        else:
            print("Results for FSR:")
        curr_label = -1
        count = 0
        n_phases = 0
        stride_samples = 0
        phases_time = [[] for x in range(n_classes)]
        stride_time = []
        for label in label_group:
            if curr_label != label:
                n_phases += 1
                stride_samples += count
                if label == 0:  # Gait start: HS
                    if n_phases == 4:   # If a whole gait cycle has past
                        stride_time.append(stride_samples/fs_fsr)
                    n_phases = 0
                    stride_samples = 0
                phases_time[label-1].append(count/fs_fsr)
                curr_label = label
                count = 1
            else:
                count += 1.0
        for phase in range(n_classes):
            mean_time = np.mean(phases_time[phase])
            phase_std = np.std(phases_time[phase])
            print "(" + state_names[phase] + ")"
            print "Mean time: " + str(mean_time) + " + " + str(phase_std)
            print("CoV: " + str(phase_std/mean_time*100.0))
        mean_time = np.mean(stride_time)
        phase_std = np.std(stride_time)
        print "(Stride)"
        print "Mean time: " + str(mean_time) + " + " + str(phase_std)
        print("CoV: " + str(phase_std/mean_time*100.0))
        n_group += 1

    """Accuracy"""
    # acc = sum/len(class_labels)
    # print true_pos
    # print false_pos
    # print true_neg
    # print false_neg
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

    print "Accuracy: {}%".format(acc*100.0)
    print "Sensitivity: {}%".format(tpr*100.0)
    print "Specificity: {}%".format(tnr*100.0)
    """Goodness index"""
    G = np.sqrt((1-tpr)**2 + (1-tnr)**2)
    if G <= 0.25:
        print "Optimum classifier (G = {} <= 0.25)".format(G)
    elif G > 0.25 and G <= 0.7:
        print "Good classifier (0.25 < G = {} <= 0.7)".format(G)
    elif G == 0.7:
        print "Random classifier (G = 0.7)"
    else:
        print "Bad classifier (G = {} > 0.7)".format(G)

if __name__ == "__main__":
    main()
    sys.stdout.close()
