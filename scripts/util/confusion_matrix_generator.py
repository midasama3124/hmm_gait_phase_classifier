#!/usr/bin/env python
# coding=utf-8
import rospy
import rospkg
import csv

def main():
    """ROS initialization"""
    rospy.init_node('confusion_matrix')
    rospack = rospkg.RosPack()
    packpath = rospack.get_path('exo_control')
    datapath = packpath + "/log"

    """Variable declaration"""
    subs = {"healthy": ["daniel", "erika", "felipe", "jonathan", "luis", "nathalia", "paula", "pedro", "tatiana"],
            "patient": ["andres", "carlos", "carmen", "carolina", "catalina", "claudia", "emmanuel", "fabian", "gustavo"]}
    labels = {"ref": [],
              "tb": [],
              "intra": [],
              "inter": []}
    # Sensitivity and Specificity values
    n_classes = 4        # Number of gait phases
    n_trials = 3         # Number of performed trials
    sum = 0.0
    true_pos = {"tb": 0.0, "intra": 0.0, "inter": 0.0}
    false_pos = {"tb": 0.0, "intra": 0.0, "inter": 0.0}
    true_neg = {"tb": 0.0, "intra": 0.0, "inter": 0.0}
    false_neg = {"tb": 0.0, "intra": 0.0, "inter": 0.0}
    fs = 200      # Hz
    tol = 6e-2       # Tolerance window of 60 ms
    tol_window = int((tol/2) / (1/float(fs)))
    # tol_window = 5
    print tol_window
    conf_matrix = {"tb": [[0 for x in range(n_classes)] for y in range(n_classes)],
                  "intra": [[0 for x in range(n_classes)] for y in range(n_classes)],
                  "inter": [[0 for x in range(n_classes)] for y in range(n_classes)]}

    for group in subs:
        print "Confusion matrix of " + group + " subjects:"
        for approach in conf_matrix:
            conf_matrix[approach] = [[0 for x in range(n_classes)] for y in range(n_classes)]    # Reinitialization of confusion matrix
        for approach in true_pos:
            true_pos[approach] = 0.0            # Reinitialization of indeces
            false_pos[approach] = 0.0
            true_neg[approach] = 0.0
            false_neg[approach] = 0.0
        for subject in subs[group]:
            print subject
            for trial in range(1,n_trials+1):
                for name in labels:
                    if (name == "inter" and trial == 3)  or name != "inter":
                        labels[name] = []      # Reinitialization of label lists
                        """Data loading"""
                        filename = datapath + "/" + name + "_labels/" + subject + "_labels" + str(trial) + ".csv"
                        # print(filename)
                        with open(filename) as csv_file:
                            csv_reader = csv.reader(csv_file, delimiter=',')
                            for row in csv_reader:
                                exec("labels['" + name + "'].append(int(row[0]))")
                # for k in labels:
                #     print(k + ": " + str(len(labels[k])))
                """Confusion matrix with tolerance window"""
                n_labels = len(labels["intra"])
                for phase in range(n_classes):
                    for i in range(n_labels):
                        """Tolerance window"""
                        if i >= tol_window and i < n_labels-tol_window:
                            win = []
                            for win_label in labels["ref"][i-tol_window:i+tol_window+1]:
                                win.append(win_label)
                            for approach in labels:
                                if (approach != "ref" and approach != "inter") or (approach == "inter" and trial == 3):
                                    if labels[approach][i] == phase:
                                        if labels[approach][i] in win:     # True Positive
                                            true_pos[approach] += 1.0
                                            conf_matrix[approach][phase][phase] += 1
                                        else:                          # False positive
                                            false_pos[approach] += 1.0
                                            conf_matrix[approach][labels["ref"][i]][phase] += 1
                                    else:
                                        if phase != labels["ref"][i]:     # True negative
                                            true_neg[approach] += 1.0
                                            conf_matrix[approach][labels["ref"][i]][labels["ref"][i]] += 1
                                        else:                          # False negative
                                            false_neg[approach] += 1.0
                                            conf_matrix[approach][labels["ref"][i]][labels[approach][i]] += 1
                        else:
                            for approach in labels:
                                if (approach != "ref" and approach != "inter") or (approach == "inter" and trial == 3):
                                    if labels[approach][i] == phase:
                                        if labels[approach][i] == labels["ref"][i]:    # True Positive
                                            true_pos[approach] += 1.0
                                            conf_matrix[approach][phase][phase] += 1
                                        else:                                   # False positive
                                            false_pos[approach] += 1.0
                                            conf_matrix[approach][labels["ref"][i]][phase] += 1
                                    else:
                                        if phase != labels["ref"][i]:  # True negative
                                            true_neg[approach] += 1.0
                                            conf_matrix[approach][labels["ref"][i]][labels["ref"][i]] += 1
                                        else:                                   # False negative
                                            false_neg[approach] += 1.0
                                            conf_matrix[approach][labels["ref"][i]][labels[approach][i]] += 1
        """Print confusion matrix of each study group"""
        for approach in conf_matrix:
            print(approach + " Conf. Matrix:")
            n_checks = true_neg[approach] + true_pos[approach] + false_neg[approach] + false_pos[approach]
            # print(n_checks)
            print(conf_matrix[approach])
            # sum_conf = 0
            # n_posit = 0
            # for i in range(len(conf_matrix[approach])):
            #     for j in range(len(conf_matrix[approach][i])):
            #         sum_conf += conf_matrix[approach][i][j]
            #         if i == j:
            #             n_posit += conf_matrix[approach][i][j]
            # # print(sum_conf)
            # rospy.logwarn("Accuracy: {}%".format(n_posit/float(sum_conf)*100.0))
            """Accuracy"""
            if (true_neg[approach]+true_pos[approach]+false_neg[approach]+false_pos[approach]) != 0.0:
                acc = (true_neg[approach] + true_pos[approach])/n_checks
            else:
                acc = 0.0
            rospy.logwarn("Accuracy (" + approach + "): {}%".format(acc*100.0))

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt as e:
        print("Program finished\n")
        sys.stdout.close()
        os.system('clear')
        raise
