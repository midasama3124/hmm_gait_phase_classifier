#!/usr/bin/python
# import rospy
import numpy as np
import matplotlib.pyplot as plt
import hmms

class CHMM_gait_phase_detection(object):
    def __init__(self, A = None, B = None, Pi = None):
        if A is not None and B is not None and Pi is not None:
            self.A = A      # A is the matrix of transition probabilities from state i [row] to state j [column].
            self.B = B      # B is the matrix of probabilities that the state [row] will emmit output variable [column].
            self.Pi = Pi    # Pi is the vector of initial state probabilities.
            self.model_init()
        else:
            self.rnd_model_init()

    def model_init(self):
        """Initialization of a continuous Hidden Markov Model given certain A, B
        and Pi.
        No input values
        No output values"""
        # Create CtHMM by given parameters.
        self.chmm = hmms.CtHMM(self.A,self.B,self.Pi)
        print("Model has been initialized.")

    def rnd_model_init(self, num_states = 4, num_obs = 2):
        """Initialization of Hidden Markov Model using random parameters.
        No input values
        No output values"""
        self.chmm = hmms.CtHMM.random(num_states,num_obs,method="unif")
        print("Model has been randomly initialized.")

    def save_param(self, filename):
        """This function saves model parameters into a .npz file.
        **Input values:
        -filename (string)
        No output values"""
        # Save parameters in file
        self.chmm.save_params( str(filename) )

    def read_param(self, filename):
        """This function reads model parameters from a .npz file.
        **Input values:
        -filename (string)
        No output values"""
        # Read parameters from file
        self.chmm_from_file = hmms.CtHMM.from_file( str(filename) )

    def set_params_from_file(self):
        """This function sets model parameters from a .npz file.
        **Input values:
        -filename (string)
        No output values"""
        self.chmm.set_params_from_file( filename )

    def generate_rnd_seq(self, seq_len = 10):
        """This function generates a random state and emission sequences based
        on initialized model.
        **Input values:
        -sequence length (int)
        No output values"""
        # t_seq: time sequence, s_seq: state sequence, e_seq: emission sequence
        self.t_seq, self.s_seq, self.e_seq = self.chmm.generate(seq_len,0.5)
        # Function with own time sequence
        # self.t_seq, self.s_seq, self.e_seq = self.chmm.generate(seq_len, time=[0,3,5,7,8,11,14])
        #resize plot
        plt.rcParams['figure.figsize'] = [20,20]
        hmms.plot_hmm(self.s_seq, self.e_seq, time=self.t_seq)

    def viterbi_alg(self, verbose = False):
        """This function uses the Viterbi algorithm to find the most probable
        state sequence, as a solution for the Problem 2 of this type of modeling.
        No input values
        No output values"""
        # log_prob: logarithm of the probability of the sequence
        # s_seq: state sequence
        self.generate_rnd_seq()
        log_prob, self.s_seq = self.chmm.viterbi( self.e_seq )
        if verbose: print("Most probability state sequence with p = {}".format( np.exp(log_prob) ))
        # Let's print the most likely state sequence, it can be same or differ from the sequence above.
        hmms.plot_hmm( self.s_seq, self.e_seq )

    def get_state_confidence(self):
        """This function gets the probability that the given emission was generated
        by some concrete state.
        No input values
        No output values"""
        self.generate_rnd_seq()
        self.log_prob_table = self.chmm.states_confidence( self.e_seq )
        print(np.exp( self.log_prob_table ))

    def generate_dataset(self, seq_num = 3, seq_len = 10, verbose = False):
        """This functions generates many emission sequences in once, which are in the
        form that is suitable for training of parameters.
        **Input values:
        -seq_num (int): Number of data sequences
        -seq_len (int): Length of each sequence
        -verbose (bool): If true, function prints generated dataset
        No output values"""
        self.art_s_seqs, self.art_e_seqs = self.chmm.generate_data((seq_num,seq_len))
        if verbose:
            print("State sequences:\n" + str( self.art_s_seqs ))
            print("Emission sequences:\n" + str( self.art_e_seqs ))

    def print_likelihood_est(self, e_seqs):
        """This functions prints the likelihood estimation of the model from this
        class, given certain emission sequences.
        **Input values:
        -e_seqs (list/tuple): Emission sequences to evaluate the likelihood estimation
        No output values"""
        print(np.exp( self.chmm.data_estimate(e_seqs) ))

    def print_params(self):
        """This functions prints A, B and Pi of the model from this class.
        No input values
        No output values"""
        hmms.print_parameters(self.chmm)

    def baum_welch_alg(self, iter = 10):
        """This function trains the model parameters by EM algorithm from several
        emission sequences.
        **Input values:
        -iter (int): Number of iterations in training process
        No output values"""
        print("Generating artificial emission sequences from given model...")
        self.generate_dataset(5,50)
        print("Likelihood estimation before training: "),
        self.print_likelihood_est(self.art_e_seqs)
        print("Training model...")
        self.chmm.baum_welch( self.art_e_seqs, iter )
        print("Likelihood estimation after training: "),
        self.print_likelihood_est(self.art_e_seqs)
        self.print_params()

    def mle_alg(self):
        """This function estimates the model parameters by the maximum likelihood
        estimation (MLE) method, as we have a dataset of full observations (i.e.
        both emission and hidden states sequences).
        No input values
        No output values"""
        # Generate artificial dataset of both hidden states and emissions sequences
        self.generate_dataset(5,50)
        log_est = self.chmm.full_data_estimate(self.art_s_seqs,self.art_e_seqs)
        self.chmm.maximum_likelihood_estimation(self.art_s_seqs,self.art_e_seqs)
        log_est_mle = self.chmm.full_data_estimate(self.art_s_seqs,self.art_e_seqs)

        print("The probability of the dataset being generated by the original model is: " + str( np.exp(log_est) ) + ".")
        print("The probability of the dataset being generated by the original model is: " + str( np.exp(log_est_mle) ) + ".")

def main():
    A = np.array([[0.9,0.1],[0.4,0.6]])
    B = np.array([[0.9,0.08,0.02],[0.2,0.5,0.3]])
    Pi = np.array([0.8,0.2])
    # hmm = HMM_gait_phase_detection(A,B,Pi)
    chmm = CHMM_gait_phase_detection()
    chmm.generate_rnd_seq()
    # chmm.viterbi_alg(verbose = True)
    # chmm.get_state_confidence()
    # chmm.generate_dataset(5,50,True)
    # chmm.baum_welch_alg(100)
    # chmm.mle_alg()

if __name__ == "__main__":
    main()
