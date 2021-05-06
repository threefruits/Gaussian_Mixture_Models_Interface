import numpy as np 
from hmmlearn import hmm
from sklearn.externals import joblib # Save and load HMM
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import scipy.io as sio
import random
import scipy
from model_utils import *

def Pure_GMM(training_data, num_components=1, max_iter=1000, verbose=True):

	"""
	This function serves to train a GMM.

	Input:
		-- training_data: a numpy array containing training features.
		-- num_components: number of mixtures
		-- max_iter: maximum training iterations

	Output:
		-- gmm.means_, gmm.covariances_, gmm.weights_, gmm
	"""

	gmm = GaussianMixture(n_components = num_components, covariance_type='full', verbose=verbose)
	#print('Begin training GMM.')
	gmm.fit(training_data)
	#print('GMM training is finishied.')
	print('GMM BIC score is ' +str(gmm.bic(training_data)))


	return gmm


def HMM_Gaussian(training_data, seq_length_array, num_hidden_states=3,  max_iter=1000, verbose=False):

	"""
	This function serves to train a HMM with Gaussian emission distributions.

	Input:
		-- training_data: A numpy array containing training features. All the sequences are stacked, each row
						  is a feature vector.
		-- seq_length_array : A numpy array containing the training sequence length for each sequence.
		-- num_hidden_states
		-- max_iter

	Output:
		-- model.means_, model.covars_
	"""

	#EM algorithm to fit HMM-Gaussian model
	model = hmm.GaussianHMM(n_components = num_hidden_states, covariance_type = 'full', n_iter=max_iter, verbose=verbose)
	print('Begin training HMM-Gaussian model.')
	model.fit(training_data, seq_length_array)
	print('HMM-Gaussian model training is finishied.')

	return model.means_, model.covars_, model


##############################################

class LHMM(object):

	"""
	This class implements the two-layer HMM.
	"""

	def __init__(self):
		pass

	def HMM_Gaussian(self, training_data, seq_length_array, num_hidden_states=3,  max_iter=1000, verbose=True):

		"""
		This function serves to train a HMM with Gaussian emission distributions.
	
		Input:
			-- training_data: A numpy array containing training features. All the sequences are stacked, each row
							  is a feature vector.
			-- seq_length_array : A numpy array containing the training sequence length for each sequence.
			-- num_hidden_states
			-- max_iter
	
		Output:
			-- model.means_, model.covars_
		"""

		#EM algorithm to fit HMM-Gaussian model
		model = hmm.GaussianHMM(n_components = num_hidden_states, covariance_type = 'full', n_iter=max_iter, verbose=verbose)
		print('Begin training HMM-Gaussian model.')
		model.fit(training_data, seq_length_array)
		print('HMM-Gaussian model training is finishied.')
	
		return model
	
	def first_layer_train(self, num_HMM, num_hidden_states_list, train_data_list, seq_length_array_list, dist_flag=0):
		"""
		This function serves to train the first layer HMMs.

		Input:
		-- num_HMM: the number of first layer HMMs.
		-- num_hidden_states_list: a list containing hidden state numbers. 
		   The number of elements should equal to num_HMM.
		-- train_data: a list containing several training sequences, each is an array;
		-- seq_length_array_list: a list containing seq_length_array corresponding to each train_data;
		-- dist_flag: use which emission distribution: 0: Gaussian; 1: GMM;
					  Default is Gaussian.

		Output:
		-- a list of well-trained models of first layer HMM (parameters)

		"""

		if dist_flag == 0: # emission is Gaussian
			hmm10 = self.HMM_Gaussian(train_data_list[0], seq_length_array_list[0], num_hidden_states=num_hidden_states_list[0])
			print('hmm10 training is done!')
			hmm11 = self.HMM_Gaussian(train_data_list[1], seq_length_array_list[1], num_hidden_states=num_hidden_states_list[1])
			print('hmm11 training is done!')
			hmm12 = self.HMM_Gaussian(train_data_list[2], seq_length_array_list[2], num_hidden_states=num_hidden_states_list[2])
			print('hmm12 training is done!')
			hmm13 = self.HMM_Gaussian(train_data_list[3], seq_length_array_list[3], num_hidden_states=num_hidden_states_list[3])
			print('hmm13 training is done!')
			hmm14 = self.HMM_Gaussian(train_data_list[4], seq_length_array_list[4], num_hidden_states=num_hidden_states_list[4])
			print('hmm14 training is done!')
			hmm15 = self.HMM_Gaussian(train_data_list[5], seq_length_array_list[5], num_hidden_states=num_hidden_states_list[5])
			print('hmm15 training is done!')
			hmm16 = self.HMM_Gaussian(train_data_list[6], seq_length_array_list[6], num_hidden_states=num_hidden_states_list[6])
			print('hmm16 training is done!')

		else:
			pass

		return [hmm10, hmm11, hmm12, hmm13, hmm14, hmm15, hmm16]


	def first_layer_inference_train(self, hmm1_list, obs_seq_array, obs_seq_length_array, time_length=10):
		"""
		This function serves to generate observation sequences for the sceond layer HMM in training process;

		2. Obtain inference results of the first layer and propagate to the second layer in test process.

		Input:
		-- hmm1_list: a list containing HMM models in the first layer;
		-- obs_seq_array: an array containing the training obs sequences (entire trajectories);
		-- obs_seq_length_array: an array containing seq lengths;
		-- time_length: the length of each obs subsequence;

		Output:
		-- likelihood_seq_array
		-- likelihood_seq_length_array

		Note: When training, call this function for each situations (because we need labelled training seqs for the second layer)

		"""

		[hmm10, hmm11, hmm12, hmm13, hmm14, hmm15, hmm16] = hmm1_list

		likelihood_seq_list = [] # initialize likelihood sequence list
		likelihood_seq_length_list = []

		for i in range(len(obs_seq_length_array)):
			current_seq_length = obs_seq_length_array[i]
			current_seq = obs_seq_array[sum(obs_seq_length_array[:i]):sum(obs_seq_length_array[:i])+current_seq_length, :]

			current_likilihood_matrix = np.zeros((current_seq.shape[0]-time_length+1, len(hmm1_list))) # initialize current likelihood matrix
			for t in range(current_seq.shape[0] - time_length + 1):
				sub_obs_seq = current_seq[t : t+time_length, :]
				L10 = hmm10.score(sub_obs_seq) # log-likelihood
				L11 = hmm11.score(sub_obs_seq)
				L12 = hmm12.score(sub_obs_seq)
				L13 = hmm13.score(sub_obs_seq)
				L14 = hmm14.score(sub_obs_seq)
				L15 = hmm15.score(sub_obs_seq)
				L16 = hmm16.score(sub_obs_seq)

				current_likilihood_matrix[t, :] = np.array([L10,L11,L12,L13,L14,L15,L16])

			likelihood_seq_list.append(current_likilihood_matrix) # register the likelihood seqence (each row is a likelihood vector at a time step)
			likelihood_seq_length_list.append(current_seq.shape[0]-time_length+1)

		# Change list into array
		likelihood_seq_length_array = np.array(likelihood_seq_length_list)
		likelihood_seq_array = likelihood_seq_list[0]
		for j in range(1, len(likelihood_seq_list)):
			likelihood_seq_array = np.concatenate((likelihood_seq_array, likelihood_seq_list[j]), axis=0)



		return likelihood_seq_array, likelihood_seq_length_array



	def first_layer_inference_test(self, hmm1_list, obs_seq_array, time_length=10):
		"""
		This function serves to make inference for testing in the first layer.

		Input:
		-- hmm1_list: a list containing all first-layer HMM models;
		-- obs_seq_array: an array containing test seqences;

		Output:
		-- current_likilihood_matrix: give the inference result of the first layer.
		"""

		[hmm10, hmm11, hmm12, hmm13, hmm14, hmm15, hmm16] = hmm1_list

		current_likilihood_matrix = np.zeros((obs_seq_array.shape[0]-time_length+1, len(hmm1_list))) # initialize current likelihood matrix
		for t in range(obs_seq_array.shape[0]-time_length+1):
			sub_obs_seq = obs_seq_array[t : t+time_length, :]
			L10 = hmm10.score(sub_obs_seq) # log-likelihood
			L11 = hmm11.score(sub_obs_seq)
			L12 = hmm12.score(sub_obs_seq)
			L13 = hmm13.score(sub_obs_seq)
			L14 = hmm14.score(sub_obs_seq)
			L15 = hmm15.score(sub_obs_seq)
			L16 = hmm16.score(sub_obs_seq)

			current_likilihood_matrix[t, :] = np.array([L10,L11,L12,L13,L14,L15,L16])


		return current_likilihood_matrix



	def second_layer_train(self, num_HMM, num_hidden_states_list, train_data_list, seq_length_array_list, dist_flag=0):
		"""
		This function serves to train the second layer HMMs.

		Input:
		-- num_HMM: the number of second layer HMMs.
		-- num_hidden_states_list: a list containing hidden state numbers. 
		   The number of elements should equal to num_HMM.
		-- train_data: a list containing several training sequences, each is an array;
		-- seq_length_array_list: a list containing seq_length_array corresponding to each train_data;
		-- dist_flag: use which emission distribution: 0: Gaussian; 1: GMM;
					  Default is Gaussian.

		Output:
		-- a list of well-trained models of second layer HMM (parameters)

		"""
			
		if dist_flag == 0: # emission is Gaussian
			hmm20 = self.HMM_Gaussian(train_data_list[0], seq_length_array_list[0], num_hidden_states=num_hidden_states_list[0])
			print('hmm20 training is done!')
			hmm21 = self.HMM_Gaussian(train_data_list[1], seq_length_array_list[1], num_hidden_states=num_hidden_states_list[1])
			print('hmm21 training is done!')

		else:
			pass

		return [hmm20, hmm21]



	def second_layer_inference_test(self, hmm2_list, obs_seq_array, time_length=10):
		"""
		This function serves to make inference for testing in the second layer.

		Input:
		-- hmm2_list: a list containing all second-layer HMM models;
		-- obs_seq_array: an array containing test seqences;

		Output:
		-- current_likilihood_matrix: give the inference result of the second layer.
		"""

		[hmm20, hmm21] = hmm2_list

		L20 = hmm20.score(obs_seq_array)
		L21 = hmm21.score(obs_seq_array)

		# current_likilihood_matrix = np.zeros((obs_seq_array.shape[0]-time_length+1, len(hmm2_list))) # initialize current likelihood matrix
		# for t in range(obs_seq_array.shape[0]-time_length+1):
		# 	sub_obs_seq = obs_seq_array[t : t+time_length, :]
		# 	L20 = hmm20.score(sub_obs_seq) # log-likelihood
		# 	L21 = hmm21.score(sub_obs_seq)

		# 	current_likilihood_matrix[t, :] = np.array([L20,L21])


		#return current_likilihood_matrix
		return L20, L21


	def state2action_train(self, training_data, num_components, max_iter=1000):
		"""
		This function serves to use a GMM to approximate a mapping from state feature space to action space.

		Input:
		-- training data
		-- num_components

		Output:
		-- gmm: the fitted model
		"""

		gmm = Pure_GMM(training_data=training_data, num_components=num_components, max_iter=max_iter, verbose=False)

		return gmm


	def state2action_inference(self, gmm_model, test_input):
		"""
		This function serves to use the well-trained GMM to propagate the states from current time step.

		Input:
		-- gmm_model
		-- test_data: as the 'input' in the feature vector ([I | O])

		Output:
		-- The corresponding output action to the test_input: sample_output
		"""

		gmm = gmm_model
		GMM_COMPONENTS = len(gmm.weights_)
		WEIGHTS = gmm.weights_
		MEANS = gmm.means_
		COVS = gmm.covariances_

		INPUT_DIM = len(test_input)
		OUTPUT_DIM = MEANS.shape[1] - INPUT_DIM

		I = test_input
		#print(I.shape)
		#print(INPUT_DIM)

		weights_given_i = GMM_get_weights_given_input(I, GMM_COMPONENTS, WEIGHTS, MEANS, COVS, INPUT_DIM, OUTPUT_DIM)
		means_out = GMM_get_means_given_input(I, GMM_COMPONENTS, weights_given_i, MEANS, COVS, INPUT_DIM, OUTPUT_DIM)
		sigma_out = GMM_get_cov_given_input(I, GMM_COMPONENTS, weights_given_i, MEANS, COVS, INPUT_DIM, OUTPUT_DIM)

		gmm.weights_ = weights_given_i
		gmm.means_ = means_out
		gmm.covariances_ = sigma_out
		print(gmm.weights_.shape)
		print(gmm.means_.shape)
		print(gmm.covariances_.shape)

		sample_output = gmm.sample()#[0][0].reshape(1, OUTPUT_DIM)
		print(sample_output.shape)

		return sample_output


###################################

class HMM(object):
	"""
	This class implements the canonical HMM classifier.
	"""

	def __init__(self):
		pass

	def HMM_Gaussian(self, training_data, seq_length_array, num_hidden_states=3,  max_iter=1000, verbose=True):

		"""
		This function serves to train a HMM with Gaussian emission distributions.
	
		Input:
			-- training_data: A numpy array containing training features. All the sequences are stacked, each row
							  is a feature vector.
			-- seq_length_array : A numpy array containing the training sequence length for each sequence.
			-- num_hidden_states
			-- max_iter
	
		Output:
			-- model.means_, model.covars_
		"""

		#EM algorithm to fit HMM-Gaussian model
		model = hmm.GaussianHMM(n_components = num_hidden_states, covariance_type = 'full', n_iter=max_iter, verbose=verbose)
		print('Begin training HMM-Gaussian model.')
		model.fit(training_data, seq_length_array)
		print('HMM-Gaussian model training is finishied.')
	
		return model

	def hmm_train(self, num_hidden_states_list, train_data_list, seq_length_array_list, dist_flag=0):
		"""
		Train individual HMMs.
		"""

		if dist_flag == 0: # emission is Gaussian
			hmm0 = self.HMM_Gaussian(training_data=train_data_list[0], seq_length_array=seq_length_array_list[0], \
									num_hidden_states=num_hidden_states_list[0])
			print('hmm0 training is done!')
			hmm1 = self.HMM_Gaussian(training_data=train_data_list[1], seq_length_array=seq_length_array_list[1], \
									num_hidden_states=num_hidden_states_list[1])
			print('hmm1 training is done!')

		else:
			pass

		return [hmm0, hmm1]

	def hmm_inference(self, hmm_list, obs_seq_array):

		[hmm0, hmm1] = hmm_list
		L0 = hmm0.score(obs_seq_array)
		L1 = hmm1.score(obs_seq_array)

		return L0, L1





	











