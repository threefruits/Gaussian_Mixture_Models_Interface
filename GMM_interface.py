import numpy as np
from model import LHMM
import matplotlib.pyplot as plt
import pickle
import scipy.io as sio
import math
from hmmlearn import hmm
from sklearn.externals import joblib # Save and load HMM
from sklearn.mixture import GaussianMixture
from scipy.stats import multivariate_normal
import time
import pdb
n_com =1
INPUT_DIM = 6
OUTPUT_DIM = 2
GMM_COMPONENTS = n_com
np.set_printoptions(suppress=True)
def norm_pdf_multivariate(x, mu, sigma):
    return multivariate_normal(mean=mu, cov=sigma).pdf(x)



def get_weights_given_input(train_tuple,I, NUM_COMPONENTS, global_weights, global_means, global_cov):
    mu_i_for_getting_weights = train_tuple[1]
    sigma_i_for_getting_weights = train_tuple[2]
    weights_given_i_array = np.zeros(NUM_COMPONENTS)
    pdf_list = []
    pdf_weights_total = 0.0

    pdf_i = multivariate_normal.pdf(I, mu_i_for_getting_weights, sigma_i_for_getting_weights)
    # print(pdf_i)
    pdf_list.append(pdf_i)
    pdf_weights_total += global_weights[0] * pdf_i


    weights_given_i_array[0] = global_weights[0] * pdf_list[0] / pdf_weights_total
    # print (weights_given_i_array)
    return weights_given_i_array


def get_means_given_input(train_tuple,I, NUM_COMPONENTS, weights_given_i_array, means, cov):
    sigma_i_inverse = train_tuple[6]
    mu_i_for_getting_means = train_tuple[4]
    mu_o_for_getting_means = train_tuple[3]
    sigma_oi = train_tuple[5]
    mu_o_given_i_array = np.zeros((NUM_COMPONENTS, OUTPUT_DIM))
    mu_o_given_i = mu_o_for_getting_means + sigma_oi.dot(sigma_i_inverse).dot(I - mu_i_for_getting_means)

    mu_o_given_i_array[0, :] = mu_o_given_i

    return mu_o_given_i_array


def get_cov_given_input(NUM_COMPONENTS,  means, cov):
    sigma_o_given_i_array = np.zeros((NUM_COMPONENTS, OUTPUT_DIM, OUTPUT_DIM))

    sigma_o = cov[0, INPUT_DIM:, INPUT_DIM:]
    sigma_oi = cov[0, INPUT_DIM:, :INPUT_DIM]
    sigma_i_inverse = np.linalg.inv(cov[0, :INPUT_DIM, :INPUT_DIM])
    sigma_io = cov[0, :INPUT_DIM, INPUT_DIM:]
    sigma_o_given_i = sigma_o - sigma_oi.dot(sigma_i_inverse).dot(sigma_io)
    sigma_o_given_i_array[0, :, :] = sigma_o_given_i
    print(sigma_o_given_i)
    return sigma_o_given_i_array


def initialize_GMM():
    # # load data
    try:
        car_data = sio.loadmat('./car_data_new.mat')['car_data']
    except:
        print("data loading error")
  
    training_data= np.delete(car_data, [0 , 5, 10],axis=1 )
    print(training_data.shape)
    for i in range(10720) :
        training_data[i, 2] = math.sqrt(training_data[i, 2] * training_data[i, 2] + training_data[i, 3] * training_data[i, 3])
        training_data[i, 6] = math.sqrt(training_data[i, 6] * training_data[i, 6] + training_data[i, 7] * training_data[i, 7])
        training_data[i,8]=math.sqrt(training_data[i,8]*training_data[i,8]+training_data[i,9]*training_data[i,9])
        training_data[i,9]=math.sqrt(training_data[i,10]*training_data[i,10]+training_data[i,11]*training_data[i,11])
    # utils functions
    # Train the model
    
    training_data= np.delete(training_data, [3, 7, 10, 11],axis=1)
   # for i in range(39740) :
    for i in range(10720):
        self_v_e=(training_data[i,6]+training_data[i+1,6]+training_data[i+2,6]+training_data[i+3,6]+training_data[i+4,6])/5
        ob_v_e=(training_data[i,7]+training_data[i+1,7]+training_data[i+2,7]+training_data[i+3,7]+training_data[i+4,7])/5
        training_data[i,6]=self_v_e
        training_data[i,7]=ob_v_e
    np.savetxt("training_data.txt",training_data)
    print(training_data[0,:])
    lhmm = LHMM()
    n_com =1
    gmm = lhmm.state2action_train(training_data=training_data, num_components=n_com, max_iter=20000)

    GLOBAL_WEIGHTS = gmm.weights_
    GLOBAL_MEANS = gmm.means_
    GLOBAL_COV = gmm.covariances_
    mu_i_for_getting_weights = GLOBAL_MEANS[0, :INPUT_DIM]
    sigma_i_for_getting_weights = GLOBAL_COV[0, :INPUT_DIM, :INPUT_DIM]
    mu_o_for_getting_means = GLOBAL_MEANS[0, INPUT_DIM:]
    mu_i_for_getting_means = GLOBAL_MEANS[0, :INPUT_DIM]
    sigma_oi = GLOBAL_COV[0, INPUT_DIM:, :INPUT_DIM]
    sigma_i_inverse = np.linalg.inv(GLOBAL_COV[0, :INPUT_DIM, :INPUT_DIM])
    sigma_out = get_cov_given_input( GMM_COMPONENTS, GLOBAL_MEANS, GLOBAL_COV)
    # print(GLOBAL_MEANS)
    # np.savetxt("pa.txt",GLOBAL_COV[0])

    return (gmm,mu_i_for_getting_weights,sigma_i_for_getting_weights,mu_o_for_getting_means,mu_i_for_getting_means,sigma_oi,sigma_i_inverse,sigma_out)

def test_gmm(train_tuple,current_state_vector,future_vel_vector ):
    gmm = train_tuple[0]
    sigma_out = train_tuple[7]
    GLOBAL_WEIGHTS = gmm.weights_
    GLOBAL_MEANS = gmm.means_
    GLOBAL_COV = gmm.covariances_
    # np.savetxt("pa.txt",GLOBAL_COV)


    # change according to features

    I = current_state_vector
    weights_given_i = get_weights_given_input(train_tuple, I, GMM_COMPONENTS, GLOBAL_WEIGHTS, GLOBAL_MEANS, GLOBAL_COV)
    means_out = get_means_given_input(train_tuple, I, GMM_COMPONENTS, weights_given_i, GLOBAL_MEANS, GLOBAL_COV)

    cond_gmm = GaussianMixture(n_components=n_com,
                               weights_init=weights_given_i,
                               means_init=means_out,
                               precisions_init=None)
    cond_gmm.fit(np.random.randn(10,OUTPUT_DIM))
    cond_gmm.weights_ = (weights_given_i)
    cond_gmm.means_ = (means_out)

    cond_gmm.covariances_ = (sigma_out)
    print(means_out)
    result = norm_pdf_multivariate(future_vel_vector, cond_gmm.means_[0], cond_gmm.covariances_[0])
    print("the result is", result)
    return result 



if __name__ == "__main__":
    train_tuple = initialize_GMM()
    current_state_vector=[-7.731, -13.70, 0.797, 6.883, -2.898, 4.997]
    #current_state_vector=[-40.543,-16,4.64, 28.6,23.2,2.549]
   # current_state_vector = [40.288, -11.174, 0.229, 21.22, -13.38, 4.088]
    future_vel_vector=[ 0.87, 5.096]
    start = time.time()
    times = 10
    for i in range(times):
        result = test_gmm(train_tuple, current_state_vector, future_vel_vector)
    end = time.time()
    print("time:",(end-start)/times)
    print (result)
