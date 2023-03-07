# This is a script to test out implementation of the NCAN SSVEP feature extractor swuite
# Written by Eli Kinney-Lang (EKL) on 06-March-2023


import os
import sys

# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

sys.path.append(os.path.join('plugins\FeatureExtractorSSVEP',os.pardir))

sys.path.append('/plugins/FeatureExtractorSSVEP')

# # from src.bci_data import *
from bci_essentials.bci_data import *

# import NCAN FeatureExtractor folder - hidden in Git at the moment.
from FeatureExtractorSSVEP import *

# import plotting
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import pyxdf

xdf = pyxdf.load_xdf("examples/data/p001_ssvep_boccia.xdf")

##Dealing with Feature Extractor SSVEP

#This is hardcoded to the above participant for the moment. Would need to get data by type instead
data = xdf[0][2]['time_series']
#get rid of the last dimension, as that is the trigger channel, and makes our data rank deficient
data = data[:,0:7]

#they want the data to be in info x channels x samples - so we need to transform our data
data = data.T
#Make data 3D from our time series
data = np.expand_dims(data,axis=0)

# Feature extraction setup (mandatory options)
harmonics_count = 3
targets_frequencies = np.array([7.692307,10,11.1111])
sampling_rate = int(xdf[0][2]['info']['nominal_srate'][0])

#Add a filter bank if we want
subbands = np.array([])

cca = FeatureExtractorCCA()

# cca.setup_feature_extractor(
#     harmonics_count=harmonics_count,
#     targets_frequencies=targets_frequencies,
#     sampling_frequency=sampling_rate,
#     voters_count=1,
#     use_gpu=False,
#     max_batch_size=16,
#     samples_count=data.shape[2],
#     max_correlation_only=False,
#     filter_order=12,
#     subbands=None,
#     embedding_dimension=0,
#     delay_step=0,
#     )

cca.setup_feature_extractor(
    harmonics_count = harmonics_count,
    targets_frequencies = targets_frequencies,
    sampling_frequency = sampling_rate,
    voters_count = 1,
    filter_order=5)

cca_features = cca.extract_features(data)




# # Initialize the SSVEP
# # should try to automate the reading of some of this stuff from the file header
# test_ssvep = EEG_data()

# # Define the classifier
# # test_ssvep.classifier = ssvep_riemannian_mdm_classifier(subset=[])
# test_ssvep.classifier = ssvep_ts_classifier(subset=[])

# # fn = "Z:\BCI Program\BCI Studies\Boccia_Software_Pilot-March2023\Data\sub-P003\ses-S003\eeg\sub-P003_ses-S003_task-T1_run-001_eeg.xdf"
# fn = "examples/data/p001_ssvep_boccia.xdf"

# # Load from xdf into erp_data format
# # test_ssvep.load_offline_eeg_data(filename = "examples/data/adam_ssvep_boccia.xdf", format='xdf')
# test_ssvep.load_offline_eeg_data(filename = fn,format='xdf')

# test_ssvep.classifier.set_ssvep_settings(n_splits=3, random_seed=42, n_harmonics=2, f_width=1.0,covariance_estimator='oas',sgd_loss='squared_hinge',l1_ratio=0.15,penalty='elasticnet')

# # initial_subset=['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Cp4', 'C4', 'F4', 'Cp3', 'C3', 'F3', 'Cz', 'Fz']
# # test_ssvep.classifier.setup_channel_selection(method = "SBS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
# #                                 max_time= 999, min_channels=2, max_channels=16, performance_delta=0,      # stopping criterion
# #                                 n_jobs=-1, print_output="verbose") 

# test_ssvep.main(online=False, training=True, max_samples=5120, pp_type="bandpass", pp_low=3, pp_high=50)

# print("debug")