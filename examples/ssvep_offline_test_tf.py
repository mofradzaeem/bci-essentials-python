# This is a script to test the functionality of python SSVEP processing
# Written by Brian Irvine on 08/05/2021

import os
import sys


# Add parent directory to path to access bci_essentials
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))


# # from src.bci_data import *
from bci_essentials.bci_data import *
from bci_essentials.visuals import *

# import
import matplotlib.pyplot as plt

# Initialize the SSVEP
# should try to automate the reading of some of this stuff from the file header
test_ssvep = EEG_data()

# Define the classifier
#test_ssvep.classifier = ssvep_riemannian_mdm_classifier(subset=[])
test_ssvep.classifier = ssvep_cca_classifier(subset=[])

# fn = "Z:\BCI Program\BCI Studies\Boccia_Software_Pilot-March2023\Data\sub-P003\ses-S003\eeg\sub-P003_ses-S003_task-T1_run-001_eeg.xdf"
fn = "examples/data/p001_ssvep_boccia.xdf"

# Load from xdf into erp_data format
# test_ssvep.load_offline_eeg_data(filename = "examples/data/adam_ssvep_boccia.xdf", format='xdf')
test_ssvep.load_offline_eeg_data(filename = fn,format='xdf')

# Select appropriate target frequencies depending on the trial
target_freqs = [7.692307, 10, 11.11111]
# target_freqs = [8.33, 10, 12.5]
test_ssvep.classifier.set_ssvep_settings(target_freqs = target_freqs)



#initial_subset=['PO7', 'PO3', 'POz', 'PO4', 'PO8', 'O1', 'Oz', 'O2', 'Cp4', 'C4', 'F4', 'Cp3', 'C3', 'F3', 'Cz', 'Fz']
#test_ssvep.classifier.setup_channel_selection(method = "SBS", metric="accuracy", initial_channels = initial_subset,    # wrapper setup
#                                max_time= 999, min_channels=2, max_channels=16, performance_delta=0,      # stopping criterion
#                                n_jobs=-1, print_output="verbose") 

test_ssvep.main(online=False, training=False,train_complete=True,print_performance=True,print_predict=True, max_samples=5120)

# TODO compute prediction accuracy by comparing test_ssvep_classifier.predictions to test_ssvep.labels
#      only take the labels every 5th value.

print("debug")

# # Some optional plotting
# # # plot a spectrogram of the session
# for ci, ch in enumerate(test_ssvep.channel_labels):
#     eeg = np.array(test_ssvep.classifier.X[0,ci,:])
#     tv = [e/test_ssvep.fsample for e in list(range(0,len(eeg)))]

#     plt.plot(tv, eeg)

# plt.show()
# plt.clf()

# for i in range(48):
#     eeg = np.array(test_ssvep.classifier.X[i,15,:])
#     tv = [e/test_ssvep.fsample for e in list(range(0,len(eeg)))]

#     f, t, Sxx = scipy.signal.spectrogram(eeg, fs=test_ssvep.fsample, nperseg=512)

#     fp, Pxx = scipy.signal.welch(eeg, fs=test_ssvep.fsample, nperseg=512, return_onesided = True)

#     f_target = test_ssvep.classifier.target_freqs[int(test_ssvep.classifier.y[i])]

#     # Plot the EEG for inspection
#     plt.subplot(311)
#     plt.plot(tv,eeg)
#     plt.title(f_target)

#     # psd
#     plt.subplot(312)
#     plt.plot(fp, Pxx)
#     plt.vlines(x=[f_target,f_target*2,f_target*3], ymin=-1000, ymax=1000)
#     plt.xlim([-0.5,50])
#     plt.ylim([-0.5,20])
    
#     # spectrogram
#     plt.subplot(313)
#     plt.pcolormesh(t, f, Sxx, shading='gouraud')
#     plt.ylabel('Frequency [Hz]')
#     plt.xlabel('Time [sec]')
#     plt.ylim([0,30])
#     plt.xlim([0,4])
#     plt.hlines(y=[f_target,f_target*2,f_target*3], xmin=-5, xmax=5, color='r')
#     plt.show()

#     print("debug")

#     # clear
#     plt.clf()

