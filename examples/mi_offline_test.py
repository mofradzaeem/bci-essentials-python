"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

import os
import sys

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

# from src.bci_data import *
from bci_essentials.bci_data import *
from bci_essentials.visuals import *
from itertools import combinations
import pickle
import numpy as np

data, index = EEG_data().load_offline_eeg_data(filename=r"C:\Users\elyss\PycharmProjects\bci-essentials-python\examples\data\mi_example.xdf", find_data=True)
default_rate = float(data[index]['info']['nominal_srate'][0])
print("The default rate is", default_rate)

min_num_samples = int(input("Please choose the minimum number of samples. Do not choose less than 78 samples: "))  #DO NOT choose less than 78 samples
max_num_samples = int(input("Please choose the maximum number of samples. Do not choose more than 256 samples: ")) #DO NOT choose more than 1189 samples
print("NOTE: The shape will be twice as large for the number of samples.\n")
min_samples_divider = default_rate/min_num_samples #upper bound for range of number of samples
max_samples_divider = default_rate/max_num_samples #lower bound for range of number of samples

num = max_samples_divider

while num >= max_samples_divider and num <= min_samples_divider:

    num = round(num, 2)

# Initialize data object
    test_mi = EEG_data()

    proper_list = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Pz']
    temp = combinations(proper_list, 3)

    l = []
    for i in temp:
        l.append(list(i))

    d = dict()
    # j=[l[0],l[1],l[2]]
    for k in l:
        # Select a classifier
        # print(k)
        test_mi.classifier = mi_classifier(subset=k)  # you can add a subset here

# Select a classifier
    #test_mi.classifier = mi_classifier() # you can add a subset here

# Define the classifier settings
        test_mi.classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)

# Load the xdf
        test_mi.load_offline_eeg_data(filename= r"C:\Users\elyss\PycharmProjects\bci-essentials-python\examples\data\mi_example.xdf", divider=num) # you can also add a subset here

# Run main loop, this will do all of the classification for online or offline
        test_mi.main(online=False, training=True)

        new_test = test_mi.classifier.offline_accuracy
        d[str(k)] = new_test

    num += 0.25



with open('channel3.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

