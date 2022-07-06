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

# Initialize data object
test_mi = EEG_data()

proper_list = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Pz']
d = dict()

for i in proper_list:
# Select a classifier
    test_mi.classifier = mi_classifier(subset=[i]) # you can add a subset here


    # Define the classifier settings
    test_mi.classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)



    # Load the xdf
    test_mi.load_offline_eeg_data(filename  = "/Users/maziyardowlat/PycharmProjects/bci-essentials-python3/examples/data/mi_example.xdf") # you can also add a subset here

    # Run main loop, this will do all of the classification for online or offline
    test_mi.main(online=False, training=True)

    print("debug")

    new_test = test_mi.classifier.offline_accuracy
    d[i]= new_test
    for i in range(len(d.values())):
        print(d.keys(),": ", d.values())
#get a for lopp, check where the accuracy is stored and find a way to try and access it as a variable if possibke.