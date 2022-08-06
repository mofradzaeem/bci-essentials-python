"""
Test Motor Imagery (MI) classification offline using data from an existing stream

"""

import os
import sys
# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

# from src.bci_data import *
from bci_essentials.bci_data import *
from itertools import combinations
import pickle
import numpy as np
from bci_essentials.visuals import *

# Initialize data object
test_mi = EEG_data()

proper_list = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Pz']
temp = combinations(proper_list, 16)

l=[]
for i in temp:
    l.append(list(i))

d = dict()
#j=[l[0],l[1],l[2]]
for k in l:
# Select a classifier
    print(k)
    test_mi.classifier = mi_classifier(subset = k)# you can add a subset here


        # Define the classifier settings

    test_mi.classifier.set_mi_classifier_settings(n_splits=3, type="TS", random_seed=35)



        # Load the xdf
    test_mi.load_offline_eeg_data(filename  =
"/Users/maziyardowlat/PycharmProjects/bci-essentials-python3/examples/data/TeamData/May5/BI/MI/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-StandardMIFeetTongue_run-001_eeg.xdf") # you can also add a subset here

        # Run main loop, this will do all of the classification for online or offline
    test_mi.main(online=False, training=True)

    # print("debug")

    new_test = test_mi.classifier.offline_accuracy
    d[str(k)]= new_test
    # for i in range(len(d.values())):
    #     print(d.keys(),": ", d.values())
#get a for lopp, check where the accuracy is stored and find a way to try and access it as a variable if possibke.
with open('May5_StandardMi_feet.pickle', 'wb') as handle:
     pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

