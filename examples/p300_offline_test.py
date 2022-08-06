"""
Test P300 offline using data from an existing stream

"""

import os
import sys

# # Add parent directory to path to access bci_essentials
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),os.pardir))

# from src.bci_data import *
from itertools import combinations
import pickle

from bci_essentials.bci_data import *
from bci_essentials.visuals import *

# Initialize the ERP data object
test_erp = ERP_data()

proper_list = ['FC3', 'FCz', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'Pz']
temp = combinations(proper_list, 16)

l=[]
for i in temp:
    l.append(list(i))

d = dict()

for k in l:
# Select a classifier
    print(k)
# Choose a classifier
    test_erp.classifier = erp_rg_classifier(subset= k) # you can add a subset here

    # Set classifier settings
    test_erp.classifier.set_p300_clf_settings(n_splits=5, lico_expansion_factor=1, oversample_ratio=0, undersample_ratio=0)

    # Load the xdf
    test_erp.load_offline_eeg_data(filename = "/Users/maziyardowlat/PycharmProjects/bci-essentials-python3/examples/data/TeamData/April6/sub-P001/ses-S001/eeg/sub-P001_ses-S001_task-SingleFlash_run-001_eeg.xdf", format='xdf') # you can also add a subset here

    # Run main loop, this will do all of the classification for online or offline
    test_erp.main(training=True, online=False, pp_low=0.1, pp_high=10, pp_order=5, plot_erp=False, window_start=0.0, window_end=0.8)

    #print("debug")

    new_test = test_erp.classifier.offline_accuracy
    d[str(k)] = new_test
    # for i in range(len(d.values())):
    #     print(d.keys(),": ", d.values())

with open('sub-P001_ses-S001_task-SingleFlash_run-001_eeg.pickle', 'wb') as handle:
    pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)