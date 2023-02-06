#Import unittest
import unittest

#Importing our test modules
import test_bci_data
import test_lsl_nework

#initialize the test suite
loader = unittest.TestLoader()
suite = unittest.TestSuite()

#add tests to the suite
suite.addTests(loader.loadTestsFromModule(test_bci_data))
suite.addTests(loader.loadTestsFromModule(test_lsl_nework))

#initialize a runer to pass through the suite
runner = unittest.TextTestRunner(verbosity=3)
result = runner.run(suite)