import unittest
import sys

if __name__ == '__main__':
    test_suite = unittest.defaultTestLoader.discover('./tests', '*test*.py')
    test_runner = unittest.TextTestRunner(resultclass=unittest.TextTestResult)
    result = test_runner.run(test_suite)
    sys.exit(not result.wasSuccessful())
