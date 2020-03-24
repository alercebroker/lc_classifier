import unittest
import os
import sys

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_PATH)

from features import *

if __name__ == "__main__":
    unittest.main()
