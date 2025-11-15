import os
import sys

# extract path of current directory and set as root
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

# ensure to set root dir. if missing
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
