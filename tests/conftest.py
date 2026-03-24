import matplotlib
import os
import sys

# Force a non-interactive backend so tests run in headless environments.
matplotlib.use("Agg")

TESTS_DIR = os.path.dirname(__file__)
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)
