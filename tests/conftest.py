"""
conftest.py — Pytest configuration for the test suite.

Adds the project root directory to sys.path so that test files
can import project modules like 'agent', 'store', 'models', etc.
"""

import sys
import os

# Add the project root (one level up from the tests/ folder) to the Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
