#!/usr/bin/env python3
"""
Heterogeneous Threshold Sensitivity Test (Appendix D)
=====================================================
Wrapper script for fixes/heterogeneous_thresholds.py

Tests compounding vulnerability under three threshold distributions
(uniform, beta-low, beta-high) centered on φ=0.22.

Usage: python heterogeneous_threshold_test.py

Results: fixes/heterogeneous_threshold_results.json
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fixes'))
from heterogeneous_thresholds import *

if __name__ == "__main__":
    print("Running heterogeneous threshold sensitivity test...")
    print("See fixes/heterogeneous_thresholds.py for full implementation")
    print("Results: fixes/heterogeneous_threshold_results.json")
    # Execute the test
    import subprocess
    subprocess.run([sys.executable, os.path.join("fixes", "heterogeneous_thresholds.py")])
