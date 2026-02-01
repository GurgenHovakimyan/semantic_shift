import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path so we can import mmd
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.mmd import mmd_rbf

def generate_blobs(n_samples=100, dim=768, shift=0.0):
    """
    Generates two Gaussian blobs. 
    'shift' moves the mean of the second blob.
    """
    # Reference distribution (Standard Normal)
    X_ref = torch.randn(n_samples, dim)
    
    # Stream distribution (Shifted)
    X_stream = torch.randn(n_samples, dim) + shift
    
    return X_ref, X_stream

def run_sensitivity_test():
    print("Running Phase 1: Synthetic Sensitivity Test...")
    shifts = [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]
    scores = []
    
    for s in shifts:
        # Generate data
        ref, stream = generate_blobs(n_samples=200, dim=50, shift=s)
        
        # Calculate MMD
        mmd_score = mmd_rbf(ref, stream).item()
        scores.append(mmd_score)
        
        print(f"Shift: {s:.2f} | MMD Score: {mmd_score:.6f}")
        
    # Simple check: Does MMD go up when shift goes up?
    if scores[-1] > scores[0]:
        print("\n✅ SUCCESS: MMD is detecting drift correctly.")
    else:
        print("\n❌ FAILURE: MMD is not detecting drift.")

if __name__ == "__main__":
    run_sensitivity_test()