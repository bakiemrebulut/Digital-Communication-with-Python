import numpy as np
import matplotlib.pyplot as plt


def rcosdesign_custom(beta, span, sps, filter_type='sqrt', debug_level=0):
    """
    Raised Cosine Filter Design
    
    Parameters:
    - beta: rolloff factor (0 <= beta <= 1)
    - span: filter span in symbols
    - sps: samples per symbol
    - filter_type: 'normal' or 'sqrt' for square root raised cosine
    - debug_level: if 1, plots the filter
    
    Returns:
    - h: filter coefficients
    """
    
    if filter_type == "normal":
        delay = span * sps / 2
        t = np.arange(-delay, delay + 1) / sps
        h = np.sinc(t / sps) * np.cos(np.pi * beta * t / sps) / (1 - (2 * beta * t / sps) ** 2)
    
    elif filter_type == "sqrt":
        # Create time vector
        delay = span * sps / 2
        t = np.arange(-delay, delay + 1) / sps
        
        # Initialize filter coefficients
        h = np.zeros(len(t))
        
        # Case 1: t = 0
        idx1 = np.where(t == 0)[0]
        if len(idx1) > 0:
            h[idx1] = -1 / (np.pi * sps) * (np.pi * (beta - 1) - 4 * beta)
        
        # Case 2: |4 * beta * t| = 1 (singularities)
        idx2 = np.where(np.abs(np.abs(4 * beta * t) - 1) < np.finfo(float).eps ** 0.5)[0]
        if len(idx2) > 0:
            h[idx2] = 1 / (2 * np.pi * sps) * (
                np.pi * (beta + 1) * np.sin(np.pi * (beta + 1) / (4 * beta))
                - 4 * beta * np.sin(np.pi * (beta - 1) / (4 * beta))
                + np.pi * (beta - 1) * np.cos(np.pi * (beta - 1) / (4 * beta))
            )
        
        # Case 3: General case (remaining points)
        all_indices = np.arange(len(t))
        ignore_indices = np.concatenate([idx1, idx2])
        ind = np.setdiff1d(all_indices, ignore_indices)
        
        if len(ind) > 0:
            nind = t[ind]
            
            # Main SRRC formula
            numerator = -4 * beta / sps * (
                np.cos((1 + beta) * np.pi * nind) +
                np.sin((1 - beta) * np.pi * nind) / (4 * beta * nind)
            )
            
            denominator = np.pi * ((4 * beta * nind) ** 2 - 1)
            
            h[ind] = numerator / denominator
    
    else:
        raise ValueError("filter_type should be 'normal' or 'sqrt'")
    
    # Normalize filter energy
    h = h / np.sqrt(np.sum(h ** 2))
    
    if debug_level:
        plt.figure()
        plt.stem(h)
        plt.title(f'Square Root Raised Cosine Filter (β = {beta})')
        plt.grid(True)
        plt.show()
    
    return h
